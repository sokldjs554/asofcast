"""Fail-closed ONNX output validation and matched CPU inference measurements.

NumPy input -> model execution -> NumPy output is timed for both engines.
This includes Python adapter overhead, not preprocessing, HTTP or production SLA.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from asofcast.bundle import load_bundle, select_serving_forecaster
from asofcast.experiment import build_examples
from asofcast.timeline import Timeline


def _optional_runtime():
    try:
        import onnx  # noqa: F401
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        raise RuntimeError("onnx and onnxruntime are required; install the 'optimize' extra") from exc
    return ort


def _write_report(path: Path, report: dict) -> None:
    """Never leave a previous verified JSON behind after a failed evaluation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n',
                    encoding='utf-8')
    temp.replace(path)


def _summary(samples: list[float]) -> dict:
    if not samples or not np.isfinite(samples).all():
        raise ValueError('latency samples must be nonempty and finite')
    return {'p50_ms': float(np.quantile(samples, .5)),
            'p95_ms': float(np.quantile(samples, .95)), 'repeats': len(samples)}


def _paired_latency(torch_call, ort_call, *, repeats: int, warmup: int, rounds: int) -> dict:
    all_samples = {'torch': [], 'onnxruntime': []}
    records = []
    calls = {'torch': torch_call, 'onnxruntime': ort_call}
    # This matches serving, including forward calls in warmup and timing.
    with torch.inference_mode():
        for _ in range(warmup):
            torch_call()
            ort_call()
        for round_index in range(rounds):
            samples = {'torch': [], 'onnxruntime': []}
            for index in range(repeats):
                order = ('torch', 'onnxruntime') if (index + round_index) % 2 == 0 else ('onnxruntime', 'torch')
                for name in order:
                    start = time.perf_counter_ns()
                    calls[name]()
                    samples[name].append((time.perf_counter_ns() - start) / 1e6)
            records.append({'round': round_index + 1,
                            **{name: _summary(values) for name, values in samples.items()},
                            'samples_ms': samples})
            for name, values in samples.items():
                all_samples[name].extend(values)
    return {'rounds': records, 'torch_latency': _summary(all_samples['torch']),
            'onnxruntime_latency': _summary(all_samples['onnxruntime'])}


def evaluate_onnx(bundle_dir: Path, report_path: Path, repeats: int = 200, *,
                  threads: int = 2, warmup: int = 20, rounds: int = 3) -> dict:
    """Return verified evidence, or save failed evidence and raise RuntimeError.

    Accuracy gates are independent of latency: a fast but incorrect export can
    never produce verified status. Optional-runtime exceptions also fail closed.
    """
    path = Path(report_path)
    result = {'status': 'failed', 'schema_version': 2, 'gate': {'passed': False}}
    previous_threads = torch.get_num_threads()
    try:
        for name, value, lower in [('repeats', repeats, 2), ('threads', threads, 1),
                                   ('warmup', warmup, 1), ('rounds', rounds, 1)]:
            if isinstance(value, bool) or not isinstance(value, int) or value < lower:
                raise ValueError(f'{name} must be an integer >= {lower}')
        path.parent.mkdir(parents=True, exist_ok=True)
        ort = _optional_runtime()
        bundle = load_bundle(bundle_dir)
        torch.set_num_threads(threads)
        normalized = Timeline(bundle.timeline.times, bundle.scaler.transform(bundle.timeline.values),
                              bundle.timeline.arrivals, bundle.timeline.columns)
        x, _ = build_examples(normalized, bundle.test_origins[:32], bundle.config,
                              bundle.manifest['target_channel'])
        flat = x.reshape((-1,) + x.shape[2:]).astype(np.float32)
        if len(flat) < 16 or not np.isfinite(flat).all():
            raise ValueError('at least 16 finite parity examples are required')
        onnx_path = path.with_suffix('.onnx')
        forecaster, forecaster_name = select_serving_forecaster(bundle)
        forecaster.eval()
        result.update({
            'asofcast_run_id': bundle.report['run_id'], 'forecaster': forecaster_name,
            'onnx_path': onnx_path.name, 'opset': 18, 'dynamic_batch': True,
            'measurement': {
                'torch_inference_mode': True, 'torch_grad_enabled': False,
                'intra_op_threads': threads, 'torch_inter_op_threads': torch.get_num_interop_threads(),
                'ort_execution_mode': 'ORT_SEQUENTIAL', 'ort_spinning': False,
                'warmup_per_engine': warmup, 'repeats_per_round': repeats, 'rounds': rounds,
                'order': 'paired calls with alternating first engine in each round',
                'input_output': 'same contiguous NumPy float32 input; NumPy output',
                'batch_size': 1, 'device': 'CPU', 'platform': platform.platform(),
                'machine': platform.machine(), 'logical_cpu_count': os.cpu_count(),
                'python_version': platform.python_version(), 'numpy_version': np.__version__,
                'torch_version': torch.__version__, 'onnxruntime_version': ort.__version__,
            },
            'timing_scope': 'batch=1 NumPy-input to NumPy-output inference, including Python adapters; '
                            'excludes preprocessing and HTTP; local execution host, not production SLA',
            'parity': {},
        })
        with torch.inference_mode():
            torch.onnx.export(forecaster, torch.from_numpy(flat[:1].copy()), onnx_path,
                              input_names=['features'], output_names=['prediction'],
                              dynamic_axes={'features': {0: 'batch'}, 'prediction': {0: 'batch'}},
                              opset_version=18, do_constant_folding=True, dynamo=False)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        options.add_session_config_entry('session.inter_op.allow_spinning', '0')
        session = ort.InferenceSession(str(onnx_path), sess_options=options,
                                       providers=['CPUExecutionProvider'])
        result['providers'] = session.get_providers()
        result['onnx_bytes'] = onnx_path.stat().st_size
        result['onnx_sha256'] = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
        result['parity_input_sha256'] = hashlib.sha256(flat.tobytes()).hexdigest()
        max_drift = 0.0
        # Validate both required batch sizes AND all selected replay examples.
        for label, inputs in [('1', flat[:1].copy()), ('16', flat[:16].copy()),
                              ('all', flat.copy())]:
            with torch.inference_mode():
                expected = forecaster(torch.from_numpy(inputs)).cpu().numpy()
            actual = np.asarray(session.run(['prediction'], {'features': inputs})[0])
            entry = {'batch_size': len(inputs), 'expected_shape': list(expected.shape),
                     'actual_shape': list(actual.shape), 'passed': False}
            result['parity'][label] = entry
            if actual.shape != expected.shape:
                raise ValueError(f'ONNX parity shape mismatch at batch {label}')
            if not np.isfinite(expected).all() or not np.isfinite(actual).all():
                raise ValueError(f'ONNX parity nonfinite output at batch {label}')
            drift = float(np.max(np.abs(expected.astype(np.float64) - actual.astype(np.float64))))
            entry['max_abs_drift_standardized'] = drift
            max_drift = max(max_drift, drift)
            result['max_abs_drift_standardized'] = max_drift
            if drift > 1e-4:
                raise ValueError(f'ONNX parity drift exceeds 1e-4 at batch {label}: {drift}')
            entry['passed'] = True
        result['gate'] = {'passed': True, 'finite_outputs': True, 'matching_shapes': True,
                          'max_abs_drift_lte_1e-4': max_drift <= 1e-4}
        one = flat[:1].copy()
        def torch_call():
            return forecaster(torch.from_numpy(one)).cpu().numpy()
        def ort_call():
            return session.run(['prediction'], {'features': one})[0]
        timings = _paired_latency(torch_call, ort_call, repeats=repeats, warmup=warmup, rounds=rounds)
        result.update(timings)
        result['status'] = 'verified'
        _write_report(path, result)
        return result
    except Exception as exc:
        result['status'] = 'failed'
        result['gate']['passed'] = False
        result['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        _write_report(path, result)
        raise RuntimeError(f'ONNX evaluation failed: {exc}; report: {path}') from exc
    finally:
        torch.set_num_threads(previous_threads)
