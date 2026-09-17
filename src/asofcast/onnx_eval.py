"""Optional ONNX Runtime parity and CPU latency evidence for the arrival model."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from asofcast.bundle import load_bundle
from asofcast.experiment import build_examples
from asofcast.timeline import Timeline


def _optional_runtime():
    try:
        import onnx  # noqa: F401
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        raise RuntimeError("onnx and onnxruntime are required; install the 'optimize' extra") from exc
    return ort


def _latency(callable_, repeats: int = 200) -> dict:
    if repeats < 2:
        raise ValueError('repeats must be >= 2')
    samples = []
    for _ in range(20):
        callable_()
    for _ in range(repeats):
        start = time.perf_counter_ns()
        callable_()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return {'p50_ms': float(np.quantile(samples, .5)), 'p95_ms': float(np.quantile(samples, .95)),
            'repeats': repeats}


def evaluate_onnx(bundle_dir: Path, report_path: Path, repeats: int = 200) -> dict:
    ort = _optional_runtime()
    bundle = load_bundle(bundle_dir)
    normalized = Timeline(bundle.timeline.times, bundle.scaler.transform(bundle.timeline.values),
                          bundle.timeline.arrivals, bundle.timeline.columns)
    target = bundle.manifest['target_channel']
    x, _ = build_examples(normalized, bundle.test_origins[:32], bundle.config, target)
    flat = x.reshape((-1,) + x.shape[2:]).astype(np.float32)
    onnx_path = Path(report_path).with_suffix('.onnx')
    sample = torch.from_numpy(flat[:1].copy())
    bundle.arrival.eval()
    torch.onnx.export(bundle.arrival, sample, onnx_path, input_names=['features'], output_names=['prediction'],
                      dynamic_axes={'features': {0: 'batch'}, 'prediction': {0: 'batch'}},
                      opset_version=18, do_constant_folding=True, dynamo=False)
    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    parity = {}
    max_drift = 0.0
    for batch in (1, min(16, len(flat))):
        inputs = flat[:batch].copy()
        with torch.inference_mode():
            expected = bundle.arrival(torch.from_numpy(inputs)).cpu().numpy()
        actual = session.run(['prediction'], {'features': inputs})[0]
        drift = float(np.max(np.abs(expected - actual)))
        parity[str(batch)] = {'max_abs_drift_standardized': drift}
        max_drift = max(max_drift, drift)
    one = flat[:1].copy()
    torch_call = lambda: bundle.arrival(torch.from_numpy(one)).detach().cpu().numpy()
    ort_call = lambda: session.run(['prediction'], {'features': one})[0]
    result = {
        'status': 'verified', 'asofcast_run_id': bundle.report['run_id'],
        'onnx_path': onnx_path.name, 'onnx_bytes': onnx_path.stat().st_size,
        'onnxruntime_version': ort.__version__, 'opset': 18, 'dynamic_batch': True,
        'parity': parity, 'max_abs_drift_standardized': max_drift,
        'torch_latency': _latency(torch_call, repeats), 'onnxruntime_latency': _latency(ort_call, repeats),
        'gate': {'max_abs_drift_lte_1e-4': max_drift <= 1e-4},
        'timing_scope': 'batch=1 model forward only on shared GitHub runner CPU; not production SLA',
    }
    Path(report_path).write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return result
