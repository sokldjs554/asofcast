"""Verified serving runtime selection for the primary forecast model.

PyTorch remains the training and reference engine. ONNX Runtime may serve the
primary forecaster only after a bundle-bound export passes parity and checksum
validation. Policy and acquisition models remain on their existing PyTorch path.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from asofcast.bundle import Bundle, load_bundle, select_serving_forecaster
from asofcast.experiment import build_examples
from asofcast.timeline import Timeline
from asofcast.training import predict as torch_predict

MANIFEST_NAME = 'runtime-manifest.json'
ONNX_NAME = 'forecaster.onnx'
MAX_PARITY_DRIFT = 1e-4


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _optional_runtime():
    try:
        import onnx  # noqa: F401
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        raise RuntimeError("onnx and onnxruntime are required; install the 'optimize' extra") from exc
    return ort


def _export_model(model: torch.nn.Module, sample: np.ndarray, path: Path) -> None:
    model.eval()
    with torch.inference_mode():
        torch.onnx.export(
            model,
            torch.from_numpy(np.asarray(sample, dtype=np.float32).copy()),
            path,
            input_names=['features'],
            output_names=['prediction'],
            dynamic_axes={'features': {0: 'batch'}, 'prediction': {0: 'batch'}},
            opset_version=18,
            do_constant_folding=True,
            dynamo=False,
        )


def _create_session(path: Path, threads: int):
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError('threads must be a positive integer')
    ort = _optional_runtime()
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry('session.intra_op.allow_spinning', '0')
    options.add_session_config_entry('session.inter_op.allow_spinning', '0')
    return ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])


def _parity_examples(bundle: Bundle, limit: int = 32) -> np.ndarray:
    normalized = Timeline(
        bundle.timeline.times,
        bundle.scaler.transform(bundle.timeline.values),
        bundle.timeline.arrivals,
        bundle.timeline.columns,
    )
    origins = bundle.test_origins[: max(1, min(limit, len(bundle.test_origins)))]
    x, _ = build_examples(normalized, origins, bundle.config, bundle.manifest['target_channel'])
    flat = x.reshape((-1,) + x.shape[2:]).astype(np.float32)
    if len(flat) < 1 or not np.isfinite(flat).all():
        raise ValueError('finite parity examples are required')
    return flat


def _predict_session(session, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim < 2 or len(x) < 1 or not np.isfinite(x).all():
        raise ValueError('invalid ONNX serving input')
    result = np.asarray(session.run(['prediction'], {'features': np.ascontiguousarray(x)})[0])
    if result.shape != (len(x),):
        raise RuntimeError(f'ONNX serving output shape mismatch: {result.shape}')
    if not np.isfinite(result).all():
        raise RuntimeError('ONNX serving output is nonfinite')
    return result.astype(np.float32, copy=False)


@dataclass
class ServingRuntime:
    backend: str
    forecaster_name: str
    bundle_run_id: str
    runtime_sha256: str | None
    fallback_reason: str | None
    _predict: Callable[[np.ndarray], np.ndarray]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._predict(x)

    def metadata(self) -> dict:
        return {
            'serving_backend': self.backend,
            'serving_forecaster': self.forecaster_name,
            'runtime_sha256': self.runtime_sha256,
            'runtime_fallback_reason': self.fallback_reason,
        }


def _torch_runtime(bundle: Bundle, *, reason: str | None = None) -> ServingRuntime:
    model, name = select_serving_forecaster(bundle)
    return ServingRuntime(
        backend='torch',
        forecaster_name=name,
        bundle_run_id=bundle.report['run_id'],
        runtime_sha256=None,
        fallback_reason=reason,
        _predict=lambda x: torch_predict(model, x),
    )


def prepare_serving_runtime(bundle_dir: Path, out_dir: Path, *, threads: int = 2,
                            parity_limit: int = 32,
                            max_abs_drift: float = MAX_PARITY_DRIFT) -> dict:
    """Export and verify an ONNX runtime bound to one verified model bundle."""
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError('threads must be a positive integer')
    if isinstance(parity_limit, bool) or not isinstance(parity_limit, int) or parity_limit < 1:
        raise ValueError('parity_limit must be a positive integer')
    if not np.isfinite(max_abs_drift) or max_abs_drift <= 0:
        raise ValueError('max_abs_drift must be positive and finite')

    bundle = load_bundle(bundle_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    onnx_path = out / ONNX_NAME
    manifest_path = out / MANIFEST_NAME
    if onnx_path.exists() or manifest_path.exists():
        raise ValueError('runtime output already exists; use a fresh directory')

    examples = _parity_examples(bundle, parity_limit)
    model, forecaster_name = select_serving_forecaster(bundle)
    _export_model(model, examples[:1], onnx_path)
    if not onnx_path.is_file() or onnx_path.stat().st_size < 1:
        raise RuntimeError('ONNX export did not produce a model file')
    session = _create_session(onnx_path, threads)

    checks = []
    max_drift = 0.0
    for count in sorted({1, min(16, len(examples)), len(examples)}):
        inputs = examples[:count].copy()
        expected = torch_predict(model, inputs)
        actual = _predict_session(session, inputs)
        if expected.shape != actual.shape:
            raise RuntimeError(f'ONNX parity shape mismatch at batch {count}')
        drift = float(np.max(np.abs(expected.astype(np.float64) - actual.astype(np.float64))))
        max_drift = max(max_drift, drift)
        checks.append({'batch_size': count, 'max_abs_drift_standardized': drift})
        if drift > max_abs_drift:
            raise RuntimeError(f'ONNX parity drift exceeds {max_abs_drift}: {drift}')

    try:
        ort_version = _optional_runtime().__version__
    except AttributeError:
        ort_version = 'unknown'
    manifest = {
        'schema_version': 1,
        'status': 'verified',
        'backend': 'onnxruntime',
        'bundle_run_id': bundle.report['run_id'],
        'forecaster': forecaster_name,
        'onnx_file': ONNX_NAME,
        'onnx_sha256': _sha256(onnx_path),
        'input_shape': list(examples.shape[1:]),
        'output_shape': ['batch'],
        'opset': 18,
        'threads': threads,
        'max_abs_drift_standardized': max_drift,
        'max_abs_drift_allowed': float(max_abs_drift),
        'parity_checks': checks,
        'providers': list(session.get_providers()),
        'torch_version': torch.__version__,
        'onnxruntime_version': ort_version,
        'scope': 'primary serving forecaster only; policy and acquisition models remain PyTorch',
    }
    temp = manifest_path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(manifest_path)
    return manifest


def _load_manifest(bundle: Bundle, runtime_dir: Path) -> tuple[dict, Path]:
    root = Path(runtime_dir)
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise RuntimeError('verified ONNX runtime manifest is missing')
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f'invalid ONNX runtime manifest: {exc}') from exc
    if manifest.get('schema_version') != 1 or manifest.get('status') != 'verified':
        raise RuntimeError('ONNX runtime manifest is not verified')
    if manifest.get('backend') != 'onnxruntime':
        raise RuntimeError('ONNX runtime backend contract mismatch')
    if manifest.get('bundle_run_id') != bundle.report['run_id']:
        raise RuntimeError('ONNX runtime run id does not match the model bundle')
    _, expected_name = select_serving_forecaster(bundle)
    if manifest.get('forecaster') != expected_name:
        raise RuntimeError('ONNX runtime forecaster does not match the model bundle')
    filename = manifest.get('onnx_file')
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise RuntimeError('unsafe ONNX runtime file path')
    model_path = root / filename
    if not model_path.is_file():
        raise RuntimeError('ONNX runtime model file is missing')
    expected_hash = manifest.get('onnx_sha256')
    if not isinstance(expected_hash, str) or _sha256(model_path) != expected_hash:
        raise RuntimeError('ONNX runtime checksum mismatch')
    allowed = manifest.get('max_abs_drift_allowed')
    drift = manifest.get('max_abs_drift_standardized')
    if (not isinstance(allowed, (int, float)) or not isinstance(drift, (int, float))
            or not np.isfinite([allowed, drift]).all() or allowed <= 0 or drift > allowed
            or allowed > MAX_PARITY_DRIFT):
        raise RuntimeError('ONNX runtime parity contract is invalid')
    expected_shape = [bundle.config['lookback'], len(bundle.timeline.columns), 4]
    if manifest.get('input_shape') != expected_shape:
        raise RuntimeError('ONNX runtime input shape does not match the model bundle')
    return manifest, model_path


def _verified_onnx_runtime(bundle: Bundle, runtime_dir: Path) -> ServingRuntime:
    manifest, model_path = _load_manifest(bundle, runtime_dir)
    session = _create_session(model_path, int(manifest['threads']))
    examples = _parity_examples(bundle, limit=4)
    model, name = select_serving_forecaster(bundle)
    expected = torch_predict(model, examples)
    actual = _predict_session(session, examples)
    drift = float(np.max(np.abs(expected.astype(np.float64) - actual.astype(np.float64))))
    if drift > float(manifest['max_abs_drift_allowed']):
        raise RuntimeError(f'ONNX runtime startup parity drift exceeds manifest limit: {drift}')
    return ServingRuntime(
        backend='onnxruntime',
        forecaster_name=name,
        bundle_run_id=bundle.report['run_id'],
        runtime_sha256=manifest['onnx_sha256'],
        fallback_reason=None,
        _predict=lambda x: _predict_session(session, x),
    )


def load_serving_runtime(bundle: Bundle, runtime_dir: Path | None, *, mode: str = 'auto') -> ServingRuntime:
    """Load the requested serving backend with explicit fail-closed semantics."""
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {'auto', 'torch', 'onnx'}:
        raise ValueError('serving backend must be one of auto, torch, onnx')
    if normalized_mode == 'torch':
        return _torch_runtime(bundle)
    try:
        if runtime_dir is None:
            raise RuntimeError('verified ONNX runtime directory is not configured')
        return _verified_onnx_runtime(bundle, Path(runtime_dir))
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        if normalized_mode == 'onnx':
            raise RuntimeError(str(exc)) from exc
        return _torch_runtime(bundle, reason=str(exc))
