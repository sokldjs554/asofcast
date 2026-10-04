from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient


@pytest.fixture(scope='module')
def runtime_bundle(tmp_path_factory):
    from asofcast.data import write_synthetic_csv
    from asofcast.experiment import run_experiment

    root = tmp_path_factory.mktemp('serving-runtime')
    source = root / 'source.csv'
    write_synthetic_csv(source, n=480, seed=31)
    out = root / 'bundle'
    run_experiment(
        source,
        out,
        {'lookback': 12, 'epochs': 1, 'policy_epochs': 1, 'acquisition_epochs': 1},
        source_kind='synthetic',
    )
    return out


def _install_fake_onnx(monkeypatch, runtime_bundle):
    from asofcast.bundle import load_bundle, select_serving_forecaster
    from asofcast import serving_runtime

    model, _ = select_serving_forecaster(load_bundle(runtime_bundle))

    class Session:
        def get_providers(self):
            return ['CPUExecutionProvider']

        def run(self, names, feeds):
            del names
            with torch.inference_mode():
                return [model(torch.from_numpy(feeds['features'].copy())).cpu().numpy()]

    def fake_export(_model, _sample, path):
        Path(path).write_bytes(b'FAKE-VERIFIED-ONNX')

    monkeypatch.setattr(serving_runtime, '_export_model', fake_export)
    monkeypatch.setattr(serving_runtime, '_create_session', lambda path, threads: Session())
    monkeypatch.setattr(serving_runtime, '_optional_runtime', lambda: type('ORT', (), {'__version__': 'TEST'})())


def test_prepare_runtime_writes_verified_hash_bound_manifest(monkeypatch, runtime_bundle, tmp_path):
    from asofcast.serving_runtime import prepare_serving_runtime

    _install_fake_onnx(monkeypatch, runtime_bundle)
    out = tmp_path / 'runtime'
    result = prepare_serving_runtime(runtime_bundle, out, threads=1)
    manifest = json.loads((out / 'runtime-manifest.json').read_text(encoding='utf-8'))

    assert result == manifest
    assert manifest['status'] == 'verified'
    assert manifest['backend'] == 'onnxruntime'
    assert manifest['bundle_run_id']
    assert manifest['forecaster'] == 'staleness_calibrated_dlinear'
    assert manifest['max_abs_drift_standardized'] <= 1e-4
    assert manifest['input_shape'][0] == 12
    assert manifest['onnx_sha256'] == hashlib.sha256((out / 'forecaster.onnx').read_bytes()).hexdigest()


def test_explicit_onnx_rejects_tampered_model_but_auto_falls_back(monkeypatch, runtime_bundle, tmp_path):
    from asofcast.bundle import load_bundle
    from asofcast.serving_runtime import load_serving_runtime, prepare_serving_runtime

    _install_fake_onnx(monkeypatch, runtime_bundle)
    out = tmp_path / 'runtime'
    prepare_serving_runtime(runtime_bundle, out, threads=1)
    (out / 'forecaster.onnx').write_bytes(b'TAMPERED')
    bundle = load_bundle(runtime_bundle)

    with pytest.raises(RuntimeError, match='checksum'):
        load_serving_runtime(bundle, out, mode='onnx')
    fallback = load_serving_runtime(bundle, out, mode='auto')
    assert fallback.backend == 'torch'
    assert 'checksum' in fallback.fallback_reason


def test_explicit_onnx_rejects_manifest_from_different_bundle(monkeypatch, runtime_bundle, tmp_path):
    from asofcast.bundle import load_bundle
    from asofcast.serving_runtime import load_serving_runtime, prepare_serving_runtime

    _install_fake_onnx(monkeypatch, runtime_bundle)
    out = tmp_path / 'runtime'
    prepare_serving_runtime(runtime_bundle, out, threads=1)
    path = out / 'runtime-manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    manifest['bundle_run_id'] = 'another-run'
    path.write_text(json.dumps(manifest), encoding='utf-8')

    with pytest.raises(RuntimeError, match='run id'):
        load_serving_runtime(load_bundle(runtime_bundle), out, mode='onnx')


def test_service_reports_verified_onnx_backend_and_uses_it(monkeypatch, runtime_bundle, tmp_path):
    from asofcast.service import create_app
    from asofcast.serving_runtime import prepare_serving_runtime

    _install_fake_onnx(monkeypatch, runtime_bundle)
    out = tmp_path / 'runtime'
    manifest = prepare_serving_runtime(runtime_bundle, out, threads=1)

    with TestClient(create_app(runtime_bundle, runtime_dir=out, backend_mode='onnx')) as client:
        ready = client.get('/ready')
        assert ready.status_code == 200, ready.text
        body = ready.json()
        assert body['serving_backend'] == 'onnxruntime'
        assert body['runtime_sha256'] == manifest['onnx_sha256']
        meta = client.get('/api/metadata').json()
        assert meta['serving_backend'] == 'onnxruntime'
        replay = client.get('/api/replay', params={'case_id': 0, 'scenario': 'mixed'})
        assert replay.status_code == 200
        assert np.isfinite(replay.json()['steps'][0]['prediction'])


def test_explicit_onnx_missing_runtime_is_alive_but_not_ready(runtime_bundle, tmp_path):
    from asofcast.service import create_app

    with TestClient(create_app(runtime_bundle, runtime_dir=tmp_path / 'missing', backend_mode='onnx')) as client:
        assert client.get('/health').status_code == 200
        assert client.get('/ready').status_code == 503
        assert client.get('/api/metadata').status_code == 503
