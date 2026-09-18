import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope='module')
def bundle_dir(tmp_path_factory):
    from asofcast.data import write_synthetic_csv
    from asofcast.experiment import run_experiment
    root = tmp_path_factory.mktemp('service')
    path = root / 'source.csv'
    write_synthetic_csv(path, n=720, seed=9)
    out = root / 'run'
    run_experiment(path, out, {'lookback':12, 'epochs':2, 'policy_epochs':3}, source_kind='synthetic')
    return out


def test_missing_model_is_alive_but_not_ready(tmp_path):
    from asofcast.service import create_app
    with TestClient(create_app(tmp_path / 'missing')) as client:
        assert client.get('/health').status_code == 200
        assert client.get('/ready').status_code == 503
        assert client.get('/api/metadata').status_code == 503


def test_replay_runs_loaded_model_and_marks_synthetic_provenance(bundle_dir):
    from asofcast.service import create_app
    from asofcast.bundle import load_bundle
    from asofcast.timeline import Timeline
    from asofcast.training import predict
    bundle = load_bundle(bundle_dir)
    with TestClient(create_app(bundle_dir)) as client:
        assert client.get('/ready').status_code == 200
        assert 'AsOfCast' in client.get('/').text
        meta = client.get('/api/metadata').json()
        assert meta['source_kind'] == 'synthetic'
        assert meta['serving_forecaster'] == 'staleness_calibrated_dlinear'
        response = client.get('/api/replay', params={'case_id':0, 'scenario':'mixed'})
        assert response.status_code == 200, response.text
        result = response.json()
        assert len(result['steps']) == 3
        assert len({s['target_time'] for s in result['steps']}) == 1
        assert result['ground_truth_scope'] == 'retrospective_only_not_a_policy_input'
        raw = bundle.timeline
        normalized = Timeline(raw.times, bundle.scaler.transform(raw.values), raw.arrivals, raw.columns)
        origin = int(bundle.test_origins[0])
        x = normalized.snapshot(origin, 0, bundle.config['lookback'], bundle.config['horizon']).features()[None]
        target = bundle.manifest['target_channel']
        expected = predict(bundle.calibrated, x)[0] * bundle.scaler.scale[target] + bundle.scaler.mean[target]
        assert result['steps'][0]['prediction'] == pytest.approx(float(expected), abs=1e-5)
        for step in result['steps']:
            assert all(t is None or t <= result['origin_time'] for t in step['latest_source_times'])


def test_replay_invalid_inputs_rejected(bundle_dir):
    from asofcast.service import create_app
    with TestClient(create_app(bundle_dir)) as client:
        assert client.get('/api/replay?case_id=-1').status_code == 422
        assert client.get('/api/replay?case_id=999999').status_code == 404
        assert client.get('/api/replay?scenario=madeup').status_code == 422


def live_payload(bundle_dir):
    from asofcast.bundle import load_bundle
    b = load_bundle(bundle_dir)
    origin = int(b.test_origins[0])
    lo = origin - b.config['lookback'] + 1
    raw = b.timeline
    return {'event_times': raw.times[lo:origin+1].tolist(), 'values':raw.values[lo:origin+1].tolist(),
            'arrival_times': [[float(v) if np.isfinite(v) else None for v in row]
                              for row in raw.arrivals[lo:origin+1]],
            'columns': list(raw.columns), 'wait_seconds':0.0}


def test_online_prediction_requires_only_past_measurements(bundle_dir):
    from asofcast.service import create_app
    payload = live_payload(bundle_dir)
    with TestClient(create_app(bundle_dir)) as client:
        response = client.post('/api/predict', json=payload)
        assert response.status_code == 200, response.text
        result = response.json()
        assert result['target_time'] > payload['event_times'][-1]
        assert 'target_actual' not in result
        assert result['action'] in ('WAIT', 'COMMIT')
        assert np.isfinite(result['prediction'])


def test_online_shape_and_negative_arrival_validation(bundle_dir):
    from asofcast.service import create_app
    payload = live_payload(bundle_dir)
    with TestClient(create_app(bundle_dir)) as client:
        payload['columns'][0] = 'wrong'
        assert client.post('/api/predict', json=payload).status_code == 422
        payload = live_payload(bundle_dir)
        payload['arrival_times'][0][0] = payload['event_times'][0] - 1
        assert client.post('/api/predict', json=payload).status_code == 422
        payload = live_payload(bundle_dir)
        payload['wait_seconds'] = 1700
        assert client.post('/api/predict', json=payload).status_code == 422


def test_model_output_changes_with_available_input_not_static_demo(bundle_dir):
    from asofcast.service import create_app
    payload = live_payload(bundle_dir)
    for row, timestamp in zip(payload['arrival_times'], payload['event_times'], strict=True):
        row[:] = [timestamp] * len(row)
    with TestClient(create_app(bundle_dir)) as client:
        before = client.post('/api/predict', json=payload).json()['prediction']
        for row in payload['values']: row[-1] += 20
        after = client.post('/api/predict', json=payload).json()['prediction']
    assert abs(after - before) > .1


def test_dashboard_assets_are_local_and_served(bundle_dir):
    from asofcast.service import create_app
    with TestClient(create_app(bundle_dir)) as client:
        html = client.get('/').text
        for path, kind in [('/static/app.js', 'javascript'), ('/static/style.css', 'text/css')]:
            assert path in html
            response = client.get(path)
            assert response.status_code == 200
            assert kind in response.headers['content-type']
            assert len(response.content) > 200
        assert '합성 데이터 결과' in html
        assert '실측 ETTh1 검증' in html
        assert '51,894,720' in html
        assert 'ONNX Runtime' in html
        assert 'Render · LIVE' in html


def test_service_uses_recorded_cpu_thread_budget(bundle_dir):
    import torch
    from asofcast.bundle import load_bundle
    from asofcast.service import create_app
    old = torch.get_num_threads()
    budget = load_bundle(bundle_dir).config['cpu_threads']
    try:
        torch.set_num_threads(budget + 1)
        create_app(bundle_dir)
        assert torch.get_num_threads() == budget
    finally:
        torch.set_num_threads(old)
