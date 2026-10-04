from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return path


def test_failed_evaluation_is_recorded_as_rejected_without_model_bundle(tmp_path):
    from asofcast.release_gate import build_release_decision

    evaluation = _write_json(tmp_path / 'evaluation.json', {
        'schema': 'asofcast.release-evaluation.v1',
        'candidate_id': 'posterior-risk-20261002',
        'general_goal_achieved': False,
        'scope': 'all predeclared new-dataset conditions',
        'gate': {'passed': False, 'reasons': ['one condition failed']},
    })
    out = tmp_path / 'decision.json'
    result = build_release_decision(evaluation, out)

    assert result['status'] == 'rejected'
    assert result['gate_passed'] is False
    assert result['candidate_id'] == 'posterior-risk-20261002'
    assert result['reasons'] == ['one condition failed']
    assert result['evaluation_sha256'] == hashlib.sha256(evaluation.read_bytes()).hexdigest()
    assert 'bundle' not in result


def test_passed_evaluation_requires_verified_bundle(tmp_path):
    from asofcast.release_gate import build_release_decision

    evaluation = _write_json(tmp_path / 'evaluation.json', {
        'schema': 'asofcast.release-evaluation.v1',
        'candidate_id': 'candidate-pass',
        'general_goal_achieved': True,
        'gate': {'passed': True, 'reasons': []},
    })
    with pytest.raises(ValueError, match='bundle'):
        build_release_decision(evaluation, tmp_path / 'decision.json')


def test_promoted_manifest_binds_verified_bundle_and_runtime(monkeypatch, tmp_path):
    from asofcast.data import write_synthetic_csv
    from asofcast.experiment import run_experiment
    from asofcast import serving_runtime
    from asofcast.bundle import load_bundle, select_serving_forecaster
    from asofcast.release_gate import build_release_decision, validate_release_decision
    import torch

    source = tmp_path / 'source.csv'
    write_synthetic_csv(source, n=400, seed=41)
    bundle_dir = tmp_path / 'bundle'
    run_experiment(source, bundle_dir, {'lookback': 12, 'epochs': 1, 'policy_epochs': 1,
                                       'acquisition_epochs': 1}, source_kind='synthetic')
    model, _ = select_serving_forecaster(load_bundle(bundle_dir))

    class Session:
        def get_providers(self): return ['CPUExecutionProvider']
        def run(self, names, feeds):
            del names
            with torch.inference_mode():
                return [model(torch.from_numpy(feeds['features'].copy())).cpu().numpy()]
    monkeypatch.setattr(serving_runtime, '_export_model', lambda m, sample, p: Path(p).write_bytes(b'ONNX'))
    monkeypatch.setattr(serving_runtime, '_create_session', lambda p, threads: Session())
    monkeypatch.setattr(serving_runtime, '_optional_runtime', lambda: type('ORT', (), {'__version__':'TEST'})())
    runtime_dir = tmp_path / 'runtime'
    serving_runtime.prepare_serving_runtime(bundle_dir, runtime_dir, threads=1)

    evaluation = _write_json(tmp_path / 'evaluation.json', {
        'schema': 'asofcast.release-evaluation.v1',
        'candidate_id': 'candidate-pass',
        'general_goal_achieved': True,
        'gate': {'passed': True, 'reasons': []},
    })
    out = tmp_path / 'decision.json'
    result = build_release_decision(evaluation, out, bundle_dir=bundle_dir, runtime_dir=runtime_dir)
    validated = validate_release_decision(out)

    assert validated == result
    assert result['status'] == 'promoted'
    assert result['gate_passed'] is True
    assert result['bundle']['run_id'] == load_bundle(bundle_dir).report['run_id']
    assert result['runtime']['backend'] == 'onnxruntime'
    assert len(result['bundle']['manifest_sha256']) == 64
    assert len(result['runtime']['onnx_sha256']) == 64


def test_inconsistent_goal_and_gate_is_rejected_as_invalid_input(tmp_path):
    from asofcast.release_gate import build_release_decision

    evaluation = _write_json(tmp_path / 'evaluation.json', {
        'schema': 'asofcast.release-evaluation.v1',
        'candidate_id': 'bad-contract',
        'general_goal_achieved': True,
        'gate': {'passed': False, 'reasons': ['failed']},
    })
    with pytest.raises(ValueError, match='inconsistent'):
        build_release_decision(evaluation, tmp_path / 'decision.json')


def test_release_decision_validation_detects_tampering(tmp_path):
    from asofcast.release_gate import build_release_decision, validate_release_decision

    evaluation = _write_json(tmp_path / 'evaluation.json', {
        'schema': 'asofcast.release-evaluation.v1',
        'candidate_id': 'rejected-candidate',
        'general_goal_achieved': False,
        'gate': {'passed': False, 'reasons': ['failed']},
    })
    out = tmp_path / 'decision.json'
    build_release_decision(evaluation, out)
    payload = json.loads(out.read_text())
    payload['status'] = 'promoted'
    out.write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises(ValueError, match='status'):
        validate_release_decision(out)


def test_release_gate_cli_writes_rejected_decision(tmp_path):
    from asofcast.cli import main

    evaluation = _write_json(tmp_path / 'evaluation.json', {
        'schema': 'asofcast.release-evaluation.v1',
        'candidate_id': 'cli-reject',
        'general_goal_achieved': False,
        'gate': {'passed': False, 'reasons': ['predeclared gate failed']},
    })
    out = tmp_path / 'decision.json'
    assert main(['release-gate', '--evaluation', str(evaluation), '--out', str(out)]) == 0
    assert json.loads(out.read_text())['status'] == 'rejected'


def test_mlflow_release_logging_records_governance_tags(monkeypatch, tmp_path):
    from asofcast.release_gate import build_release_decision
    from asofcast import mlops

    evaluation = _write_json(tmp_path / 'evaluation.json', {
        'schema': 'asofcast.release-evaluation.v1',
        'candidate_id': 'mlflow-reject',
        'general_goal_achieved': False,
        'gate': {'passed': False, 'reasons': ['quality gate failed']},
    })
    decision = tmp_path / 'decision.json'
    build_release_decision(evaluation, decision)

    state = {'params': None, 'tags': None, 'artifact': None, 'tracking_uri': None, 'experiment': None}

    class Active:
        info = type('Info', (), {'run_id': 'mlflow-run-1'})()
        def __enter__(self): return self
        def __exit__(self, *args): return False

    class FakeMLflow:
        def set_tracking_uri(self, uri): state['tracking_uri'] = uri
        def set_experiment(self, name):
            state['experiment'] = name
            return type('Experiment', (), {'experiment_id': 'exp-1'})()
        def start_run(self, run_name):
            assert run_name == 'mlflow-reject'
            return Active()
        def log_params(self, params): state['params'] = params
        def set_tags(self, tags): state['tags'] = tags
        def log_artifact(self, path, artifact_path): state['artifact'] = (path, artifact_path)

    monkeypatch.setattr(mlops, '_require_mlflow', lambda: FakeMLflow())
    result = mlops.log_release_decision_mlflow(decision, 'sqlite:///mlflow.db')

    assert result['release_status'] == 'rejected'
    assert state['params']['gate_passed'] is False
    assert state['tags']['release.status'] == 'rejected'
    assert state['tags']['release.candidate_id'] == 'mlflow-reject'
    assert len(state['tags']['release.evaluation_sha256']) == 64
    assert state['artifact'][1] == 'release'


def test_track_release_cli_calls_mlflow_adapter(monkeypatch, tmp_path):
    from asofcast.release_gate import build_release_decision
    from asofcast import mlops
    from asofcast.cli import main

    evaluation = _write_json(tmp_path / 'evaluation.json', {
        'schema': 'asofcast.release-evaluation.v1',
        'candidate_id': 'track-release-cli',
        'general_goal_achieved': False,
        'gate': {'passed': False, 'reasons': ['failed']},
    })
    decision = tmp_path / 'decision.json'
    build_release_decision(evaluation, decision)
    called = {}
    def fake_log(path, tracking_uri, experiment):
        called.update(path=Path(path), tracking_uri=tracking_uri, experiment=experiment)
        return {'status': 'logged', 'release_status': 'rejected'}
    monkeypatch.setattr(mlops, 'log_release_decision_mlflow', fake_log, raising=False)

    assert main(['track-release', '--decision', str(decision), '--tracking-uri', 'file:mlruns',
                 '--experiment', 'Releases']) == 0
    assert called == {'path': decision, 'tracking_uri': 'file:mlruns', 'experiment': 'Releases'}
