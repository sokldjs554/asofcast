from __future__ import annotations

import hashlib
import json
from unittest.mock import MagicMock

import pytest


@pytest.fixture(scope='module')
def verified_bundle(tmp_path_factory):
    from asofcast.data import write_synthetic_csv
    from asofcast.experiment import run_experiment

    root = tmp_path_factory.mktemp('release-identity')
    source = root / 'source.csv'
    write_synthetic_csv(source, n=400, seed=41)
    folder = root / 'bundle'
    run_experiment(source, folder, {'lookback': 12, 'epochs': 1, 'policy_epochs': 1,
                                    'acquisition_epochs': 1}, source_kind='synthetic')
    return folder


def evaluated_identity(folder):
    return {
        'run_id': json.loads((folder / 'report.json').read_text())['run_id'],
        'manifest_sha256': hashlib.sha256((folder / 'manifest.json').read_bytes()).hexdigest(),
        'report_sha256': hashlib.sha256((folder / 'report.json').read_bytes()).hexdigest(),
    }


def passing_evaluation(tmp_path, identity):
    path = tmp_path / 'evaluation.json'
    payload = {'candidate_id': 'candidate-A', 'gate': {'passed': True, 'reasons': []}}
    if identity is not None:
        payload['evaluated_bundle'] = identity
    path.write_text(json.dumps(payload))
    return path


@pytest.mark.parametrize('wrong_field', [None, 'run_id', 'manifest_sha256', 'report_sha256'])
def test_promotion_rejects_missing_or_unrelated_evaluated_bundle(
        tmp_path, verified_bundle, wrong_field):
    from asofcast.release_gate import build_release_decision

    identity = evaluated_identity(verified_bundle)
    if wrong_field is None:
        identity = None
    else:
        identity[wrong_field] = 'another-run' if wrong_field == 'run_id' else '0' * 64
    evaluation = passing_evaluation(tmp_path, identity)
    out = tmp_path / 'decision.json'
    with pytest.raises(ValueError, match='evaluated bundle'):
        build_release_decision(evaluation, out, bundle_dir=verified_bundle)
    assert not out.exists()


def test_matching_evaluated_bundle_can_be_promoted(tmp_path, verified_bundle):
    from asofcast.release_gate import build_release_decision, validate_release_decision

    identity = evaluated_identity(verified_bundle)
    evaluation = passing_evaluation(tmp_path, identity)
    out = tmp_path / 'decision.json'
    result = build_release_decision(evaluation, out, bundle_dir=verified_bundle)
    assert result['status'] == 'promoted'
    assert result['evaluated_bundle'] == identity
    assert validate_release_decision(out)['evaluated_bundle'] == identity


@pytest.mark.parametrize('wrong_field', [None, 'run_id', 'manifest_sha256', 'report_sha256'])
def test_decision_reader_rejects_detached_promotion(tmp_path, verified_bundle, wrong_field):
    from asofcast.release_gate import build_release_decision, validate_release_decision

    identity = evaluated_identity(verified_bundle)
    evaluation = passing_evaluation(tmp_path, identity)
    out = tmp_path / 'decision.json'
    result = build_release_decision(evaluation, out, bundle_dir=verified_bundle)
    result['evaluated_bundle'] = identity
    if wrong_field is None:
        result.pop('evaluated_bundle')
    else:
        result['bundle'][wrong_field] = 'another-run' if wrong_field == 'run_id' else '0' * 64
    out.write_text(json.dumps(result))
    with pytest.raises(ValueError, match='evaluated bundle'):
        validate_release_decision(out)


def test_decision_reader_rejects_non_hex_evaluation_digest(tmp_path):
    from asofcast.release_gate import build_release_decision, validate_release_decision

    evaluation = tmp_path / 'failed.json'
    evaluation.write_text(json.dumps({'candidate_id': 'rejected',
                                      'gate': {'passed': False, 'reasons': ['failed']}}))
    out = tmp_path / 'decision.json'
    result = build_release_decision(evaluation, out)
    result['evaluation_sha256'] = 'z' * 64
    out.write_text(json.dumps(result))
    with pytest.raises(ValueError, match='hash'):
        validate_release_decision(out)


@pytest.mark.parametrize('damage', ['missing', 'corrupt'])
def test_mlflow_rejects_damaged_bundle_before_tracking(
        monkeypatch, tmp_path, verified_bundle, damage):
    import shutil

    from asofcast import mlops

    folder = tmp_path / 'bundle'
    shutil.copytree(verified_bundle, folder)
    weights = folder / 'arrival.pt'
    if damage == 'missing':
        weights.unlink()
    else:
        weights.write_bytes(b'corrupt model')
    tracker = MagicMock()
    monkeypatch.setattr(mlops, '_require_mlflow', lambda: tracker)
    with pytest.raises(ValueError, match='checksum'):
        mlops.log_bundle_mlflow(folder, 'sqlite:///unused.db')
    assert tracker.mock_calls == []


def test_mlflow_accepts_verified_bundle(monkeypatch, verified_bundle):
    from asofcast import mlops

    tracker = MagicMock()
    tracker.set_experiment.return_value.experiment_id = 'test-experiment'
    tracker.start_run.return_value.__enter__.return_value.info.run_id = 'tracked-run'
    monkeypatch.setattr(mlops, '_require_mlflow', lambda: tracker)
    result = mlops.log_bundle_mlflow(verified_bundle, 'sqlite:///unused.db')
    assert result['status'] == 'logged'
    assert result['asofcast_run_id'] == evaluated_identity(verified_bundle)['run_id']
    tracker.log_artifacts.assert_called_once_with(str(verified_bundle), artifact_path='verified_bundle')
