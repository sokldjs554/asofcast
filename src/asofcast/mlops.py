"""Optional MLflow experiment tracking for verified AsOfCast bundles."""
from __future__ import annotations

import json
from pathlib import Path

from asofcast.bundle import load_bundle


def _require_mlflow():
    try:
        import mlflow
    except ModuleNotFoundError as exc:
        raise RuntimeError("mlflow is required; install the 'mlops' extra") from exc
    return mlflow


def selected_policy_metrics(report: dict) -> tuple[str, dict]:
    forecaster = report.get('policy_selection', {}).get('forecaster', 'arrival')
    metric_name = f'{forecaster}_learned'
    return metric_name, report.get('test_metrics', {}).get(metric_name, {})


def log_bundle_mlflow(bundle_dir: Path, tracking_uri: str,
                      experiment_name: str = 'AsOfCast') -> dict:
    """Verify and log an experiment bundle without selecting on test data."""
    mlflow = _require_mlflow()
    bundle_dir = Path(bundle_dir)
    report = load_bundle(bundle_dir).report
    if not report.get('run_id') or 'test_metrics' not in report:
        raise ValueError('invalid experiment report')
    metric_name, learned = selected_policy_metrics(report)
    required_metrics = ('mae', 'rmse', 'mean_wait_seconds')
    if any(key not in learned for key in required_metrics):
        raise ValueError(f'{metric_name} metrics are incomplete')

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=report['run_id']) as active:
        config = report.get('config', {})
        params = {f'cfg.{k}': (json.dumps(v, separators=(',', ':')) if isinstance(v, (list, dict)) else v)
                  for k, v in config.items()}
        params.update({'source.kind': report.get('source', {}).get('kind', 'unknown'),
                       'paper_score_reproduced': report.get('paper_score_reproduced', False),
                       'cloud_deployed': report.get('cloud_deployed', False)})
        mlflow.log_params(params)
        mlflow.log_metrics({f'test.{key}': float(learned[key]) for key in required_metrics})
        mlflow.set_tags({'asofcast.run_id': report['run_id'],
                         'arrival_times': report.get('source', {}).get('arrival_times', 'unknown'),
                         'selection_split': report.get('policy_selection', {}).get('split', 'unknown'),
                         'policy_metric': metric_name})
        mlflow.log_artifacts(str(bundle_dir), artifact_path='verified_bundle')
        run_id = active.info.run_id
    return {'status': 'logged', 'tracking_uri': tracking_uri,
            'experiment_id': experiment.experiment_id, 'mlflow_run_id': run_id,
            'asofcast_run_id': report['run_id'], 'policy_metric': metric_name,
            'artifact_path': 'verified_bundle'}


def log_release_decision_mlflow(decision_path: Path, tracking_uri: str,
                                experiment_name: str = 'AsOfCast Releases') -> dict:
    """Log a validated promotion/rejection decision as a separate governance run."""
    from asofcast.release_gate import validate_release_decision

    mlflow = _require_mlflow()
    decision_path = Path(decision_path)
    decision = validate_release_decision(decision_path)
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=decision['candidate_id']) as active:
        mlflow.log_params({
            'candidate_id': decision['candidate_id'],
            'status': decision['status'],
            'gate_passed': decision['gate_passed'],
            'reason_count': len(decision['reasons']),
        })
        tags = {
            'release.status': decision['status'],
            'release.candidate_id': decision['candidate_id'],
            'release.evaluation_sha256': decision['evaluation_sha256'],
            'release.scope': decision['scope'],
        }
        if 'bundle' in decision:
            tags['release.bundle_run_id'] = decision['bundle']['run_id']
        if 'runtime' in decision:
            tags['release.runtime_backend'] = decision['runtime']['backend']
            tags['release.runtime_sha256'] = decision['runtime']['onnx_sha256']
        mlflow.set_tags(tags)
        mlflow.log_artifact(str(decision_path), artifact_path='release')
        run_id = active.info.run_id
    return {
        'status': 'logged', 'tracking_uri': tracking_uri,
        'experiment_id': experiment.experiment_id, 'mlflow_run_id': run_id,
        'candidate_id': decision['candidate_id'], 'release_status': decision['status'],
    }
