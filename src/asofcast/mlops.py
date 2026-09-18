"""Optional MLflow experiment tracking for verified AsOfCast bundles."""
from __future__ import annotations

import json
from pathlib import Path


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
    """Log one already-verified experiment bundle without selecting on test data."""
    mlflow = _require_mlflow()
    bundle_dir = Path(bundle_dir)
    report_path = bundle_dir / 'report.json'
    if not report_path.is_file():
        raise ValueError('bundle report.json is missing')
    report = json.loads(report_path.read_text(encoding='utf-8'))
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
