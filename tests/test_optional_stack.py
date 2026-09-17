import json
from pathlib import Path

import pytest


def test_mlflow_adapter_reports_missing_dependency(tmp_path):
    from asofcast.mlops import log_bundle_mlflow
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'report.json').write_text(json.dumps({'run_id': 'abc', 'source': {'kind': 'synthetic'}, 'config': {}, 'test_metrics': {'arrival_learned': {'mae': 1.0, 'rmse': 2.0, 'mean_wait_seconds': 3.0}}}), encoding='utf-8')
    try:
        import mlflow  # noqa: F401
    except ModuleNotFoundError:
        with pytest.raises(RuntimeError, match='mlflow'):
            log_bundle_mlflow(bundle, 'file:' + str(tmp_path / 'mlruns'))


def test_large_data_measurement_cell_count():
    from asofcast.large_data import measurement_cells
    assert measurement_cells(140_256, 370) == 51_894_720
    with pytest.raises(ValueError):
        measurement_cells(0, 370)


def test_onnx_adapter_reports_missing_dependency(tmp_path):
    from asofcast.onnx_eval import evaluate_onnx
    try:
        import onnxruntime  # noqa: F401
    except ModuleNotFoundError:
        with pytest.raises(RuntimeError, match='onnx'):
            evaluate_onnx(tmp_path / 'missing', tmp_path / 'out.json')
