import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def evaluation():
    p = ROOT / "scripts/evaluate_posterior_risk.py"
    assert p.is_file(), "real-data evaluation integration is missing"
    spec = importlib.util.spec_from_file_location("posterior_eval_test", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def protocol():
    return json.loads((ROOT / "configs/posterior_risk_development_20261002.json").read_text())


def test_valid_development_protocol():
    evaluation().validate_protocol(protocol())


@pytest.mark.parametrize(
    "key,value", [("cost_weight", 0.0), ("delay_weight", 0.0), ("deadline_seconds", 7200)]
)
def test_cannot_relax_existing_cost_or_deadline(key, value):
    p = copy.deepcopy(protocol())
    p["objective"][key] = value
    with pytest.raises(ValueError):
        evaluation().validate_protocol(p)


def test_cannot_relabel_exposed_data_as_independent():
    p = protocol()
    p["development_only"] = False
    with pytest.raises(ValueError):
        evaluation().validate_protocol(p)


def test_cannot_relax_gate():
    p = protocol()
    p["gate"]["minimum_relative_improvement"] = 0
    with pytest.raises(ValueError):
        evaluation().validate_protocol(p)


def test_output_path_cannot_be_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        evaluation().evaluate(Path("missing-bundle"), tmp_path, protocol())
