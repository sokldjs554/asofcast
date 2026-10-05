"""Behavioral checks for multi-action value learning; independent toy outcomes."""

import importlib

import numpy as np
import pytest

from asofcast.research_diagnosis import rollout
from asofcast.timeline import Timeline


def implementation():
    assert importlib.util.find_spec("asofcast.research_plan_policy") is not None, (
        "terminal plan policy is missing"
    )
    return importlib.import_module("asofcast.research_plan_policy")


class InteractionForecast:
    def predict(self, x):
        observed = (x[:, -1, :, 1] >= 0.5).sum(1)
        return np.choose(observed, [2.0, 0.0, 10.0])


def fixture(arrival=1000):
    times = np.arange(80) * 100
    values = np.full((80, 2), 10.0)
    arrivals = np.repeat(times[:, None], 2, axis=1).astype(float)
    arrivals[50] += arrival
    return Timeline(times, values, arrivals, ("target", "sensor"))


CFG = {"lookback": 48, "horizon": 6, "waits_seconds": [0, 50, 100]}
SETTINGS = {"n_estimators": 32, "max_depth": 4, "min_samples_leaf": 1}


def examples(timeline):
    return implementation().terminal_plan_examples(
        timeline,
        np.full(32, 50),
        CFG,
        0,
        InteractionForecast(),
        np.array([1.0, 1.0]),
        seed=42,
        weight=0.03,
        delay=0.02,
        augment=True,
    )


def test_joint_gain_survives_harmful_individual_pulls():
    f, g, eligible, plans = examples(fixture())
    row = 0
    assert g[row, plans.index((0, (0,)))] == pytest.approx(-2.03)
    assert g[row, plans.index((0, (0, 1)))] == pytest.approx(7.94)
    policy = implementation().TerminalPlanPolicy.fit(
        f, g, eligible, plans, CFG["waits_seconds"], SETTINGS, seed=42
    )
    result = rollout(fixture(), np.array([50]), CFG, InteractionForecast(), policy, [1.0, 1.0])
    np.testing.assert_array_equal(result["actions"][0, :3], [0, 1, -1])
    assert result["prediction"][0] == 10
    assert result["cost"][0] == 2


def test_future_natural_arrival_is_free_in_training_label():
    f, g, eligible, plans = examples(fixture(70))
    assert g[0, plans.index((2, (0, 1)))] == pytest.approx(7.98)
    assert g[0, plans.index((1, (0, 1)))] == pytest.approx(7.93)


def test_current_inputs_are_unchanged_when_future_values_mutate():
    a = fixture()
    values = a.values.copy()
    values[51:] = 0
    b = Timeline(a.times, values, a.arrivals, a.columns)
    fa, ga, ea, plans = examples(a)
    fb, gb, eb, _ = examples(b)
    np.testing.assert_array_equal(fa, fb)
    np.testing.assert_array_equal(ea, eb)
    assert not np.array_equal(ga, gb)
    policy = implementation().TerminalPlanPolicy.fit(
        fa, ga, ea, plans, CFG["waits_seconds"], SETTINGS, seed=42
    )
    for timeline in (a, b):
        result = rollout(timeline, np.array([50]), CFG, InteractionForecast(), policy, [1.0, 1.0])
        np.testing.assert_array_equal(result["actions"][0, :3], [0, 1, -1])


def test_margin_rejects_unprofitable_gain_and_deadline_never_waits():
    f, g, e, plans = examples(fixture())
    policy = implementation().TerminalPlanPolicy.fit(
        f, g, e, plans, CFG["waits_seconds"], SETTINGS, seed=42
    )
    policy.margin = 100
    result = rollout(fixture(), np.array([50]), CFG, InteractionForecast(), policy, [1.0, 1.0])
    assert result["actions"][0, 0] == -1
    policy.margin = 0
    x = fixture().snapshot(50, 100, 48, 6).features()[None]
    action = policy.choose(x, np.array([2.0]), np.array([2]), np.zeros((1, 2), bool), np.ones(2))
    assert action[0] in (0, 1, -1)


def test_known_sensors_are_never_paid_for():
    f, g, e, plans = examples(fixture(0))
    policy = implementation().TerminalPlanPolicy.fit(
        f, g, e, plans, CFG["waits_seconds"], SETTINGS, seed=42
    )
    result = rollout(fixture(0), np.array([50]), CFG, InteractionForecast(), policy, [1.0, 1.0])
    assert result["cost"][0] == 0
    assert result["actions"][0, 0] == -1


def test_valid_action_regression_ignores_ineligible_rows():
    f, g, e, plans = examples(fixture())
    settings = {
        "algorithm": "hist",
        "max_leaf_nodes": 7,
        "max_iter": 30,
        "min_samples_leaf": 2,
        "l2_regularization": 1.0,
        "max_bins": 32,
        "early_stopping": False,
    }
    corrupted = g.copy()
    corrupted[~e] = 1e10
    left = implementation().TerminalPlanPolicy.fit(
        f, g, e, plans, CFG["waits_seconds"], settings, seed=42
    )
    right = implementation().TerminalPlanPolicy.fit(
        f, corrupted, e, plans, CFG["waits_seconds"], settings, seed=42
    )
    a = rollout(fixture(), np.array([50]), CFG, InteractionForecast(), left, [1.0, 1.0])
    b = rollout(fixture(), np.array([50]), CFG, InteractionForecast(), right, [1.0, 1.0])
    np.testing.assert_array_equal(a["actions"], b["actions"])
    assert a["prediction"][0] == 10
    assert a["acquired_count"][0] == 2
    assert a["wait_seconds"][0] <= 100


def test_runner_rejects_stale_bundle_contract():
    from types import SimpleNamespace

    from scripts import run_terminal_plan_confirmation as runner

    cfg = {
        "lookback": 48,
        "horizon": 6,
        "stride": 4,
        "seed": 42,
        "target": "target",
        "waits_seconds": [0, 50, 100],
    }
    meta = {"grid_seconds": 100, "target": "target", "prepared_sha256": "expected"}
    bundle = SimpleNamespace(
        config=cfg.copy(),
        timeline=SimpleNamespace(grid_seconds=100),
        report={"source": {"sha256": "wrong"}},
        manifest={"target_channel": 0},
    )
    with pytest.raises(ValueError, match="source"):
        runner.validate_bundle_contract(bundle, cfg, meta, 42)
    bundle.report["source"]["sha256"] = "expected"
    bundle.config["seed"] = 43
    with pytest.raises(ValueError, match="config"):
        runner.validate_bundle_contract(bundle, cfg, meta, 42)


def test_consumed_training_configuration_is_frozen(tmp_path, monkeypatch):
    from scripts import run_terminal_plan_confirmation as runner

    c = tmp_path / "configs"
    c.mkdir()
    (c / "terminal_plan_confirmation_20261005.json").write_text("{}")
    (c / "ett_m1.json").write_text('{"epochs":25}')
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    before = runner.source_lock()
    (c / "ett_m1.json").write_text('{"epochs":1}')
    assert runner.source_lock() != before


def test_policy_save_does_not_replace_good_artifact_with_incomplete_stream(tmp_path, monkeypatch):
    from scripts import run_terminal_plan_confirmation as runner

    target = tmp_path / "policy.joblib"
    target.write_bytes(b"existing-good-artifact")

    def interrupted_dump(value, path, **kwargs):
        from pathlib import Path

        Path(path).write_bytes(b"incomplete")

    monkeypatch.setattr(runner.joblib, "dump", interrupted_dump)
    with pytest.raises(ValueError, match="integrity"):
        runner.save_plan_policy(object(), target)
    assert target.read_bytes() == b"existing-good-artifact"


def test_numeric_save_preserves_previous_file_when_writer_is_incomplete(tmp_path, monkeypatch):
    from scripts import run_terminal_plan_confirmation as runner

    target = tmp_path / "arrays.npz"
    target.write_bytes(b"previous-valid-file")

    def interrupted_write(handle, **arrays):
        handle.write(b"incomplete")

    monkeypatch.setattr(runner.np, "savez_compressed", interrupted_write)
    with pytest.raises(ValueError):
        runner.save_numeric_arrays(target, {"values": np.arange(4.0)})
    assert target.read_bytes() == b"previous-valid-file"


def test_rejected_resume_preserves_completed_provenance(tmp_path, monkeypatch):
    import json
    import sys

    from scripts import run_terminal_plan_confirmation as runner

    out = tmp_path / "completed"
    out.mkdir()
    original = {
        "environment.json": b"original environment",
        "selection-lock.json": b"original selection",
    }
    for name, data in original.items():
        (out / name).write_bytes(data)
    incoming = tmp_path / "wrong-lock.json"
    incoming.write_text(json.dumps({"source_lock": {}}))
    monkeypatch.setattr(runner.np, "__version__", "2.3.5")
    monkeypatch.setattr(runner.sklearn, "__version__", "1.8.0")
    monkeypatch.setattr(runner.torch, "__version__", "2.10.0+cpu")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--stage",
            "confirmation",
            "--out",
            str(out),
            "--data",
            str(tmp_path),
            "--selection-lock",
            str(incoming),
        ],
    )
    with pytest.raises(ValueError, match="changed after selection freeze"):
        runner.main()
    for name, data in original.items():
        assert (out / name).read_bytes() == data
