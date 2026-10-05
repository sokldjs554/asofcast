"""Independent behavioral contracts for calibrated whole-plan decisions."""

import importlib

import numpy as np
import pytest

from asofcast.timeline import Timeline


def impl():
    assert importlib.util.find_spec("asofcast.research_safe_plan") is not None, (
        "safe plan module missing"
    )
    return importlib.import_module("asofcast.research_safe_plan")


class PairForecast:
    def predict(self, x):
        return np.choose((x[:, -1, :, 1] >= 0.5).sum(1), [2.0, 0.0, 10.0])


def timeline(arrival=1000):
    times = np.arange(80) * 100
    values = np.full((80, 2), 10.0)
    arrivals = np.repeat(times[:, None], 2, 1).astype(float)
    arrivals[50] += arrival
    return Timeline(times, values, arrivals, ("target", "sensor"))


CFG = {"lookback": 48, "horizon": 6, "waits_seconds": [0, 50, 100]}


def test_complete_plan_does_not_drop_second_sensor_after_harmful_first_pull():
    result = impl().execute_plans(
        timeline(), np.array([50]), CFG, PairForecast(), np.ones(2), [(0, (0, 1))]
    )
    np.testing.assert_array_equal(result["actions"][0, :3], [0, 1, -1])
    assert result["prediction"][0] == 10
    assert result["cost"][0] == 2
    assert result["acquired_count"][0] == 2


def test_whole_plan_waits_once_per_step_and_does_not_pay_natural_arrivals():
    result = impl().execute_plans(
        timeline(70), np.array([50]), CFG, PairForecast(), np.ones(2), [(2, (0, 1))]
    )
    np.testing.assert_array_equal(result["actions"][0, :3], [-2, -2, -1])
    assert result["cost"][0] == 0
    assert result["acquired_count"][0] == 0
    assert result["wait_seconds"][0] == 100
    assert result["prediction"][0] == 10


def test_illegal_plans_are_rejected():
    for plan in [(3, ()), (0, (0, 0)), (0, (2,))]:
        with pytest.raises(ValueError, match="plan"):
            impl().execute_plans(
                timeline(), np.array([50]), CFG, PairForecast(), np.ones(2), [plan]
            )


def test_initial_features_do_not_reveal_future_or_terminal_observations():
    a = timeline()
    values = a.values.copy()
    values[51:] = 12345
    b = Timeline(a.times, values, a.arrivals, a.columns)
    fa = impl().initial_features(a, np.array([50]), CFG, PairForecast())
    fb = impl().initial_features(b, np.array([50]), CFG, PairForecast())
    np.testing.assert_array_equal(fa, fb)
    values = a.values.copy()
    values[50] = 777
    hidden = Timeline(a.times, values, a.arrivals, a.columns)
    np.testing.assert_array_equal(
        fa, impl().initial_features(hidden, np.array([50]), CFG, PairForecast())
    )


def test_honest_calibration_rejects_optimistic_plan_on_separate_origins():
    f = np.zeros((120, 2))
    g = np.column_stack((np.zeros(120), np.r_[np.full(72, 0.3), np.full(48, -0.2)]))
    origins = np.arange(120) * 4
    router = impl().HonestPlanRouter.fit(
        f,
        g,
        origins,
        {
            "max_depth": 2,
            "min_samples_leaf": 8,
            "fit_fraction": 0.6,
            "purge_observations": 0,
            "minimum_calibration_origins": 8,
        },
        seed=42,
    )
    choice = router.select(np.zeros((5, 2)), strength=0)
    np.testing.assert_array_equal(choice, np.zeros(5, dtype=int))
    assert router.diagnostics["fitting_origin_max"] < router.diagnostics["calibration_origin_min"]


def test_honest_calibration_keeps_supported_positive_plan():
    f = np.zeros((120, 2))
    g = np.column_stack((np.zeros(120), np.full(120, 0.2)))
    origins = np.arange(120) * 4
    router = impl().HonestPlanRouter.fit(
        f,
        g,
        origins,
        {
            "max_depth": 2,
            "min_samples_leaf": 8,
            "fit_fraction": 0.6,
            "purge_observations": 0,
            "minimum_calibration_origins": 8,
        },
        seed=42,
    )
    np.testing.assert_array_equal(
        router.select(np.zeros((5, 2)), strength=1), np.ones(5, dtype=int)
    )


def test_calibration_cannot_count_multiple_views_as_independent_origins():
    f = np.zeros((240, 2))
    g = np.column_stack((np.zeros(240), np.full(240, 0.2)))
    origins = np.repeat(np.arange(12) * 4, 20)
    router = impl().HonestPlanRouter.fit(
        f,
        g,
        origins,
        {
            "max_depth": 2,
            "min_samples_leaf": 8,
            "fit_fraction": 0.6,
            "purge_observations": 0,
            "minimum_calibration_origins": 8,
        },
        seed=42,
    )
    np.testing.assert_array_equal(
        router.select(np.zeros((2, 2)), strength=0), np.zeros(2, dtype=int)
    )


def test_routed_execution_uses_only_selected_complete_strategy():
    from asofcast.research_learning import SimplePolicy

    bank = impl().strategy_bank(2)
    selected = np.array([bank.index(("plan", 0, (0, 1))), 0])
    result = impl().execute_strategies(
        timeline(),
        np.array([50, 51]),
        CFG,
        PairForecast(),
        np.ones(2),
        bank,
        selected,
        SimplePolicy(),
    )
    np.testing.assert_array_equal(result["prediction"], [10.0, 10.0])
    np.testing.assert_array_equal(result["cost"], [2.0, 0.0])
    np.testing.assert_array_equal(result["actions"][0, :3], [0, 1, -1])
    assert result["actions"][1, 0] == -1


def test_optimistic_in_sample_router_is_preserved_only_as_ablation():
    f = np.zeros((120, 2))
    g = np.column_stack((np.zeros(120), np.r_[np.full(72, 0.3), np.full(48, -0.2)]))
    router = impl().HonestPlanRouter.fit(
        f,
        g,
        np.arange(120) * 4,
        {
            "max_depth": 2,
            "min_samples_leaf": 8,
            "fit_fraction": 0.6,
            "purge_observations": 0,
            "minimum_calibration_origins": 8,
        },
        seed=42,
    )
    np.testing.assert_array_equal(router.raw_select(np.zeros((2, 2))), np.ones(2, dtype=int))
    np.testing.assert_array_equal(router.select(np.zeros((2, 2))), np.zeros(2, dtype=int))


def test_sparse_calibration_evidence_is_json_serializable_and_rejects_action():
    import json

    f = np.zeros((30, 2))
    g = np.column_stack((np.zeros(30), np.full(30, 0.2)))
    router = impl().HonestPlanRouter.fit(
        f,
        g,
        np.arange(30) * 4,
        {
            "max_depth": 2,
            "min_samples_leaf": 8,
            "fit_fraction": 0.8,
            "purge_observations": 0,
            "minimum_calibration_origins": 12,
        },
        seed=42,
    )
    json.dumps(router.diagnostics, allow_nan=False)
    np.testing.assert_array_equal(
        router.select(np.zeros((2, 2)), strength=1), np.zeros(2, dtype=int)
    )


def test_value_tree_optimizes_action_reward_instead_of_dominated_action_variance():
    f = np.tile(np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]), (40, 1))
    g = np.column_stack(
        (
            np.zeros(160),
            np.where(f[:, 0] < 0, 1.0, -1.0),
            np.where(f[:, 0] > 0, 1.0, -1.0),
            -1000.0 + 100 * f[:, 1],
        )
    )
    router = impl().HonestPlanRouter.fit(
        f,
        g,
        np.arange(160) * 4,
        {
            "algorithm": "value_tree",
            "max_depth": 1,
            "min_samples_leaf": 8,
            "fit_fraction": 0.6,
            "purge_observations": 0,
            "minimum_calibration_origins": 8,
        },
        seed=42,
    )
    np.testing.assert_array_equal(router.select(np.array([[-1.0, 0.0], [1.0, 0.0]])), [1, 2])


def test_mae_constraint_can_change_plan_without_changing_reported_objective():
    f = np.zeros((120, 2))
    g = np.tile([0.0, 0.2, 0.1], (120, 1))
    mae = np.tile([0.0, -0.3, 0.05], (120, 1))
    settings = {
        "algorithm": "value_tree",
        "max_depth": 2,
        "min_samples_leaf": 8,
        "fit_fraction": 0.6,
        "purge_observations": 0,
        "minimum_calibration_origins": 8,
        "mae_priority": 1.0,
    }
    router = impl().HonestPlanRouter.fit(f, g, np.arange(120) * 4, settings, seed=42, mae_gains=mae)
    np.testing.assert_array_equal(router.select(np.zeros((2, 2))), [2, 2])
    assert next(iter(router.leaf_records.values()))["mean"] == pytest.approx(0.1)


def test_ensemble_winner_is_calibrated_on_separate_origins():
    f = np.zeros((120, 2))
    g = np.column_stack((np.zeros(120), np.r_[np.full(72, 0.3), np.full(48, -0.2)]))
    settings = {
        "n_estimators": 16,
        "max_depth": 3,
        "min_samples_leaf": 2,
        "fit_fraction": 0.6,
        "purge_observations": 0,
        "minimum_calibration_origins": 8,
        "calibration_bins": 3,
    }
    router = impl().HonestGainRouter.fit(f, g, np.arange(120) * 4, settings, seed=42)
    np.testing.assert_array_equal(router.select(np.zeros((2, 2))), [0, 0])


def test_ensemble_can_keep_a_calibrated_profitable_plan():
    f = np.zeros((120, 2))
    g = np.column_stack((np.zeros(120), np.full(120, 0.2)))
    settings = {
        "n_estimators": 16,
        "max_depth": 3,
        "min_samples_leaf": 2,
        "fit_fraction": 0.6,
        "purge_observations": 0,
        "minimum_calibration_origins": 8,
        "calibration_bins": 3,
    }
    router = impl().HonestGainRouter.fit(f, g, np.arange(120) * 4, settings, seed=42)
    np.testing.assert_array_equal(router.select(np.zeros((2, 2))), [1, 1])


def test_training_labels_must_mature_in_the_actual_scenario():
    raw = timeline(0)
    arrivals = raw.arrivals.copy()
    arrivals[56, 0] = np.inf
    arrivals[57, 0] = 9000
    scenario = Timeline(raw.times, raw.values, arrivals, raw.columns)
    origins = np.array([50, 51, 52])
    np.testing.assert_array_equal(impl().mature_policy_origins(raw, origins, CFG, 0, 7900), origins)
    np.testing.assert_array_equal(
        impl().mature_policy_origins(scenario, origins, CFG, 0, 7900), [52]
    )


def test_value_tree_separates_adjacent_float32_features_consistently():
    low = np.nextafter(np.float32(1), np.float32(2))
    high = np.nextafter(low, np.float32(2))
    features = np.tile(np.array([[low], [high]], dtype=np.float32), (80, 1))
    gains = np.tile([[0.0, 1.0, -1.0], [0.0, -1.0, 1.0]], (80, 1))
    router = impl().HonestPlanRouter.fit(
        features,
        gains,
        np.arange(160) * 4,
        {
            "algorithm": "value_tree",
            "max_depth": 1,
            "min_samples_leaf": 8,
            "fit_fraction": 0.6,
            "purge_observations": 0,
            "minimum_calibration_origins": 8,
        },
        seed=42,
    )
    np.testing.assert_array_equal(
        router.select(np.array([[low], [high]], dtype=np.float32)), [1, 2]
    )
    assert router.select(np.array([[np.nextafter(high, np.float32(2))]], dtype=np.float32))[0] == 2


@pytest.mark.parametrize("priority", [-1.0, float("nan"), float("inf")])
def test_ensemble_rejects_invalid_mae_priority(priority):
    with pytest.raises(ValueError, match="priority"):
        impl().HonestGainRouter.fit(
            np.zeros((120, 2)),
            np.tile([0.0, 0.2], (120, 1)),
            np.arange(120) * 4,
            {
                "algorithm": "extra",
                "n_estimators": 8,
                "max_depth": 2,
                "min_samples_leaf": 8,
                "mae_priority": priority,
                "purge_observations": 0,
            },
            seed=42,
            mae_gains=np.tile([0.0, 0.1], (120, 1)),
        )
