import numpy as np
import pytest


def test_oracle_never_uses_ineligible_action_or_ignores_cost():
    from asofcast.research_diagnosis import oracle_acquisition
    result = oracle_acquisition(
        np.array([0., 0., 0.]), np.array([2., 2., 0.]),
        np.array([[0., 1.], [0., 1.], [0., 0.]]),
        np.array([[False, True], [True, True], [True, True]]),
        np.array([10., 1.]), .2)
    np.testing.assert_allclose(result['objective'], [1.2, 1.2, 0.])
    np.testing.assert_array_equal(result['choice'], [1, 1, -1])


def _timeline():
    from asofcast.timeline import Timeline
    times = np.arange(100) * 3600
    values = np.column_stack([np.arange(100), np.arange(100) * 2.])
    arrivals = times[:, None] + np.full_like(values, 7200.)
    return Timeline(times, values, arrivals, ('a', 'b'))


class LastValue:
    def predict(self, x):
        return x[:, -1, 0, 0].astype(float)


class AcquireWaitCommit:
    def choose(self, x, prediction, steps, acquired, costs):
        return np.where(~acquired[:, 0], 0, np.where(steps == 0, -2, -1))


def test_rollout_preserves_acquisition_target_and_accumulates_cost():
    from asofcast.research_diagnosis import rollout
    cfg = dict(lookback=4, horizon=3, waits_seconds=[0, 1800, 3600])
    result = rollout(_timeline(), np.array([10, 20]), cfg, LastValue(),
                     AcquireWaitCommit(), np.array([2., 3.]))
    np.testing.assert_array_equal(result['actions'][:, :3], [[0, -2, -1]] * 2)
    np.testing.assert_allclose(result['prediction'], [10., 20.])
    np.testing.assert_allclose(result['cost'], [2., 2.])
    np.testing.assert_allclose(result['wait_seconds'], [1800., 1800.])
    np.testing.assert_array_equal(result['target_times'], [13 * 3600, 23 * 3600])


def test_rollout_actions_cannot_read_future_targets():
    from asofcast.research_diagnosis import rollout
    from asofcast.timeline import Timeline
    t = _timeline()
    changed = t.values.copy()
    changed[11:] += 10000
    other = Timeline(t.times, changed, t.arrivals, t.columns)
    cfg = dict(lookback=4, horizon=3, waits_seconds=[0, 1800, 3600])
    left = rollout(t, np.array([10]), cfg, LastValue(), AcquireWaitCommit(), np.ones(2))
    right = rollout(other, np.array([10]), cfg, LastValue(), AcquireWaitCommit(), np.ones(2))
    for key in left:
        np.testing.assert_array_equal(left[key], right[key])


def test_rollout_rejects_duplicate_acquisition():
    from asofcast.research_diagnosis import rollout
    class Repeated:
        def choose(self, x, prediction, steps, acquired, costs):
            return np.zeros(len(x), dtype=int)
    with pytest.raises(ValueError, match='eligible'):
        rollout(_timeline(), np.array([10]), dict(lookback=4, horizon=3,
                waits_seconds=[0, 1800, 3600]), LastValue(), Repeated(), np.ones(2))


def test_value_ablation_is_invariant_to_arrival_metadata():
    from asofcast.research_diagnosis import multiscale_features
    rng = np.random.default_rng(2)
    x = rng.normal(size=(5, 48, 2, 4)).astype(np.float32)
    other = x.copy()
    other[..., 1:] += 25
    np.testing.assert_array_equal(multiscale_features(x, metadata=False),
                                  multiscale_features(other, metadata=False))
    assert not np.array_equal(multiscale_features(x), multiscale_features(other))


def test_gate_requires_practical_effect_in_every_cell():
    from asofcast.research_diagnosis import confirmatory_gate
    comparison = dict(mean=.0005, lower95=.0001, upper95=.0009,
                      seed_improvements=[.0005] * 3, relative_improvement=.005)
    cells = {'new/mixed': {'comparisons': {'legacy_joint': comparison,
                                          'validation_simple': comparison},
                           'mae_regressions': [0.] * 3}}
    gate = confirmatory_gate(cells, ['new/mixed'], min_relative=.01, max_regression=.01)
    assert not gate['passed']
    assert any('practical' in reason for reason in gate['reasons'])


def test_hourly_aggregation_never_assigns_future_values_to_earlier_hour():
    import pandas as pd

    from asofcast.research_data import hourly_complete
    frame = pd.DataFrame({'a': np.arange(13.)},
                         index=pd.date_range('2020-01-01', periods=13, freq='10min'))
    got = hourly_complete(frame)
    assert list(got.index) == [pd.Timestamp('2020-01-01 01:00'), pd.Timestamp('2020-01-01 02:00')]
    np.testing.assert_allclose(got['a'], [3.5, 9.5])
    with pytest.raises(ValueError, match='gap'):
        hourly_complete(frame.drop(frame.index[8]))


def test_source_columns_normalize_tetouan_spacing_without_changing_measurements():
    from asofcast.research_data import canonical_column
    assert canonical_column('Zone 2  Power Consumption') == 'Zone 2 Power Consumption'
    assert canonical_column('Zone 3  Power Consumption ') == 'Zone 3 Power Consumption'


def test_transition_labels_keep_acquisition_after_wait_and_mask_unavailable_actions():
    from asofcast.research_learning import transition_examples
    t = _timeline()
    cfg = dict(lookback=4, horizon=3, waits_seconds=[0, 1800, 3600])
    features, gains, eligible = transition_examples(t, np.array([10]), cfg, 0,
                                                   LastValue(), seed=4)
    assert features.shape[0] == 6  # Empty plus one seeded subset at each wait.
    np.testing.assert_allclose(gains[0, 0], 2.)  # Pull replaces stale 8 by 10; target=13.
    assert eligible[0, 0]
    assert not eligible[-1, -1]  # No wait after deadline.
    assert np.isfinite(gains[eligible]).all()


def test_residual_candidate_selection_can_retain_baseline_without_test_targets():
    from asofcast.research_learning import fit_residual_forecast
    rng = np.random.default_rng(4)
    x = rng.normal(size=(100, 48, 2, 4)).astype(np.float32)
    base = LastValue()
    cfg = dict(leaf_candidates=[3], weights=[0, 1], iterations=4,
               min_samples_leaf=8, l2_regularization=10, loss='absolute_error', early_stopping=False)
    fitted, record = fit_residual_forecast(base, x[:70], base.predict(x[:70]) + 10,
                                          x[70:], base.predict(x[70:]), cfg, seed=2)
    assert record['weight'] == 0
    np.testing.assert_allclose(fitted.predict(x), base.predict(x))
