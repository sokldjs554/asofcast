import numpy as np
import pytest

pytest.importorskip('sklearn')


def test_continuation_charges_cost_and_ignores_unavailable_future_actions():
    from asofcast.information_policy import continuation_targets
    gain = np.array([.1, -.1, .2])
    net = np.array([[.2, 5.], [-.2, -.1], [.4, .1]])
    eligible = np.array([[True, False], [True, True], [False, True]])
    np.testing.assert_allclose(continuation_targets(gain, net, eligible), [.3, -.1, .3])


def test_ensemble_learns_to_buy_only_when_expected_gain_exceeds_cost():
    from asofcast.information_policy import GainEnsemble
    rng = np.random.default_rng(42)
    features = rng.normal(size=(480, 3))
    gains = np.column_stack((np.where(features[:, 0] > 0, .4, -.1), np.zeros(480)))
    model = GainEnsemble.fit(features, gains, np.ones_like(gains, dtype=bool),
                            np.arange(480), dict(members=3, iterations=25, leaves=3,
                            min_leaf=12, ridge=1., block_origins=8), seed=42)
    x = np.array([[2., 0., 0.], [-2., 0., 0.]])
    net = model.net(x, np.array([[.03, .01]] * 2), uncertainty=1.)
    assert net[0, 0] > .2
    assert net[1].max() < 0
    assert model.net(x, np.ones((2, 2)), uncertainty=1.).max() < 0


def test_validation_selector_can_retain_simple_policy_and_reject_error_regression():
    from asofcast.information_policy import select_candidate
    simple = {'objective': np.ones(8), 'native_error': np.ones(8)}
    expensive = {'objective': np.ones(8) * 1.01, 'native_error': np.ones(8) * .9}
    worse_error = {'objective': np.ones(8) * .9, 'native_error': np.ones(8) * 1.1}
    assert select_candidate({'simple': simple, 'expensive': expensive, 'worse_error': worse_error}) == 'simple'
    better = {'objective': np.ones(8) * .97, 'native_error': np.ones(8) * .99}
    assert select_candidate({'simple': simple, 'better': better}) == 'better'


def test_arrival_simulation_preserves_distribution_in_grid_units():
    from asofcast.preprocessing import simulate_arrivals
    hourly = simulate_arrivals(np.arange(200) * 3600, 3, 17)
    seconds = simulate_arrivals(np.arange(200) * 10, 3, 17)
    np.testing.assert_allclose(hourly / 3600, seconds / 10)


def test_short_grid_sequential_snapshot_keeps_fixed_future_target():
    from asofcast.research_diagnosis import rollout
    from asofcast.research_learning import SimplePolicy
    from asofcast.timeline import Timeline
    times = np.arange(100) * 10
    values = np.column_stack((times, times * 2.))
    timeline = Timeline(times, values, times[:, None] + np.ones_like(values) * 15, ('a', 'b'))
    class LastValue:
        def predict(self, x):
            return x[:, -1, 0, 0]
    result = rollout(timeline, np.array([50]), dict(lookback=4, horizon=6,
                     waits_seconds=[0, 5, 10]), LastValue(), SimplePolicy(2, 'target'), np.ones(2))
    assert result['wait_seconds'][0] == 10
    assert result['target_times'][0] == 560
    assert result['prediction'][0] == 500
    assert result['cost'][0] == 1


def test_contiguous_measurements_exclude_gaps_and_sentinels_without_imputation():
    import pandas as pd

    from asofcast.information_data import longest_complete_segment
    frame = pd.DataFrame({'a': [1., 2., -9999., 4., 5., 6., 7.]},
                         index=pd.date_range('2020-01-01', periods=7, freq='10min'))
    result = longest_complete_segment(frame, 600)
    np.testing.assert_array_equal(result['a'], [4., 5., 6., 7.])
    assert result.index[0] == frame.index[3]


def test_training_transitions_retain_pulls_and_do_not_offer_them_again():
    from asofcast.information_policy import training_transitions
    from asofcast.timeline import Timeline
    times = np.arange(100) * 3600
    values = np.column_stack((np.arange(100.), np.arange(100.) * 2))
    timeline = Timeline(times, values, times[:, None] + np.ones_like(values) * 7200, ('a', 'b'))
    class LastValue:
        def predict(self, x):
            return x[:, -1, 0, 0]
    parts = training_transitions(timeline, np.array([20]), dict(lookback=4, horizon=3,
                                 waits_seconds=[0, 1800, 3600]), 0, LastValue(), np.ones(2), seed=42)
    assert parts[0]['gains'][0, 0] == 2
    assert not parts[0]['successors'][0]['eligible'][0, 0]
    assert parts[0]['successors'][0]['eligible'][0, 2]
    assert not parts[-2]['eligible'][0, -1]
    assert parts[-2]['successors'][-1] is None


def test_sensor_grid_uses_only_recent_past_measurements():
    import pandas as pd

    from asofcast.information_data import causal_grid_sample
    raw = pd.DataFrame({'S0': [1., 2., 999., 3.]}, index=[0., 9.97, 10.01, 20.4])
    sampled = causal_grid_sample(raw, 10, .1)
    assert list(sampled.index) == [0., 10.]
    np.testing.assert_array_equal(sampled['S0'], [1., 2.])
