import numpy as np
import pytest


def test_scaler_uses_only_arrived_training_values():
    from asofcast.preprocessing import TrainScaler
    values = np.array([[1., 4.], [3., 4.], [10000., 4.], [50000., 4.]])
    arrivals = np.array([[0., 0.], [60., 60.], [1000., 120.], [180., 180.]])
    scaler = TrainScaler.fit(values, arrivals, 150)
    np.testing.assert_allclose(scaler.mean, [2, 4])
    np.testing.assert_allclose(scaler.scale, [1, 1])
    np.testing.assert_allclose(scaler.inverse(scaler.transform(values)), values)
    restored = TrainScaler.from_dict(scaler.to_dict())
    np.testing.assert_array_equal(restored.mean, scaler.mean)


def test_scaler_without_any_arrival_rejected():
    from asofcast.preprocessing import TrainScaler
    with pytest.raises(ValueError):
        TrainScaler.fit(np.ones((2, 2)), np.full((2, 2), np.inf), 10)


def test_splits_have_disjoint_input_and_label_intervals():
    from asofcast.preprocessing import split_origins
    splits = split_origins(1000, 24, 6, 2)
    assert list(splits) == ['train', 'policy', 'validation', 'test']
    for a, b in zip(list(splits.values())[:-1], list(splits.values())[1:], strict=True):
        assert a[-1] + 6 < b[0] - 24 + 1
    for indices in splits.values():
        assert np.diff(indices).min() == 2
    assert splits['test'][-1] + 6 < 1000


def test_too_short_series_and_zero_stride_rejected():
    from asofcast.preprocessing import split_origins
    with pytest.raises(ValueError): split_origins(30, 24, 6, 1)
    with pytest.raises(ValueError): split_origins(1000, 24, 6, 0)


def test_delay_simulation_deterministic_nonnegative_value_independent():
    from asofcast.preprocessing import simulate_arrivals
    times = np.arange(200) * 3600
    a = simulate_arrivals(times, 7, seed=42)
    b = simulate_arrivals(times, 7, seed=42)
    c = simulate_arrivals(times, 7, seed=43)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert np.all(a >= times[:, None])
    assert a.shape == (200, 7)
