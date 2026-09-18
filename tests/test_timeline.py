"""Mutation targets: dropping either cutoff, backward filling, or resampling arrivals."""
import numpy as np
import pytest


def make_timeline():
    from asofcast.timeline import Timeline
    times = np.arange(8) * 60
    values = np.column_stack([np.arange(8, dtype=float), np.arange(8, dtype=float) + 10])
    values[3, 0] = 999.0
    values[4, :] = 99999.0
    arrivals = np.broadcast_to(times[:, None], values.shape).copy().astype(float)
    arrivals[3, 0] += 30
    return Timeline(times, values, arrivals, ('a', 'b'))


def test_late_value_visible_only_after_arrival():
    t = make_timeline()
    early = t.snapshot(3, 0, 4, 3)
    late = t.snapshot(3, 30, 4, 3)
    assert early.values[-1, 0] == 2.0
    assert not early.observed[-1, 0]
    assert early.age_seconds[-1, 0] == 60
    assert late.values[-1, 0] == 999.0
    assert late.observed[-1, 0]
    assert late.age_seconds[-1, 0] == 30


def test_future_measurement_never_enters_frozen_origin():
    t = make_timeline()
    snap = t.snapshot(3, 90, 4, 3)
    assert not np.any(snap.values == 99999.0)
    assert snap.source_times[snap.known].max() <= t.times[3]
    assert snap.target_time == t.times[6]
    assert snap.origin_time == t.times[3]


def test_wait_does_not_move_target():
    t = make_timeline()
    assert t.snapshot(3, 0, 4, 3).target_time == t.snapshot(3, 90, 4, 3).target_time


def test_no_backward_fill_of_unknown_prefix():
    t = make_timeline()
    a = t.arrivals.copy()
    a[:2, 0] = np.inf
    from asofcast.timeline import Timeline
    snap = Timeline(t.times, t.values, a, t.columns).snapshot(3, 0, 4, 3)
    assert np.all(snap.values[:2, 0] == 0)
    assert not snap.known[:2, 0].any()
    assert snap.known[2:, 0].all()


def test_real_zero_is_not_missing():
    snap = make_timeline().snapshot(3, 0, 4, 3)
    assert snap.values[0, 0] == 0
    assert snap.known[0, 0]
    assert snap.observed[0, 0]


def test_boundary_inclusive_and_arrival_schedule_stable():
    t = make_timeline()
    a = t.snapshot(3, 30, 4, 3)
    b = t.snapshot(3, 30, 4, 3)
    np.testing.assert_array_equal(a.values, b.values)
    assert a.observed[-1, 0]


@pytest.mark.parametrize('wait,lookback,horizon', [(-1, 4, 3), (180, 4, 3), (0, 0, 3), (0, 5, 3), (0, 4, 0)])
def test_invalid_queries_rejected(wait, lookback, horizon):
    with pytest.raises(ValueError):
        make_timeline().snapshot(3, wait, lookback, horizon)


@pytest.mark.parametrize('fault', ['unsorted', 'duplicate', 'negative_delay', 'nan_value', 'columns', 'irregular', 'nan_arrival'])
def test_invalid_timeline_rejected(fault):
    from asofcast.timeline import Timeline
    t = make_timeline()
    times, values, arrivals, cols = t.times.copy(), t.values.copy(), t.arrivals.copy(), t.columns
    if fault == 'unsorted': times[[0, 1]] = times[[1, 0]]
    if fault == 'duplicate': times[1] = times[0]
    if fault == 'irregular': times[-1] += 1
    if fault == 'negative_delay': arrivals[2, 0] = times[2] - 1
    if fault == 'nan_value': values[2, 0] = np.nan
    if fault == 'nan_arrival': arrivals[2, 0] = np.nan
    if fault == 'columns': cols = ('a', 'a')
    with pytest.raises(ValueError):
        Timeline(times, values, arrivals, cols)


def test_future_values_mutation_cannot_change_snapshot():
    from asofcast.timeline import Timeline
    t = make_timeline()
    changed = t.values.copy()
    changed[4:] *= -10000
    other = Timeline(t.times, changed, t.arrivals, t.columns)
    np.testing.assert_array_equal(t.snapshot(3, 90, 4, 3).values, other.snapshot(3, 90, 4, 3).values)


def test_feature_channels_are_finite_and_distinguish_availability():
    snap = make_timeline().snapshot(3, 0, 4, 3)
    x = snap.features()
    assert x.shape == (4, 2, 4)
    assert x.dtype == np.float32 and np.isfinite(x).all()
    assert x[-1, 0, 1] == 0 and x[-1, 1, 1] == 1
    assert x[-1, 0, 3] == 1


def test_input_arrays_cannot_be_mutated_through_timeline():
    t = make_timeline()
    with pytest.raises(ValueError):
        t.values[0, 0] = 99


def test_online_snapshot_needs_no_future_measurements():
    t = make_timeline()
    snap = t.snapshot(len(t.times) - 1, 0, 4, 6)
    assert snap.target_time == t.times[-1] + 6 * t.grid_seconds


def test_active_acquisition_reveals_only_selected_origin_slot():
    t = make_timeline()
    baseline = t.snapshot(3, 0, 4, 3)
    acquired = t.snapshot(3, 0, 4, 3, acquired_channels=[0])
    assert baseline.values[-1, 0] == 2.0
    assert acquired.values[-1, 0] == 999.0
    assert acquired.observed[-1, 0]
    assert acquired.source_times[-1, 0] == t.times[3]
    assert acquired.values[-1, 1] == baseline.values[-1, 1]


def test_active_acquisition_never_reveals_future_event():
    t = make_timeline()
    acquired = t.snapshot(3, 90, 4, 3, acquired_channels=[0, 1])
    assert not np.any(acquired.values == 99999.0)
    assert acquired.source_times[acquired.known].max() <= t.times[3]
    assert acquired.target_time == t.times[6]


@pytest.mark.parametrize('channels', [[-1], [2], [0, 0], ['a']])
def test_invalid_active_acquisition_channels_rejected(channels):
    with pytest.raises(ValueError):
        make_timeline().snapshot(3, 0, 4, 3, acquired_channels=channels)
