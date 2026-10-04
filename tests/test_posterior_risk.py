"""Recovered model/policy contracts plus independent cutoff and Gaussian checks."""

import numpy as np
import pytest

from asofcast.posterior_policy import ArrivalSurvival, PosteriorPolicy, best_sensor_subset
from asofcast.posterior_risk import GaussianForecast


def model():
    return GaussianForecast(
        np.array([[0.8]]),
        np.array([[0.36]]),
        np.array([2.0]),
        np.array([[1.0]]),
        2,
        0,
        np.array([0.0]),
    )


def snapshot(v, o):
    v, o = np.asarray(v, float), np.asarray(o, bool)
    return np.stack([v, o.astype(float), np.zeros_like(v), np.ones_like(v)], axis=-1)


def test_scalar_unobserved_closed_form():
    p = model().moments(snapshot(np.zeros((2, 4, 1)), np.zeros((2, 4, 1), bool)))
    np.testing.assert_allclose(p.mean, [2, 2], atol=1e-12)
    np.testing.assert_allclose(p.variance, [1, 1], atol=1e-12)
    np.testing.assert_allclose(p.cross, [[0.64], [0.64]], atol=1e-12)


def test_origin_observation_closed_form():
    v = np.zeros((1, 4, 1))
    o = np.zeros_like(v, bool)
    v[0, -1, 0] = 5
    o[0, -1, 0] = True
    p = model().moments(snapshot(v, o))
    np.testing.assert_allclose(p.mean, [3.92], atol=1e-12)
    np.testing.assert_allclose(p.variance, [0.5904], atol=1e-12)
    np.testing.assert_allclose(p.cross, [[0]], atol=1e-12)


def test_hidden_measurements_cannot_affect_moments():
    x = snapshot(np.ones((2, 4, 1)), np.zeros((2, 4, 1), bool))
    x[:, 0, 0, 1] = 1
    y = x.copy()
    y[..., 0][x[..., 1] == 0] = 1e20
    a, b = model().moments(x), model().moments(y)
    np.testing.assert_array_equal(a.mean, b.mean)
    np.testing.assert_array_equal(a.variance, b.variance)


def test_forecast_batch_invariance():
    r = np.random.default_rng(42)
    x = snapshot(r.normal(size=(5, 8, 1)), r.random((5, 8, 1)) > 0.5)
    m = model()
    p = m.predict(x)
    np.testing.assert_allclose(m.predict(x[::-1]), p[::-1], atol=1e-12, rtol=0)
    for i in range(5):
        np.testing.assert_allclose(m.predict(x[i : i + 1]), p[i : i + 1], atol=1e-12, rtol=0)


@pytest.mark.parametrize("bad", [np.array([[np.nan]]), np.array([[-1.0]]), np.array([[np.inf]])])
def test_invalid_process_covariance(bad):
    with pytest.raises(ValueError):
        GaussianForecast(
            np.array([[0.8]]), bad, np.array([0.0]), np.array([[1.0]]), 2, 0, np.array([0.0])
        )


def training_values():
    r = np.random.default_rng(17)
    v = np.zeros((350, 2))
    for t in range(1, len(v)):
        v[t] = 0.8 * v[t - 1] + r.normal(size=2)
    return v


def test_fit_repeatability_and_stability():
    v = training_values()
    a = GaussianForecast.fit(v[:250], order=2, ridge=10, horizon=6, target=1, seed=42)
    b = GaussianForecast.fit(v[:250], order=2, ridge=10, horizon=6, target=1, seed=42)
    np.testing.assert_array_equal(a.transition, b.transition)
    assert max(abs(np.linalg.eigvals(a.transition))) < 1
    assert np.linalg.eigvalsh(a.noise).min() >= -1e-12 and a.training_rows <= 248
    assert np.isfinite(a.predict(snapshot(v[None, 250:298], np.ones((1, 48, 2), bool)))).all()


def test_model_roundtrip(tmp_path):
    m = model()
    p = tmp_path / "model.npz"
    m.save(p)
    b = GaussianForecast.load(p)
    x = snapshot(np.ones((2, 4, 1)), np.ones((2, 4, 1), bool))
    np.testing.assert_array_equal(m.predict(x), b.predict(x))
    with pytest.raises(FileExistsError):
        m.save(p)


def test_bad_snapshot_rejected():
    for x in [np.ones((3, 4, 1, 3)), np.ones((3, 4, 2, 4)), np.full((1, 4, 1, 4), np.nan)]:
        with pytest.raises(ValueError):
            model().predict(x)


def test_conditional_sensor_risk_scalar():
    loss, mask = best_sensor_subset(
        np.array([1.0]),
        np.array([[[1.0]]]),
        np.array([[0.64]]),
        np.array([[True]]),
        np.array([0.2]),
        np.array([0.0]),
    )
    np.testing.assert_allclose(
        loss, [np.sqrt(2 / np.pi) * np.sqrt(1 - 0.64**2) + 0.006], atol=1e-12
    )
    np.testing.assert_array_equal(mask, [[True]])


def test_complementary_sensors_full_subset():
    loss, mask = best_sensor_subset(
        np.array([3.0]),
        np.array([[[1.0, 0.99], [0.99, 1.0]]]),
        np.array([[0.1, -0.1]]),
        np.array([[True, True]]),
        np.ones(2),
        np.zeros(2),
        weight=0.01,
    )
    np.testing.assert_array_equal(mask, [[True, True]])
    np.testing.assert_allclose(loss, [np.sqrt(2 / np.pi) + 0.02], atol=1e-10)


def test_ineligible_or_expensive_sensors_not_acquired():
    for eligible, cost in [
        (np.array([[False]]), np.array([0.0])),
        (np.array([[True]]), np.array([1e6])),
    ]:
        loss, mask = best_sensor_subset(
            np.array([1.0]), np.array([[[1.0]]]), np.array([[0.5]]), eligible, cost, np.zeros(1)
        )
        assert not mask.any()
        np.testing.assert_allclose(loss, [np.sqrt(2 / np.pi)])


def test_arrival_training_cutoff_censors_future_information():
    t = np.arange(150) * 3600
    a = t[:, None] + np.array([[1800.0, 7200.0]])
    x = ArrivalSurvival.fit(t, a, cutoff=t[110], grid_seconds=3600, lookback=48)
    b = a.copy()
    b[b > t[110]] += 1e8
    y = ArrivalSurvival.fit(t, b, cutoff=t[110], grid_seconds=3600, lookback=48)
    np.testing.assert_array_equal(x.delays, y.delays)
    np.testing.assert_allclose(x.probability(np.array([0.0]), 1800), [[1.0, 0.0]])


def test_arrival_survival_keeps_dropout_mass():
    a = ArrivalSurvival(np.array([[0.0, 100.0], [10.0, np.inf], [20.0, 200.0], [np.inf, 300.0]]))
    np.testing.assert_allclose(a.probability(np.array([10.0]), 10), [[0.5, 0.0]])


def test_policy_legality_and_batch_invariance():
    m = model()
    a = ArrivalSurvival(np.array([[0.5], [1.0], [np.inf]]))
    p = PosteriorPolicy(m, a, [0.0, 0.5, 1.0], grid_seconds=1, samples=8)
    x = np.zeros((4, 4, 1, 4))
    steps = np.array([0, 1, 2, 2])
    acq = np.zeros((4, 1), bool)
    x[-1, -1, 0, 1] = 1
    acq[-1, 0] = True
    action = p.choose(x, np.zeros(4), steps, acq, np.array([0.2]))
    assert not (action[steps == 2] == -2).any()
    assert action[-1] == -1
    for i in range(4):
        np.testing.assert_array_equal(
            action[i : i + 1],
            p.choose(x[i : i + 1], np.zeros(1), steps[i : i + 1], acq[i : i + 1], np.array([0.2])),
        )
    np.testing.assert_array_equal(action, p.choose(x, np.zeros(4), steps, acq, np.array([0.2])))


def test_bad_subset_cost_rejected():
    with pytest.raises(ValueError):
        best_sensor_subset(
            np.array([1.0]),
            np.ones((1, 1, 1)),
            np.ones((1, 1)),
            np.ones((1, 1), bool),
            np.array([-1.0]),
            np.zeros(1),
        )


def test_unavailable_training_values_do_not_change_fitted_parameters():
    v = training_values()
    known = np.ones(v.shape, bool)
    known[::7, 0] = False
    changed = v.copy()
    changed[~known] = 1e10
    a = GaussianForecast.fit(v, known=known, seed=42)
    b = GaussianForecast.fit(changed, known=known, seed=42)
    for name in ("transition", "center", "noise", "prior"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def test_filter_against_independent_joint_gaussian_conditioning():
    m = model()
    n = 6
    cov = 0.8 ** np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    obs = np.array([0, 2])
    value = np.array([3.0, -1.0])
    v = np.zeros((1, 4, 1))
    o = np.zeros_like(v, bool)
    v[0, obs, 0] = value
    o[0, obs, 0] = True
    mean = 2 + cov[5, obs] @ np.linalg.solve(cov[np.ix_(obs, obs)], value - 2)
    variance = cov[5, 5] - cov[5, obs] @ np.linalg.solve(cov[np.ix_(obs, obs)], cov[obs, 5])
    p = m.moments(snapshot(v, o))
    np.testing.assert_allclose(p.mean, [mean], atol=1e-12)
    np.testing.assert_allclose(p.variance, [variance], atol=1e-12)


def test_empty_transition_rejected():
    with pytest.raises(ValueError):
        GaussianForecast(
            np.zeros((0, 0)), np.zeros((0, 0)), np.array([0.0]), np.zeros((0, 0)), 2, 0, np.zeros(1)
        )
