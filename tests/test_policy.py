import numpy as np
import pytest
import torch


def test_gain_targets_have_correct_sign_and_no_final_action():
    from asofcast.policy import gain_targets
    pred = np.array([[0., 1., 2.], [2., 0., 1.]])
    target = np.array([2., 0.])
    np.testing.assert_allclose(gain_targets(pred, target), [[1., 1.], [2., -1.]])


def test_features_only_use_current_snapshot_and_prediction():
    from asofcast.policy import policy_features
    x = np.ones((2, 12, 3, 4), dtype=np.float32)
    p = np.array([.5, -.2], dtype=np.float32)
    s = policy_features(x, p, .5)
    assert s.ndim == 2 and s.shape[0] == 2 and np.isfinite(s).all()
    np.testing.assert_allclose(s[:, -2], p)
    np.testing.assert_allclose(s[:, -1], .5)
    assert s.shape[1] == 3 * 6 + 2


class ConstantGain(torch.nn.Module):
    def __init__(self, gain):
        super().__init__()
        self.gain = gain
    def forward(self, x):
        return torch.full((len(x),), self.gain, dtype=x.dtype)


def test_always_positive_gain_stops_at_deadline():
    from asofcast.policy import choose_steps
    s = np.ones((4, 3, 8), dtype=np.float32)
    steps = choose_steps(s, ConstantGain(10.), .01, [0, 30, 60])
    np.testing.assert_array_equal(steps, [2, 2, 2, 2])


def test_no_expected_gain_stops_immediately_and_ignores_future():
    from asofcast.policy import choose_steps
    s = np.ones((4, 3, 8), dtype=np.float32)
    first = choose_steps(s, ConstantGain(0.), 0., [0, 30, 60])
    s[:, 1:] *= -999999
    other = choose_steps(s, ConstantGain(0.), 0., [0, 30, 60])
    np.testing.assert_array_equal(first, other)
    np.testing.assert_array_equal(first, np.zeros(4, dtype=int))


@pytest.mark.parametrize('waits', [[0, 0, 60], [0, -30, 60], [30, 60, 90]])
def test_bad_wait_grid_rejected(waits):
    from asofcast.policy import choose_steps
    with pytest.raises(ValueError):
        choose_steps(np.ones((2, 3, 8), dtype=np.float32), ConstantGain(1.), .01, waits)


def test_negative_threshold_rejected():
    from asofcast.policy import choose_steps
    with pytest.raises(ValueError):
        choose_steps(np.ones((2, 3, 8), dtype=np.float32), ConstantGain(1.), -1., [0, 30, 60])


def test_metrics_keep_wait_and_inference_separate():
    from asofcast.metrics import score
    y = np.array([1., 3.])
    p = np.array([[0., 1., 2.], [0., 2., 3.]])
    result = score(y, p, np.array([1, 2]), [0, 30, 60])
    assert result['mae'] == 0.0
    assert result['mean_wait_seconds'] == 45.
    assert result['deadline_violation_rate'] == 0.0
    assert 'inference_ms' not in result


def test_matched_random_baseline_is_exact_expectation():
    from asofcast.metrics import random_matched_score
    result = random_matched_score(np.array([0., 2.]), np.array([[0., 1.], [0., 2.]]), np.array([0, 1]), [0, 60])
    assert result['mae'] == .75
    assert result['mean_wait_seconds'] == 30.


def test_gain_policy_learns_not_a_hardcoded_rule():
    from asofcast.policy import GainPolicy
    from asofcast.training import predict, train_regressor
    torch.manual_seed(2)
    x = np.linspace(-1, 1, 160).astype(np.float32)[:, None]
    x = np.concatenate([x, x * 0], axis=1)
    y = x[:, 0] * .2
    model = GainPolicy(2)
    model.fit_scaling(x, y)
    train_regressor(model, x, y, epochs=80, learning_rate=.01, batch_size=80, seed=2)
    p = predict(model, x)
    assert np.mean((p - y) ** 2) < .001
    assert p[0] < 0 < p[-1]
