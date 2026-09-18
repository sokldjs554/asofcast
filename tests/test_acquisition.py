import numpy as np
import torch


def test_acquisition_gain_targets_reward_error_reduction():
    from asofcast.acquisition import acquisition_gain_targets
    current = np.array([3.0, 0.0], dtype=np.float32)
    counterfactual = np.array([[2.0, 5.0], [1.0, -2.0]], dtype=np.float32)
    targets = np.array([1.0, 1.0], dtype=np.float32)
    gain = acquisition_gain_targets(current, counterfactual, targets)
    np.testing.assert_allclose(gain, [[1.0, -2.0], [1.0, -2.0]])


def test_acquisition_features_encode_current_state_candidate_and_cost():
    from asofcast.acquisition import acquisition_features
    x = np.zeros((2, 6, 3, 4), dtype=np.float32)
    x[:, -1, :, 0] = [[1, 2, 3], [4, 5, 6]]
    x[:, -1, :, 1] = [[1, 0, 1], [0, 1, 0]]
    x[:, -2:, :, 1] = 1
    prediction = np.array([.5, -.25], dtype=np.float32)
    channels = np.array([0, 2], dtype=np.int64)
    costs = np.array([.2, .5, .9], dtype=np.float32)
    features = acquisition_features(x, prediction, channels, costs, .5)
    assert features.shape == (2, 28)
    base = 3 * 6 + 2
    np.testing.assert_allclose(features[0, base:base+4], x[0, -1, 0])
    np.testing.assert_allclose(features[1, base:base+4], x[1, -1, 2])
    np.testing.assert_allclose(features[:, base+4:base+7], [[1,0,0],[0,0,1]])
    np.testing.assert_allclose(features[:, -1], [.2, .9])
    assert np.isfinite(features).all()


def test_acquisition_features_reject_invalid_candidate_indices():
    from asofcast.acquisition import acquisition_features
    x = np.ones((1, 4, 2, 4), dtype=np.float32)
    with np.testing.assert_raises(ValueError):
        acquisition_features(x, np.array([0.]), np.array([2]), np.ones(2), .5)


def test_joint_action_prefers_best_positive_acquisition():
    from asofcast.acquisition import select_joint_action
    result = select_joint_action(
        wait_gain=.08, wait_cost=.02,
        acquisition_gains=np.array([.30, .10]),
        acquisition_costs=np.array([.20, .20]),
        eligible=np.array([True, True]),
        acquisition_cost_weight=.10,
    )
    assert result['action'] == 'ACQUIRE'
    assert result['channel'] == 0
    assert result['net_utility'] > result['wait_net_utility']


def test_joint_action_can_choose_wait_or_commit_and_excludes_ineligible():
    from asofcast.acquisition import select_joint_action
    waiting = select_joint_action(
        wait_gain=.20, wait_cost=.02,
        acquisition_gains=np.array([.25, .05]),
        acquisition_costs=np.array([1.0, .1]),
        eligible=np.array([False, True]),
        acquisition_cost_weight=.20,
    )
    assert waiting['action'] == 'WAIT'
    assert waiting['channel'] is None

    commit = select_joint_action(
        wait_gain=.01, wait_cost=.02,
        acquisition_gains=np.array([.01, -.1]),
        acquisition_costs=np.array([.5, .5]),
        eligible=np.array([True, True]),
        acquisition_cost_weight=.10,
    )
    assert commit['action'] == 'COMMIT'
    assert commit['channel'] is None


def test_acquisition_value_model_learns_nonconstant_gain():
    from asofcast.acquisition import AcquisitionValueModel
    from asofcast.training import predict, train_regressor
    x = np.linspace(-1, 1, 160, dtype=np.float32)[:, None]
    x = np.concatenate([x, np.zeros_like(x)], axis=1)
    y = .3 * x[:, 0]
    model = AcquisitionValueModel(2)
    model.fit_scaling(x, y)
    train_regressor(model, x, y, epochs=80, learning_rate=.01, batch_size=80, seed=7)
    prediction = predict(model, x)
    assert np.mean((prediction - y) ** 2) < .002
    assert prediction[0] < 0 < prediction[-1]
