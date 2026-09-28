import numpy as np
import pytest

from asofcast import decision_evidence as evidence


def test_gain_selection_rejects_unavailable_sensor_and_nonpositive_net_gain():
    gains = np.array([[99., .4], [.1, 99.], [99., 99.]])
    eligible = np.array([[False, True], [True, False], [False, False]])
    chosen = evidence.choose_gain_actions(gains, eligible, np.array([1., 2.]), .1)
    assert chosen.tolist() == [1, -1, -1]


def test_action_loss_charges_cost_in_standardized_units_and_keeps_commit():
    loss = evidence.action_losses(
        np.array([1., 1.]), np.array([3., 3.]),
        np.array([[1., np.nan], [np.nan, np.nan]]), np.array([0, -1]),
        np.array([[True, False], [False, False]]), np.array([2., 1.]), .25, 10.)
    np.testing.assert_allclose(loss['objective'], [.5, 2.])
    np.testing.assert_allclose(loss['native_error'], [0., 20.])
    np.testing.assert_allclose(loss['cost'], [2., 0.])


def test_action_loss_rejects_ineligible_or_nonfinite_selected_outcome():
    args = [np.array([1.]), np.array([3.]), np.array([[np.nan, 1.]]),
            np.array([0]), np.array([[False, True]]), np.array([1., 1.]), .03, 1.]
    with pytest.raises(ValueError):
        evidence.action_losses(*args)
    args[4] = np.array([[True, True]])
    with pytest.raises(ValueError):
        evidence.action_losses(*args)


def test_random_matched_uses_same_acquisition_rows_and_exact_expected_cost():
    loss = evidence.random_matched_losses(
        np.array([1., 1.]), np.array([3., 3.]), np.array([[1., 5.], [1., 1.]]),
        np.array([0, -1]), np.ones((2, 2), dtype=bool), np.array([1., 3.]), .25, 10.)
    np.testing.assert_allclose(loss['native_error'], [20., 20.])
    np.testing.assert_allclose(loss['cost'], [2., 0.])
    np.testing.assert_allclose(loss['objective'], [2.5, 2.])


def test_ridge_learns_distinct_sensor_response_from_eligible_training_rows():
    values = np.arange(-5., 6.)
    x = np.repeat(values[:, None, None], 2, axis=1)
    gains = np.column_stack([2 * values + 1, -3 * values + 2])
    eligible = np.ones(gains.shape, dtype=bool)
    eligible[0, 0] = False
    gains[0, 0] = np.nan
    model = evidence.RidgeGainModel.fit(x, gains, eligible, ridge=1e-6)
    predicted = model.predict(np.array([[[2.], [2.]], [[-2.], [-2.]]]))
    np.testing.assert_allclose(predicted, [[5., -4.], [-3., 8.]], atol=1e-5)
    x[:] = 1e9
    np.testing.assert_allclose(model.predict(np.array([[[2.], [2.]]])), [[5., -4.]], atol=1e-5)


def test_unseen_sensor_ridge_uses_zero_gain_and_does_not_invent_training_signal():
    model = evidence.RidgeGainModel.fit(
        np.ones((3, 2, 1)), np.array([[2., np.nan]] * 3),
        np.array([[True, False]] * 3), ridge=1.)
    np.testing.assert_allclose(model.predict(np.ones((1, 2, 1))), [[2., 0.]])


def test_validation_selection_uses_all_rows_and_rejects_invalid_scores():
    assert evidence.select_validation({'a': np.array([0., 3.]), 'b': np.array([1., 1.])}) == 'b'
    with pytest.raises(ValueError):
        evidence.select_validation({'a': np.array([0., np.nan])})


def test_paired_interval_preserves_sign_and_does_not_treat_seeds_as_new_cases():
    delta = np.full((3, 64), .25)
    result = evidence.paired_block_interval(delta, block=8, repeats=200, seed=4)
    assert result['mean'] == pytest.approx(.25)
    assert result['lower95'] == pytest.approx(.25)
    assert result['upper95'] == pytest.approx(.25)
    assert result['origins'] == 64
    assert result['training_seeds'] == 3
    reversed_result = evidence.paired_block_interval(-delta, block=8, repeats=200, seed=4)
    assert reversed_result['upper95'] == pytest.approx(-.25)


def test_block_interval_accounts_for_serial_runs_and_is_reproducible():
    delta = np.array([[-1.] * 64 + [1.] * 64])
    iid = evidence.paired_block_interval(delta, block=1, repeats=500, seed=8)
    blocked = evidence.paired_block_interval(delta, block=32, repeats=500, seed=8)
    assert blocked['upper95'] - blocked['lower95'] > 2 * (iid['upper95'] - iid['lower95'])
    assert blocked == evidence.paired_block_interval(delta, block=32, repeats=500, seed=8)


def test_positive_mean_with_uncertain_interval_does_not_pass_gate():
    comparison = {'mean': .1, 'lower95': -.01, 'seed_improvements': [.1, .2, .1]}
    cell = {'comparisons': {'mlp': comparison, 'simple': comparison},
            'mae_relative_regressions': [0., 0., 0.]}
    assert not evidence.promotion_gate({'a': cell}, ['a'], .01)['passed']


def test_gate_requires_all_cells_all_seeds_and_mae_safety():
    comparison = {'mean': .1, 'lower95': .01, 'seed_improvements': [.1, .2, .1]}
    cell = {'comparisons': {'mlp': comparison, 'simple': comparison},
            'mae_relative_regressions': [0., 0., 0.]}
    assert evidence.promotion_gate({'a': cell}, ['a'], .01)['passed']
    assert not evidence.promotion_gate({'a': cell}, ['a', 'b'], .01)['passed']
    cell['mae_relative_regressions'][1] = .011
    assert not evidence.promotion_gate({'a': cell}, ['a'], .01)['passed']


def test_nonfinite_evidence_never_passes_as_a_success():
    with pytest.raises(ValueError):
        evidence.paired_block_interval(np.array([[np.nan] * 10]), block=2, repeats=20, seed=1)


def test_incomplete_seed_evidence_cannot_pass_the_promotion_gate():
    comparison = {'mean': .1, 'lower95': .01, 'seed_improvements': [.1, .2]}
    cell = {'comparisons': {'mlp': comparison, 'simple': comparison},
            'mae_relative_regressions': [0., 0., 0.]}
    assert not evidence.promotion_gate({'a': cell}, ['a'], .01)['passed']


def test_opposing_seeds_are_averaged_at_shared_origins_before_bootstrap():
    one = np.array([-1.] * 64 + [1.] * 64)
    result = evidence.paired_block_interval(np.stack([one, -one]), block=32, repeats=500, seed=8)
    assert result['lower95'] == 0.
    assert result['upper95'] == 0.
