import importlib.util

import numpy as np
import pytest


def suite():
    assert importlib.util.find_spec("scripts.posterior_candidate_suite") is not None, (
        "candidate suite is missing"
    )
    from scripts import posterior_candidate_suite

    return posterior_candidate_suite


def test_state_aware_simple_does_not_wait_for_an_already_observed_target():
    policy = suite().TargetAvailabilityPolicy(2, 0)
    x = np.zeros((4, 48, 1, 4))
    x[0, -1, 0, 1] = 1
    x[3, -1, 0, 1] = 1
    acquired = np.zeros((4, 1), bool)
    acquired[3, 0] = True
    a = policy.choose(x, np.zeros(4), np.array([0, 0, 2, 2]), acquired, np.array([1.0]))
    np.testing.assert_array_equal(a, [-1, -2, 0, -1])


def test_selection_does_not_accept_mae_regression_for_lower_objective():
    records = {
        "simple": {"objective": 1.0, "native_error": 1.0},
        "overfit": {"objective": 0.8, "native_error": 1.02},
        "safe": {"objective": 0.99, "native_error": 1.005},
    }
    assert suite().select_family(records) == "safe"


def test_exact_tie_retains_strong_simple():
    records = {
        "simple": {"objective": 1.0, "native_error": 1.0},
        "candidate": {"objective": 1.0, "native_error": 1.0},
    }
    assert suite().select_family(records) == "simple"


def test_no_unverified_arrays_are_accepted_for_gate():
    with pytest.raises(ValueError):
        suite().check_result_alignment({"a": np.arange(4), "b": np.arange(5)})


def test_existing_aggregate_is_never_overwritten(tmp_path):
    path = tmp_path / "aggregate.json"
    path.write_text("original evidence")
    with pytest.raises(FileExistsError):
        suite().aggregate_suite(tmp_path, {})
    assert path.read_text() == "original evidence"
