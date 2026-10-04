import importlib.util

import pytest


def test_portable_runner_rejects_an_existing_output_before_training(tmp_path):
    assert importlib.util.find_spec("scripts.run_posterior_confirmation") is not None, (
        "portable runner missing"
    )
    from scripts.run_posterior_confirmation import main

    with pytest.raises(FileExistsError):
        main(["--out", str(tmp_path)])
