import importlib.util

import numpy as np
import pandas as pd
import pytest

from asofcast.data import sha256_file


def source():
    assert importlib.util.find_spec("scripts.prepare_posterior_sources") is not None, (
        "source preparation is missing"
    )
    from scripts import prepare_posterior_sources

    return prepare_posterior_sources


def test_bundled_sources_keep_values_and_retain_true_date_mapping(tmp_path):
    from pmdarima.datasets import load_msft, load_taylor

    meta = source().prepare_sources(tmp_path / "prepared")
    t = pd.read_csv(tmp_path / "prepared/Taylor.csv")
    np.testing.assert_allclose(t["demand"].to_numpy(), load_taylor(), rtol=0, atol=0)
    assert meta["Taylor"]["rows"] == 4032 and meta["Taylor"]["grid_seconds"] == 1800
    m = pd.read_csv(tmp_path / "prepared/MSFT.csv")
    original = load_msft()
    assert len(m) == 7983 and list(m.columns) == ["date", "Open", "High", "Low", "Close", "Volume"]
    np.testing.assert_allclose(
        m.iloc[:, 1:], original[["Open", "High", "Low", "Close", "Volume"]], rtol=0, atol=0
    )
    mapped = pd.read_csv(tmp_path / "prepared/MSFT-date-map.csv")
    assert mapped["original_date"].tolist() == original["Date"].tolist()
    assert np.all(np.diff(pd.to_datetime(m["date"]).astype("int64")) == 86400 * 10**9)
    assert "trading" in meta["MSFT"]["time_axis"]
    for name in meta:
        assert meta[name]["prepared_sha256"] == sha256_file(tmp_path / "prepared" / f"{name}.csv")
    with pytest.raises(FileExistsError):
        source().prepare_sources(tmp_path / "prepared")


def test_csv_serialization_declares_platform_independent_newlines(tmp_path, monkeypatch):
    original = pd.DataFrame.to_csv
    seen = []

    def recorded(self, *args, **kwargs):
        seen.append(kwargs.get("lineterminator"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", recorded)
    source().prepare_sources(tmp_path / "prepared")
    assert seen and all(value == "\n" for value in seen)
