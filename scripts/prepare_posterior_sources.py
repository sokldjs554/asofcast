"""Prepare two locally bundled real series without fitting or scoring a model.

No generated measurement fallback is allowed. Dates on the computational grid
are explicitly distinguished from original observation dates/trading sessions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import io
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pmdarima
from pmdarima.datasets import load_msft, load_taylor

from asofcast.data import sha256_file, write_json


def prepare_sources(out):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    t = np.asarray(load_taylor(), dtype=np.float64)
    msft = load_msft()
    cols = ["Open", "High", "Low", "Close", "Volume"]
    dates = pd.to_datetime(msft["Date"], errors="raise")
    numeric = msft[cols].to_numpy(dtype=np.float64)
    if (
        t.shape != (4032,)
        or numeric.shape != (7983, 5)
        or not np.isfinite(t).all()
        or not np.isfinite(numeric).all()
        or not dates.is_monotonic_increasing
        or not dates.is_unique
    ):
        raise ValueError("bundled dataset schema changed; explicit review required")
    out.mkdir(parents=True)
    tframe = pd.DataFrame(
        {"date": pd.date_range("2000-01-01", periods=len(t), freq="30min"), "demand": t}
    )
    mframe = msft[cols].copy()
    mframe.insert(0, "date", pd.date_range("2000-01-01", periods=len(msft), freq="D"))
    mapping = pd.DataFrame({"computational_date": mframe["date"], "original_date": msft["Date"]})
    mapping.to_csv(out / "MSFT-date-map.csv", index=False, lineterminator="\n")
    tframe.to_csv(out / "Taylor.csv", index=False, float_format="%.12g", lineterminator="\n")
    # Retain original numeric decimal cells instead of reserializing binary floats.
    # A float->CSV->float round trip otherwise perturbs some observations by 1 ULP.
    archive = Path(inspect.getsourcefile(load_msft)).parent / "data/msft.tar.gz"
    with tarfile.open(archive, "r:gz") as tar:
        files = [m for m in tar.getmembers() if m.isfile()]
        if len(files) != 1:
            raise ValueError("unexpected MSFT archive layout")
        native = list(csv.reader(io.TextIOWrapper(tar.extractfile(files[0]), encoding="utf-8")))
    indices = [native[0].index(c) for c in cols]
    if len(native) - 1 != len(msft):
        raise ValueError("MSFT archive count mismatch")
    with (out / "MSFT.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date"] + cols)
        for day, row in zip(mframe["date"], native[1:], strict=True):
            writer.writerow([str(day)] + [row[i] for i in indices])
    if not np.array_equal(pd.read_csv(out / "MSFT.csv")[cols].to_numpy(dtype=float), numeric):
        raise ValueError("numeric cells changed during preparation")
    common = {
        "pmdarima_version": pmdarima.__version__,
        "arrival_timestamps": "simulated, not original delivery records",
        "selection": "availability, length and different domains selected before any model score was inspected",
        "measurement_kind": "real bundled observations, not synthetic",
        "complete_series_used": True,
        "scored_during_preparation": False,
    }
    metadata = {
        "Taylor": {
            **common,
            "rows": len(t),
            "grid_seconds": 1800,
            "target": "demand",
            "columns": ["demand"],
            "raw_numeric_sha256": hashlib.sha256(t.astype("<f8").tobytes()).hexdigest(),
            "unit": "MW",
            "source_period": "2000-06-05 through 2000-08-27 as documented by provider",
            "time_axis": "half-hour order; computational 2000-01-01 epoch is artificial, not a claimed original clock alignment",
            "forecast_horizon": "6 half-hour observations = 3 hours",
            "decision_deadline": "one half-hour",
            "source": "https://pkg.robjhyndman.com/forecast/reference/taylor.html",
            "loader_sha256": sha256_file(Path(inspect.getsourcefile(load_taylor))),
        },
        "MSFT": {
            **common,
            "rows": len(msft),
            "grid_seconds": 86400,
            "target": "Close",
            "columns": cols,
            "raw_numeric_sha256": hashlib.sha256(numeric.astype("<f8").tobytes()).hexdigest(),
            "unit": "historical close quote as supplied, not verified execution prices or investment returns",
            "source_period": [str(dates.iloc[0].date()), str(dates.iloc[-1].date())],
            "time_axis": "one computational day = one trading observation; weekends/holidays are not elapsed latency",
            "date_mapping_sha256": sha256_file(out / "MSFT-date-map.csv"),
            "forecast_horizon": "6 trading observations",
            "decision_deadline": "one trading-observation interval",
            "source": "https://alkaline-ml.com/pmdarima/usecases/stocks.html",
            "original_provider": "Kaggle price-volume-data-for-all-us-stocks-etfs, cited by pmdarima loader",
            "not_financial_advice": True,
            "native_archive_sha256": sha256_file(archive),
            "numeric_cells_preserved": True,
            "loader_sha256": sha256_file(Path(inspect.getsourcefile(load_msft))),
        },
    }
    for name in metadata:
        metadata[name]["prepared_sha256"] = sha256_file(out / f"{name}.csv")
    write_json(out / "provenance.json", metadata)
    return metadata


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    prepare_sources(a.out)
