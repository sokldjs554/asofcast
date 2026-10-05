"""Prepare predeclared UCI sources after method freeze, without model scoring."""

import argparse
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from asofcast.data import sha256_file, write_json
from scripts.run_honest_plan_confirmation import scientific_source_lock


def prepare(declaration, out):
    out.mkdir(parents=True, exist_ok=False)
    metadata = {}
    for name, spec in declaration["datasets"].items():
        archive = out / f"{name}-original.zip"
        with urllib.request.urlopen(spec["raw_url"], timeout=60) as response:
            archive.write_bytes(response.read())
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:
                raise ValueError("official archive CRC failed")
            matches = [p for p in z.namelist() if Path(p).name == spec["file"]]
            if len(matches) != 1:
                raise ValueError("unexpected official archive layout")
            native = z.read(matches[0])
        (out / spec["file"]).write_bytes(native)
        frame = pd.read_csv(
            io.BytesIO(native), encoding="latin1" if name == "SeoulBike" else "utf-8"
        )
        if name == "AppliancesEnergy":
            dates = pd.to_datetime(frame["date"], errors="raise")
            selected = frame[spec["columns"]].copy()
            expected = 19735
        else:
            # Unit suffixes are removed only for the predeclared numeric channels.
            headers = {
                column: [h for h in frame.columns if h.split("(")[0].strip() == column]
                for column in spec["columns"]
            }
            if any(len(v) != 1 for v in headers.values()):
                raise ValueError(f"raw numeric header mapping differs: {headers}")
            selected = frame[[headers[c][0] for c in spec["columns"]]].copy()
            selected.columns = spec["columns"]
            dates = pd.to_datetime(
                frame["Date"], format="%d/%m/%Y", errors="raise"
            ) + pd.to_timedelta(frame["Hour"], unit="h")
            expected = 8760
        values = selected.to_numpy(dtype=float)
        if len(frame) != expected or not np.isfinite(values).all():
            raise ValueError("unexpected length or missing selected measurements; do not trim")
        intervals = (
            np.diff(dates.to_numpy(dtype="datetime64[s]")).astype("timedelta64[s]").astype(int)
        )
        if not np.equal(intervals, spec["grid_seconds"]).all():
            raise ValueError("source clock is not a complete declared regular grid")
        selected.insert(0, "date", dates)
        destination = out / f"{name}.csv"
        selected.to_csv(destination, index=False, float_format="%.17g", lineterminator="\n")
        restored = pd.read_csv(destination, float_precision="round_trip")
        np.testing.assert_array_equal(restored[spec["columns"]].to_numpy(dtype=float), values)
        metadata[name] = {
            **spec,
            "rows": len(frame),
            "raw_rows": len(frame),
            "excluded_missing_rows": 0,
            "original_time_gaps": 0,
            "prepared_sha256": sha256_file(destination),
            "raw_source_sha256": sha256_file(archive),
            "native_file_sha256": sha256_file(out / spec["file"]),
            "original_time_range": [str(dates.iloc[0]), str(dates.iloc[-1])],
            "numeric_cells_round_trip_verified": True,
            "scored_during_preparation": False,
            "arrival_timestamps": "simulated, not original delivery telemetry",
            "license": "CC BY 4.0",
        }
        print(json.dumps({"source_prepared": name, "rows": len(frame)}), flush=True)
    write_json(out / "provenance.json", metadata)
    return metadata


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lock", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    lock = json.loads(a.lock.read_text())
    if not lock["method_frozen_before_new_source_read"]:
        raise ValueError("method must be frozen before reading sources")
    if scientific_source_lock() != lock["source_lock"]:
        raise ValueError("scientific source changed after method freeze")
    prepare(lock["new_source_declaration"], a.out)
