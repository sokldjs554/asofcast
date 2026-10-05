"""Filter exposed banks without changing any labels; preserve condition identities."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from asofcast.bundle import load_bundle
from asofcast.data import sha256_file, write_json
from asofcast.preprocessing import partition_bounds
from asofcast.research_safe_plan import mature_policy_origins
from scripts.evaluate_research_diagnosis import normalized_timeline
from scripts.run_terminal_plan_confirmation import save_numeric_arrays


def main():
    p = argparse.ArgumentParser()
    for name in ["banks", "mae_banks", "prior", "headroom", "out"]:
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    design = json.loads((a.banks / "design.json").read_text())
    conditions = design["conditions"]
    records = {}
    for ds in ["Taylor", "MSFT", "BikeSharing", "BeijingPM25"]:
        case, out = (a.banks / ds, a.out / ds)
        out.mkdir()
        bundle_path = (
            a.headroom / "pinned-run2/models" / f"{ds}-42"
            if ds in ["Taylor", "MSFT"]
            else a.prior / "confirmation-1/models" / f"{ds}-42"
        )
        bundle = load_bundle(bundle_path)
        cutoff = bundle.timeline.times[
            partition_bounds(len(bundle.timeline.times))["policy"][1] - 1
        ]
        with np.load(case / "learning-arrays.npz", allow_pickle=False) as x:
            f, g, ids = [x[k].copy() for k in ["features", "gains", "origins"]]
        with np.load(a.mae_banks / ds / "mae-advantage.npz", allow_pickle=False) as m:
            mae = m["mae_gain"].copy()
            np.testing.assert_array_equal(ids, m["origins"])
        assert len(ids) % len(conditions) == 0
        view_ids = np.repeat(np.arange(len(conditions)), len(ids) // len(conditions))
        keep = np.zeros(len(ids), dtype=bool)
        records[ds] = {"source_hashes": {}, "conditions": {}}
        for i, cond in enumerate(conditions):
            rows = np.flatnonzero(view_ids == i)
            tl = normalized_timeline(bundle, cond["profile"], cond["arrival_seed"])
            mature = mature_policy_origins(
                tl, ids[rows], bundle.config, bundle.manifest["target_channel"], cutoff
            )
            keep[rows] = np.isin(ids[rows], mature)
            records[ds]["conditions"][cond["name"]] = {
                "before": len(rows),
                "after": len(mature),
                "removed_origins": ids[rows][~keep[rows]].tolist(),
                "cutoff": float(cutoff),
            }
        save_numeric_arrays(
            out / "learning-arrays.npz",
            {
                "features": f[keep],
                "gains": g[keep],
                "origins": ids[keep],
                "condition_ids": view_ids[keep],
            },
        )
        save_numeric_arrays(
            out / "mae-advantage.npz",
            {"mae_gain": mae[keep], "origins": ids[keep], "condition_ids": view_ids[keep]},
        )
        for src in [
            case / "learning-arrays.npz",
            a.mae_banks / ds / "mae-advantage.npz",
            *case.glob("validation-*.npz"),
            case / "forecaster.joblib",
            case / "default-policy.joblib",
            case / "selection.json",
        ]:
            records[ds]["source_hashes"][str(src)] = sha256_file(src)
            if src.name not in ["learning-arrays.npz", "mae-advantage.npz"]:
                shutil.copyfile(src, out / src.name)
        records[ds]["filtered_hashes"] = {
            k: sha256_file(out / k) for k in ["learning-arrays.npz", "mae-advantage.npz"]
        }
        with np.load(out / "learning-arrays.npz", allow_pickle=False) as x:
            for k, original in [("features", f), ("gains", g), ("origins", ids)]:
                np.testing.assert_array_equal(x[k], original[keep])
        print(json.dumps({"dataset": ds, "conditions": records[ds]["conditions"]}), flush=True)
    design["scope"] = (
        "exposed development; prior immature rows removed per simulated target schedule"
    )
    design["source_design_sha256"] = sha256_file(a.banks / "design.json")
    write_json(a.out / "design.json", design)
    write_json(a.out / "maturity-audit.json", records)


if __name__ == "__main__":
    main()
