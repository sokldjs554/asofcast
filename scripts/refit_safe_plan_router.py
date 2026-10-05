"""Reuse immutable exposed-development counterfactual banks for reward-tree fitting."""

import argparse
import ast
import json
from pathlib import Path

import numpy as np

from asofcast.data import sha256_file, write_json
from asofcast.research_safe_plan import HonestPlanRouter
from scripts.run_terminal_plan_confirmation import save_plan_policy

ROOT = Path(__file__).resolve().parents[1]


def definition(path, name):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.dump(node, include_attributes=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--banks", type=Path, required=True)
    p.add_argument("--frozen-module", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    for name in ["initial_features", "execute_plans", "strategy_bank", "execute_strategies"]:
        assert definition(a.frozen_module, name) == definition(
            ROOT / "src/asofcast/research_safe_plan.py", name
        )
    a.out.mkdir(parents=True, exist_ok=False)
    candidates = [
        {
            "name": f"value_d{d}_n{n}",
            "algorithm": "value_tree",
            "max_depth": d,
            "min_samples_leaf": n,
            "fit_fraction": 0.6,
            "purge_observations": 54,
            "minimum_calibration_origins": 12,
        }
        for d in [2, 3, 4]
        for n in [32, 64]
    ]
    reports = {}
    inputs = {}
    for ds in ["Taylor", "MSFT", "BikeSharing", "BeijingPM25"]:
        case = a.banks / ds
        out = a.out / ds
        out.mkdir()
        with np.load(case / "learning-arrays.npz", allow_pickle=False) as raw:
            f, g, ids = [raw[k].copy() for k in ["features", "gains", "origins"]]
        inputs[f"{ds}/learning-arrays.npz"] = sha256_file(case / "learning-arrays.npz")
        val = []
        for v in sorted(case.glob("validation-*.npz")):
            with np.load(v, allow_pickle=False) as data:
                val.append(
                    (
                        v.stem.removeprefix("validation-"),
                        *[data[k].copy() for k in ["features", "objective", "native_error"]],
                    )
                )
            inputs[f"{ds}/{v.name}"] = sha256_file(v)
        records = {}
        diagnostics = {}
        for settings in candidates:
            router = HonestPlanRouter.fit(f, g, ids, settings, seed=42)
            save_plan_policy(router, out / f"{settings['name']}.joblib")
            diagnostics[settings["name"]] = router.diagnostics
            for strength in [0.0, 0.5, 1.0]:
                for margin in [0.0, 0.005, 0.01]:
                    key = f"{settings['name']}_s{strength:g}_m{margin:g}"
                    cells = {}
                    for cn, vf, vl, ve in val:
                        choices = router.select(vf, strength=strength, margin=margin)
                        ix = np.arange(len(vf))
                        objective = vl[ix, choices].mean()
                        mae = ve[ix, choices].mean()
                        cells[cn] = {
                            "relative_improvement": float(1 - objective / vl[:, 0].mean()),
                            "mae_regression": float(mae / ve[:, 0].mean() - 1),
                            "objective": float(objective),
                            "native_error": float(mae),
                            "nondefault_fraction": float((choices != 0).mean()),
                        }
                    records[key] = {
                        "candidate": settings["name"],
                        "strength": strength,
                        "margin": margin,
                        "cells": cells,
                        "mean_improvement": float(
                            np.mean([c["relative_improvement"] for c in cells.values()])
                        ),
                        "max_mae_regression": max(c["mae_regression"] for c in cells.values()),
                    }
        admissible = [k for k, r in records.items() if r["max_mae_regression"] <= 0.01]
        best = (
            min(admissible, key=lambda k: (-records[k]["mean_improvement"], k))
            if admissible
            else None
        )
        reports[ds] = {
            "best": best,
            "selected": best if best and records[best]["mean_improvement"] > 0 else "default",
            "records": records,
            "router_calibration": diagnostics,
        }
        write_json(out / "selection.json", reports[ds])
        write_json(a.out / "progress.json", reports)
        print(json.dumps({"dataset": ds, "best": best, "record": records.get(best)}), flush=True)
    write_json(a.out / "development-report.json", reports)
    write_json(
        a.out / "design.json",
        {
            "scope": "exposed-data development only",
            "counterfactual_bank_AST_unchanged": True,
            "input_array_sha256": inputs,
            "candidates": candidates,
            "source_sha256": {
                "module": sha256_file(ROOT / "src/asofcast/research_safe_plan.py"),
                "runner": sha256_file(Path(__file__)),
            },
        },
    )


if __name__ == "__main__":
    main()
