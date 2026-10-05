"""Refit the already declared families on scenario-mature exposed banks."""

import argparse
import ast
import json
from pathlib import Path

import numpy as np

from asofcast.data import sha256_file, write_json
from asofcast.research_safe_plan import HonestGainRouter, HonestPlanRouter
from scripts.run_terminal_plan_confirmation import save_plan_policy

ROOT = Path(__file__).resolve().parents[1]


def candidate_settings(previous):
    names = [
        "development-v2",
        "development-value-tree",
        "development-mae-priority-v2",
        "development-ensemble",
    ]
    candidates = []
    for name in names:
        design = json.loads((previous / name / "design.json").read_text())
        candidates += design.get("settings", design.get("candidates", []))
    assert len(candidates) == 26
    assert len({s["name"] for s in candidates}) == 26
    return candidates


def evaluate_router(router, validation, strength, margin):
    cells = {}
    for cn, vf, vl, ve in validation:
        choices = router.select(vf, strength=strength, margin=margin)
        ix = np.arange(len(vf))
        objective, mae = (vl[ix, choices].mean(), ve[ix, choices].mean())
        cells[cn] = {
            "relative_improvement": float(1 - objective / vl[:, 0].mean()),
            "mae_regression": float(mae / ve[:, 0].mean() - 1),
            "objective": float(objective),
            "native_error": float(mae),
            "nondefault_fraction": float((choices != 0).mean()),
        }
    return {
        "cells": cells,
        "mean_improvement": float(np.mean([c["relative_improvement"] for c in cells.values()])),
        "minimum_improvement": min(c["relative_improvement"] for c in cells.values()),
        "max_mae_regression": max(c["mae_regression"] for c in cells.values()),
    }


def main():
    p = argparse.ArgumentParser()
    for name in ["banks", "previous", "out", "frozen_module"]:
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    p.add_argument("--dataset", required=True)
    a = p.parse_args()

    def definition(path, name):
        node = next(
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == name
        )
        return ast.dump(node, include_attributes=False)

    for name in ["initial_features", "execute_plans", "strategy_bank", "execute_strategies"]:
        assert definition(a.frozen_module, name) == definition(
            ROOT / "src/asofcast/research_safe_plan.py", name
        )
    a.out.mkdir(parents=True, exist_ok=False)
    case = a.banks / a.dataset
    with np.load(case / "learning-arrays.npz", allow_pickle=False) as raw:
        f, g, ids = [raw[k].copy() for k in ["features", "gains", "origins"]]
    with np.load(case / "mae-advantage.npz", allow_pickle=False) as raw:
        mae = raw["mae_gain"].copy()
        np.testing.assert_array_equal(raw["origins"], ids)
    validation = []
    for v in sorted(case.glob("validation-*.npz")):
        with np.load(v, allow_pickle=False) as d:
            validation.append(
                (
                    v.stem.removeprefix("validation-"),
                    *[d[k].copy() for k in ["features", "objective", "native_error"]],
                )
            )
    settings = candidate_settings(a.previous)
    write_json(
        a.out / "design.json",
        {
            "scope": "exposed development only",
            "candidates": settings,
            "input_hashes": {
                p.name: sha256_file(p)
                for p in [
                    case / "learning-arrays.npz",
                    case / "mae-advantage.npz",
                    *case.glob("validation-*.npz"),
                ]
            },
            "runner_sha256": sha256_file(Path(__file__)),
            "module_sha256": sha256_file(ROOT / "src/asofcast/research_safe_plan.py"),
        },
    )
    records, diagnostics = ({}, {})
    for setting in settings:
        cls = (
            HonestGainRouter if setting.get("algorithm") in ["extra", "hist"] else HonestPlanRouter
        )
        router = cls.fit(f, g, ids, setting, seed=42, mae_gains=mae)
        save_plan_policy(router, a.out / f"{setting['name']}.joblib")
        diagnostics[setting["name"]] = router.diagnostics
        for strength in [0.0, 0.5, 1.0]:
            for margin in [0.0, 0.005, 0.01]:
                key = f"{setting['name']}_s{strength:g}_m{margin:g}"
                records[key] = {
                    "candidate": setting["name"],
                    "strength": strength,
                    "margin": margin,
                    **evaluate_router(router, validation, strength, margin),
                }
        write_json(a.out / "progress.json", {"records": records, "diagnostics": diagnostics})
        print(json.dumps({"dataset": a.dataset, "fitted": setting["name"]}), flush=True)
    admissible = [k for k, r in records.items() if r["max_mae_regression"] <= 0.01]
    best = (
        min(admissible, key=lambda k: (-records[k]["mean_improvement"], k)) if admissible else None
    )
    selected = best if best and records[best]["mean_improvement"] > 0 else "default"
    report = {
        "dataset": a.dataset,
        "best": best,
        "selected": selected,
        "records": records,
        "diagnostics": diagnostics,
        "default": {"relative_improvement": 0.0, "mae_regression": 0.0},
    }
    write_json(a.out / "selection.json", report)
    print(json.dumps({"dataset": a.dataset, "best": best, "record": records.get(best)}), flush=True)


if __name__ == "__main__":
    main()
