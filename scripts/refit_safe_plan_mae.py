"""Reuse immutable exposed-development counterfactual banks for reward-tree fitting."""

import argparse
import ast
import json
from pathlib import Path

import joblib
import numpy as np

from asofcast.bundle import load_bundle
from asofcast.data import sha256_file, write_json
from asofcast.research_safe_plan import HonestPlanRouter, execute_strategies
from scripts.evaluate_research_diagnosis import normalized_timeline
from scripts.run_terminal_plan_confirmation import save_numeric_arrays, save_plan_policy

ROOT = Path(__file__).resolve().parents[1]


def definition(path, name):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.dump(node, include_attributes=False)


class ZeroForecast:
    def predict(self, x):
        return np.zeros(len(x))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--banks", type=Path, required=True)
    p.add_argument("--frozen-module", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--headroom", type=Path, required=True)
    a = p.parse_args()
    for name in ["initial_features", "execute_plans", "strategy_bank", "execute_strategies"]:
        assert definition(a.frozen_module, name) == definition(
            ROOT / "src/asofcast/research_safe_plan.py", name
        )
    a.out.mkdir(parents=True, exist_ok=False)
    candidates = [
        {
            "name": f"{algorithm}_d3_n64_p{priority:g}",
            "algorithm": algorithm,
            "max_depth": 3,
            "min_samples_leaf": 64,
            "fit_fraction": 0.6,
            "purge_observations": 54,
            "minimum_calibration_origins": 12,
            "mae_priority": priority,
        }
        for algorithm in ["regression_tree", "value_tree"]
        for priority in [0.0, 1.0, 2.0]
    ]
    reports = {}
    inputs = {}
    for ds in ["Taylor", "MSFT", "BikeSharing", "BeijingPM25"]:
        case = a.banks / ds
        out = a.out / ds
        out.mkdir()
        with np.load(case / "learning-arrays.npz", allow_pickle=False) as raw:
            f, g, ids = [raw[k].copy() for k in ["features", "gains", "origins"]]
            condition_ids = (
                raw["condition_ids"].copy()
                if "condition_ids" in raw
                else np.repeat(np.arange(3), len(ids) // 3)
            )
            assert len(condition_ids) == len(ids)
        inputs[f"{ds}/learning-arrays.npz"] = sha256_file(case / "learning-arrays.npz")
        bank = json.loads((case / "selection.json").read_text())["bank"]
        bundle_path = (
            a.headroom / "pinned-run2/models" / f"{ds}-42"
            if ds in ["Taylor", "MSFT"]
            else a.prior / "confirmation-1/models" / f"{ds}-42"
        )
        bundle = load_bundle(bundle_path)
        default = joblib.load(case / "default-policy.joblib")
        conditions = json.loads((a.banks / "design.json").read_text())["conditions"]
        matrices = []
        for condition_index, cond in enumerate(conditions):
            origins = ids[condition_ids == condition_index]
            tl = normalized_timeline(bundle, cond["profile"], cond["arrival_seed"])
            penalty = []
            for j in range(len(bank)):
                result = execute_strategies(
                    tl,
                    origins,
                    bundle.config,
                    ZeroForecast(),
                    bundle.acquisition_cost_proxy,
                    bank,
                    np.full(len(origins), j),
                    default,
                )
                penalty.append(
                    0.03 * result["cost"]
                    + 0.02 * result["wait_seconds"] / bundle.timeline.grid_seconds
                )
            penalty = np.column_stack(penalty)
            matrices.append(penalty - penalty[:, [0]])
        training_mae_gain = g + np.concatenate(matrices)
        assert np.equal(training_mae_gain[:, 0], 0).all()
        save_numeric_arrays(
            out / "mae-advantage.npz", {"mae_gain": training_mae_gain, "origins": ids}
        )
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
            router = HonestPlanRouter.fit(f, g, ids, settings, seed=42, mae_gains=training_mae_gain)
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
