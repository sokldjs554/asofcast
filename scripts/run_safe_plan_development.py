"""Exposed-data development only; never reports independent superiority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from asofcast.bundle import load_bundle
from asofcast.data import sha256_file, write_json
from asofcast.experiment import build_examples
from asofcast.posterior_risk import GaussianForecast
from asofcast.preprocessing import partition_bounds, split_origins
from asofcast.research_learning import SimplePolicy, TorchForecast, fit_residual_forecast
from asofcast.research_safe_plan import (
    HonestPlanRouter,
    execute_strategies,
    initial_features,
    mature_policy_origins,
    strategy_bank,
)
from scripts.evaluate_research_diagnosis import normalized_timeline, outcomes
from scripts.posterior_candidate_suite import TargetAvailabilityPolicy
from scripts.run_terminal_plan_confirmation import save_numeric_arrays, save_plan_policy

ROOT = Path(__file__).resolve().parents[1]


def reconstruct_reference(bundle, case, protocol):
    selection = json.loads((case / "plan-selection.json").read_text())["reference"]
    ref = selection["reference"]
    simple = selection["strong_simple"]
    cfg = bundle.config
    if simple == "reference_simple":
        name, wait, kind = ref["validation_simple"].split("__")
        policy = SimplePolicy(int(wait[4:]), kind, bundle.manifest["target_channel"])
    elif simple.startswith("old_target_available_"):
        name = ref["original_validation_simple"].split("__")[0]
        policy = TargetAvailabilityPolicy(
            int(simple.rsplit("_", 1)[1]), bundle.manifest["target_channel"]
        )
    elif simple.startswith("gaussian_target_available_"):
        name = ref["posterior_forecast"]
        policy = TargetAvailabilityPolicy(
            int(simple.rsplit("_", 1)[1]), bundle.manifest["target_channel"]
        )
    else:
        raise ValueError("unrecognized strong comparator")
    if name.startswith("var"):
        forecast = GaussianForecast.load(case / "reference-fit" / f"{name}.npz")
    else:
        forecast = TorchForecast(bundle.calibrated)
        if name != "base":
            tl = normalized_timeline(bundle)
            origins = split_origins(len(tl.times), cfg["lookback"], cfg["horizon"], cfg["stride"])
            cutoff = tl.times[partition_bounds(len(tl.times))["train"][1] - 1]
            po = origins["train"]
            po = po[tl.arrivals[po + cfg["horizon"], bundle.manifest["target_channel"]] <= cutoff]
            tx, ty = build_examples(tl, po, cfg, bundle.manifest["target_channel"])
            vx, vy = build_examples(
                tl, origins["validation"], cfg, bundle.manifest["target_channel"]
            )
            forecast, chosen = fit_residual_forecast(
                forecast,
                tx.reshape((-1,) + tx.shape[2:]),
                np.repeat(ty, 3),
                vx.reshape((-1,) + vx.shape[2:]),
                np.repeat(vy, 3),
                protocol["forecast"],
                seed=cfg["seed"],
                metadata=name != "value",
            )
            assert chosen == ref["legacy_forecast_selection"][name]
    return (forecast, policy, selection)


def collect_bank(bundle, timeline, origins, forecaster, default, protocol):
    cfg, costs = (bundle.config, bundle.acquisition_cost_proxy)
    bank = strategy_bank(len(costs), len(cfg["waits_seconds"]))
    features = initial_features(timeline, origins, cfg, forecaster)
    y = timeline.values[origins + cfg["horizon"], bundle.manifest["target_channel"]]
    objectives = []
    errors = []
    for index in range(len(bank)):
        result = execute_strategies(
            timeline, origins, cfg, forecaster, costs, bank, np.full(len(origins), index), default
        )
        scores = outcomes(bundle, result, y, protocol)
        objectives.append(scores["objective"])
        errors.append(scores["native_error"])
    return (features, np.column_stack(objectives), np.column_stack(errors), bank)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", type=Path, required=True)
    ap.add_argument("--headroom", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--datasets", nargs="+", default=["Taylor", "MSFT", "BikeSharing", "BeijingPM25"]
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    protocol = json.loads((args.prior / "protocol.json").read_text())
    conditions = [
        {"name": "mixed_1811", "profile": "mixed", "arrival_seed": 1811},
        {"name": "mixed_4811", "profile": "mixed", "arrival_seed": 4811},
        {"name": "outage_5811", "profile": "outage", "arrival_seed": 5811},
    ]
    candidates = [
        {
            "name": f"leaf_d{d}_n{n}",
            "max_depth": d,
            "min_samples_leaf": n,
            "fit_fraction": 0.6,
            "purge_observations": 54,
            "minimum_calibration_origins": 12,
        }
        for d in [2, 3, 4]
        for n in [32, 64]
    ]
    strengths = [0.0, 0.5, 1.0]
    margins = [0.0, 0.005, 0.01]
    write_json(
        args.out / "design.json",
        {
            "scope": "exposed-data development only; no test origins read",
            "changes": [
                "same strong-comparator forecaster",
                "complete executable plans",
                "chronological honest leaf calibration",
            ],
            "settings": candidates,
            "strengths": strengths,
            "margins": margins,
            "conditions": conditions,
            "cost_weight": 0.03,
            "delay_weight": 0.02,
            "gate": protocol["gate"],
            "source_hashes": {
                str(p.relative_to(ROOT)): sha256_file(p)
                for p in [ROOT / "src/asofcast/research_safe_plan.py", Path(__file__)]
            },
        },
    )
    reports = {}
    for dataset in args.datasets:
        if dataset in ["Taylor", "MSFT"]:
            bundle_path = args.headroom / "pinned-run2/models" / f"{dataset}-42"
            case = args.prior / "development-v2" / dataset
        else:
            bundle_path = args.prior / "confirmation-1/models" / f"{dataset}-42"
            case = args.prior / "confirmation-1" / f"{dataset}-42"
        out = args.out / dataset
        out.mkdir()
        bundle = load_bundle(bundle_path)
        cfg = bundle.config
        raw = bundle.timeline
        effective = {
            **protocol,
            "objective": {**protocol["objective"], "deadline_seconds": raw.grid_seconds},
        }
        forecast, default, reference = reconstruct_reference(bundle, case, protocol)
        save_plan_policy(forecast, out / "forecaster.joblib")
        save_plan_policy(default, out / "default-policy.joblib")
        origins = split_origins(len(raw.times), cfg["lookback"], cfg["horizon"], cfg["stride"])
        cutoff = raw.times[partition_bounds(len(raw.times))["policy"][1] - 1]
        po = origins["policy"]
        po = po[raw.arrivals[po + cfg["horizon"], bundle.manifest["target_channel"]] <= cutoff]
        data = []
        validation = []
        for cond in conditions:
            tl = normalized_timeline(bundle, cond["profile"], cond["arrival_seed"])
            mature = mature_policy_origins(tl, po, cfg, bundle.manifest["target_channel"], cutoff)
            f, loss, error, bank = collect_bank(bundle, tl, mature, forecast, default, effective)
            data.append((f, loss[:, [0]] - loss, mature))
            vf, vl, ve, _ = collect_bank(
                bundle, tl, origins["validation"], forecast, default, effective
            )
            validation.append((cond["name"], vf, vl, ve))
            save_numeric_arrays(
                out / f"validation-{cond['name']}.npz",
                {
                    "features": vf,
                    "objective": vl,
                    "native_error": ve,
                    "origins": origins["validation"],
                },
            )
            print(
                json.dumps(
                    {
                        "stage": "bank_complete",
                        "dataset": dataset,
                        "condition": cond["name"],
                        "policy_origins": len(mature),
                        "strategies": len(bank),
                    }
                ),
                flush=True,
            )
        f, g, ids = [np.concatenate([d[i] for d in data]) for i in range(3)]
        save_numeric_arrays(
            out / "learning-arrays.npz",
            {
                "features": f,
                "gains": g,
                "origins": ids,
                "condition_ids": np.concatenate(
                    [np.full(len(d[2]), i, dtype=int) for i, d in enumerate(data)]
                ),
            },
        )
        records = {}
        routers = {}
        for settings in candidates:
            router = HonestPlanRouter.fit(f, g, ids, settings, seed=42)
            routers[settings["name"]] = router
            save_plan_policy(router, out / f"{settings['name']}.joblib")
            for strength in strengths:
                for margin in margins:
                    key = f"{settings['name']}_s{strength:g}_m{margin:g}"
                    cells = {}
                    for cn, vf, vl, ve in validation:
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
        reports[dataset] = {
            "best": best,
            "selected": best if best and records[best]["mean_improvement"] > 0 else "default",
            "records": records,
            "reference": reference,
            "bank": bank,
            "router_calibration": {k: r.diagnostics for k, r in routers.items()},
        }
        write_json(out / "selection.json", reports[dataset])
        write_json(args.out / "progress.json", reports)
        print(
            json.dumps({"dataset": dataset, "best": best, "record": records.get(best)}), flush=True
        )
    write_json(args.out / "development-report.json", reports)
    print(json.dumps({"stage": "development_complete"}), flush=True)


if __name__ == "__main__":
    main()
