"""A frozen, independent check of calibrated complete-plan value learning."""

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
import torch

from asofcast.bundle import load_bundle
from asofcast.data import sha256_file, write_json
from asofcast.experiment import run_experiment
from asofcast.preprocessing import partition_bounds, split_origins
from asofcast.research_diagnosis import rollout
from asofcast.research_safe_plan import (
    HonestGainRouter,
    HonestPlanRouter,
    execute_strategies,
    initial_features,
    mature_policy_origins,
)
from scripts.evaluate_research_diagnosis import normalized_timeline, outcomes
from scripts.fit_strong_plan_reference import fit_strong_reference
from scripts.refit_mature_plan_candidates import evaluate_router
from scripts.run_safe_plan_development import collect_bank
from scripts.run_terminal_plan_confirmation import (
    save_numeric_arrays,
    save_plan_policy,
    source_lock,
    validate_bundle_contract,
)


def scientific_source_lock():
    root = Path(__file__).resolve().parents[1]
    return {
        **source_lock(),
        **{str(p.relative_to(root)): sha256_file(p) for p in sorted(root.glob("configs/*.json"))},
    }


def confirm_case(lock, source_dir, dataset, seed, out):
    if scientific_source_lock() != lock["source_lock"]:
        raise ValueError("scientific code changed after method freeze")
    out.mkdir(parents=True, exist_ok=False)
    meta = json.loads((source_dir / "provenance.json").read_text())[dataset]
    source = source_dir / f"{dataset}.csv"
    if sha256_file(source) != meta["prepared_sha256"]:
        raise ValueError("prepared source provenance differs")
    protocol, cfg = (lock["protocol"], lock["base_config"])
    effective = {
        **protocol,
        "objective": {**protocol["objective"], "deadline_seconds": meta["grid_seconds"]},
    }
    write_json(
        out / "environment.json",
        {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
            "lock_sha256": lock["lock_sha256"],
            "source": meta,
            "seed": seed,
        },
    )
    print(json.dumps({"stage": "base_training", "dataset": dataset, "seed": seed}), flush=True)
    bundle_path = out / "bundle"
    run_experiment(
        source,
        bundle_path,
        {
            **cfg,
            "seed": seed,
            "target": meta["target"],
            "waits_seconds": [0, meta["grid_seconds"] / 2, meta["grid_seconds"]],
        },
        source_kind="user-provided-csv",
    )
    bundle = load_bundle(bundle_path)
    validate_bundle_contract(bundle, cfg, meta, seed)
    methods, reference = fit_strong_reference(bundle, effective, out / "reference-fit")
    forecast, default = methods["validation_simple"]
    cfg = bundle.config
    raw, target = (bundle.timeline, bundle.manifest["target_channel"])
    origins = split_origins(len(raw.times), cfg["lookback"], cfg["horizon"], cfg["stride"])
    cutoff = raw.times[partition_bounds(len(raw.times))["policy"][1] - 1]
    po = mature_policy_origins(raw, origins["policy"], cfg, target, cutoff)
    learning, validation, views = ([], [], [])
    for i, cond in enumerate(lock["development_conditions"]):
        tl = normalized_timeline(bundle, cond["profile"], cond["arrival_seed"])
        mature = mature_policy_origins(tl, po, cfg, target, cutoff)
        f, loss, error, bank = collect_bank(bundle, tl, mature, forecast, default, effective)
        learning.append(
            (f, loss[:, [0]] - loss, mature, (error[:, [0]] - error) / bundle.scaler.scale[target])
        )
        views.append(np.full(len(mature), i, dtype=int))
        vf, vl, ve, _ = collect_bank(
            bundle, tl, origins["validation"], forecast, default, effective
        )
        validation.append((cond["name"], vf, vl, ve))
        save_numeric_arrays(
            out / f"validation-{cond['name']}.npz",
            {"features": vf, "objective": vl, "native_error": ve, "origins": origins["validation"]},
        )
        print(
            json.dumps(
                {
                    "stage": "bank",
                    "dataset": dataset,
                    "seed": seed,
                    "condition": cond["name"],
                    "mature_origins": len(mature),
                }
            ),
            flush=True,
        )
    f, g, ids, mae = [np.concatenate([v[i] for v in learning]) for i in range(4)]
    save_numeric_arrays(
        out / "learning-arrays.npz",
        {
            "features": f,
            "gains": g,
            "origins": ids,
            "mae_gains": mae,
            "condition_ids": np.concatenate(views),
        },
    )
    settings = lock["selected_settings"]
    cls = HonestGainRouter if settings.get("algorithm") in ["extra", "hist"] else HonestPlanRouter
    router = cls.fit(f, g, ids, settings, seed=seed, mae_gains=mae)
    save_plan_policy(router, out / "selected-router.joblib")
    records = {}
    for strength in lock["strengths"]:
        for margin in lock["margins"]:
            key = f"s{strength:g}_m{margin:g}"
            records[key] = {
                "strength": strength,
                "margin": margin,
                **evaluate_router(router, validation, strength, margin),
            }
    eligible = [k for k, r in records.items() if r["max_mae_regression"] <= 0.01]
    forced = min(eligible or list(records), key=lambda k: (-records[k]["mean_improvement"], k))
    choice = records[forced]
    admitted = choice["max_mae_regression"] <= 0.01 and choice["mean_improvement"] > 0
    selection = {
        "forced": forced,
        "selected": forced if admitted else "default",
        "records": records,
        "reference": reference,
        "router_diagnostics": router.diagnostics,
        "bank": bank,
        "scope": "selection uses development-arrival validation origins only; final cases never select",
    }
    write_json(out / "selection.json", selection)
    selection_hash = sha256_file(out / "selection.json")
    arrays = {}
    report = {"selection": selection, "conditions": {}}
    for cond in protocol["conditions"]:
        name = cond["name"]
        tl = normalized_timeline(bundle, cond["profile"], cond["arrival_seed"])
        test = bundle.test_origins
        y = tl.values[test + cfg["horizon"], target]
        features = initial_features(tl, test, cfg, forecast)
        selected = router.select(features, strength=choice["strength"], margin=choice["margin"])
        forced_result = execute_strategies(
            tl, test, cfg, forecast, bundle.acquisition_cost_proxy, bank, selected, default
        )
        simple_result = rollout(
            tl, test, cfg, *methods["validation_simple"], bundle.acquisition_cost_proxy
        )
        results = {
            "forced_plan": forced_result,
            "primary": forced_result if admitted else simple_result,
            "validation_simple": simple_result,
            "legacy_joint": rollout(
                tl, test, cfg, *methods["legacy_joint"], bundle.acquisition_cost_proxy
            ),
        }
        arrays[f"{name}__origins"] = test
        arrays[f"{name}__truth"] = y
        arrays[f"{name}__forced_choices"] = selected
        report["conditions"][name] = {}
        for method, result in results.items():
            score = outcomes(bundle, result, y, effective)
            for field in ["prediction", "actions", "cost", "wait_seconds", "acquired_count"]:
                arrays[f"{name}__{method}__{field}"] = result[field]
            for field in ["objective", "native_error"]:
                arrays[f"{name}__{method}__{field}"] = score[field]
            report["conditions"][name][method] = {k: float(v.mean()) for k, v in score.items()}
    if (
        sha256_file(out / "selection.json") != selection_hash
        or scientific_source_lock() != lock["source_lock"]
    ):
        raise ValueError("selection or code changed during confirmation")
    save_numeric_arrays(out / "evaluation-arrays.npz", arrays)
    report["selection_sha256"] = selection_hash
    report["evaluation_sha256"] = sha256_file(out / "evaluation-arrays.npz")
    write_json(out / "case-report.json", report)
    print(
        json.dumps(
            {
                "stage": "case_complete",
                "dataset": dataset,
                "seed": seed,
                "selected": selection["selected"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lock", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, required=True)
    a = p.parse_args()
    lock = json.loads(a.lock.read_text())
    lock["lock_sha256"] = sha256_file(a.lock)
    if (np.__version__, sklearn.__version__, torch.__version__) != ("2.3.5", "1.8.0", "2.10.0+cpu"):
        raise ValueError("pinned numerical dependencies required")
    if (
        a.dataset not in lock["new_source_declaration"]["datasets"]
        or a.seed not in lock["protocol"]["seeds"]
    ):
        raise ValueError("undeclared dataset or seed")
    torch.set_num_threads(2)
    confirm_case(lock, a.data, a.dataset, a.seed, a.out)
