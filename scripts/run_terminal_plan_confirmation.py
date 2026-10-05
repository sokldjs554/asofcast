"""Frozen research: development selection, then two independent complete retrainings.

No automatic service promotion. Numerical dependencies must match the freeze.
"""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
from pathlib import Path

import joblib
import numpy as np
import sklearn
import torch

from asofcast.bundle import load_bundle
from asofcast.data import sha256_file, write_json
from asofcast.experiment import RunConfig, run_experiment
from asofcast.preprocessing import partition_bounds, split_origins
from asofcast.research_diagnosis import rollout
from asofcast.research_plan_policy import TerminalPlanPolicy, terminal_plan_examples
from scripts.evaluate_research_diagnosis import normalized_timeline, outcomes
from scripts.posterior_candidate_suite import fit_family

ROOT = Path(__file__).resolve().parents[1]


def source_lock():
    paths = [
        *ROOT.glob("src/asofcast/*.py"),
        *ROOT.glob("scripts/*.py"),
        ROOT / "configs/terminal_plan_confirmation_20261005.json",
        ROOT / "configs/ett_m1.json",
    ]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(paths)}


def validate_bundle_contract(bundle, cfg, meta, seed):
    if bundle.report["source"]["sha256"] != meta["prepared_sha256"]:
        raise ValueError("bundle source differs from declared source")
    expected = RunConfig(
        **{
            **cfg,
            "seed": seed,
            "target": meta["target"],
            "waits_seconds": [0, meta["grid_seconds"] / 2, meta["grid_seconds"]],
        }
    ).model_dump()
    if bundle.config != expected or bundle.timeline.grid_seconds != meta["grid_seconds"]:
        raise ValueError("bundle config differs from frozen training contract")


def save_plan_policy(policy, destination):
    destination = Path(destination)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=destination.name + ".", delete=False
    ) as handle:
        temp = Path(handle.name)
    try:
        joblib.dump(policy, temp, compress=3)
        joblib.load(temp)
        temp.replace(destination)
    except Exception as exc:
        raise ValueError("policy artifact integrity verification failed") from exc
    finally:
        temp.unlink(missing_ok=True)


def save_numeric_arrays(destination, arrays):
    destination = Path(destination)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=destination.name + ".", delete=False
    ) as handle:
        temp = Path(handle.name)
    try:
        with temp.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        with np.load(temp, allow_pickle=False) as loaded:
            if set(loaded.files) != set(arrays):
                raise ValueError("array artifact keys differ")
            for name, value in arrays.items():
                np.testing.assert_array_equal(loaded[name], value)
        temp.replace(destination)
    finally:
        temp.unlink(missing_ok=True)


def fit_candidates(bundle, protocol, out, depths):
    out.mkdir(parents=True)
    methods, reference = fit_family(bundle, protocol, out)
    cfg, raw = bundle.config, bundle.timeline
    target, costs = bundle.manifest["target_channel"], bundle.acquisition_cost_proxy
    timeline = normalized_timeline(bundle)
    origins = split_origins(len(raw.times), cfg["lookback"], cfg["horizon"], cfg["stride"])
    cutoff = raw.times[partition_bounds(len(raw.times))["policy"][1] - 1]
    po = origins["policy"]
    po = po[raw.arrivals[po + cfg["horizon"], target] <= cutoff]
    cap = protocol["plan_learning"]["policy_origin_cap"]
    if len(po) > cap:
        po = po[np.linspace(0, len(po) - 1, cap, dtype=int)]
    forecast = methods["same_original_simple"][0]
    print(json.dumps({"stage": "plan_labels", "origins": len(po)}), flush=True)
    f, g, e, plans = terminal_plan_examples(
        timeline, po, cfg, target, forecast, costs, seed=cfg["seed"] + 17, weight=0.03, delay=0.02
    )
    save_numeric_arrays(
        out / "learning-arrays.npz", {"features": f, "gains": g, "eligible": e, "origins": po}
    )
    vo = origins["validation"]
    y = timeline.values[vo + cfg["horizon"], target]
    baseline = outcomes(
        bundle, rollout(timeline, vo, cfg, *methods["validation_simple"], costs), y, protocol
    )
    records, candidates = {}, {}
    settings = protocol["plan_learning"]
    for candidate in depths:
        print(json.dumps({"stage": "plan_fit", "candidate": candidate}), flush=True)
        policy = TerminalPlanPolicy.fit(
            f,
            g,
            e,
            plans,
            cfg["waits_seconds"],
            settings["regressors"][candidate],
            seed=cfg["seed"],
        )
        for margin in settings["margins"]:
            policy.margin = margin
            result = rollout(timeline, vo, cfg, forecast, policy, costs)
            score = outcomes(bundle, result, y, protocol)
            key = f"{candidate}_margin{margin:g}"
            records[key] = {
                "candidate": candidate,
                "margin": margin,
                **{k: float(v.mean()) for k, v in score.items()},
                "relative_improvement": 1
                - float(score["objective"].mean() / baseline["objective"].mean()),
                "mae_regression": float(
                    score["native_error"].mean() / baseline["native_error"].mean() - 1
                ),
            }
        save_plan_policy(policy, out / f"plan-{candidate}.joblib")
        candidates[candidate] = policy
    valid = [k for k, r in records.items() if r["mae_regression"] <= 0.01]
    forced = min(
        valid or list(records), key=lambda k: (records[k]["objective"], -records[k]["margin"], k)
    )
    chosen = records[forced]
    policy = candidates[chosen["candidate"]]
    policy.margin = chosen["margin"]
    save_plan_policy(policy, out / "selected-plan.joblib")
    admitted = (
        chosen["mae_regression"] <= 0.01 and chosen["objective"] < baseline["objective"].mean()
    )
    selection = {
        "forced": forced,
        "selected": "forced_plan" if admitted else "validation_simple",
        "records": records,
        "validation_simple": {k: float(v.mean()) for k, v in baseline.items()},
        "reference": reference,
        "training_origins": len(po),
        "plans": plans,
    }
    methods["forced_plan"] = (forecast, policy)
    methods["previous_primary"] = methods["primary"]
    methods["primary"] = methods["forced_plan"] if admitted else methods["validation_simple"]
    write_json(out / "plan-selection.json", selection)
    return methods, selection


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["development", "confirmation"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--development-bundles", type=Path)
    ap.add_argument("--selection-lock", type=Path)
    a = ap.parse_args()
    torch.set_num_threads(2)
    if (
        np.__version__ != "2.3.5"
        or sklearn.__version__ != "1.8.0"
        or torch.__version__ != "2.10.0+cpu"
    ):
        raise ValueError("exact NumPy2.3.5/sklearn1.8.0/torch2.10.0+cpu required")
    protocol = json.loads((ROOT / "configs/terminal_plan_confirmation_20261005.json").read_text())
    lock = None
    if a.stage == "confirmation":
        lock = json.loads(a.selection_lock.read_text())
        if source_lock() != lock["source_lock"]:
            raise ValueError("code or configuration changed after selection freeze")
    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "source_lock": source_lock(),
    }
    if a.out.exists() and a.stage == "development" and any(a.out.iterdir()):
        raise FileExistsError(
            "development output contains existing evidence; use a fresh directory"
        )
    if (a.out / "environment.json").exists():
        if json.loads((a.out / "environment.json").read_text()) != environment:
            raise ValueError("existing environment provenance differs")
    if lock is not None and (a.out / "selection-lock.json").exists():
        if json.loads((a.out / "selection-lock.json").read_text()) != lock:
            raise ValueError("existing selection provenance differs")
    a.out.mkdir(parents=True, exist_ok=True)
    if not (a.out / "environment.json").exists():
        write_json(a.out / "environment.json", environment)
    if a.stage == "development":
        reports = {}
        for name in ["Taylor", "MSFT"]:
            case = a.out / name
            meta = protocol["development_datasets"][name]
            effective = {
                **protocol,
                "objective": {**protocol["objective"], "deadline_seconds": meta["grid_seconds"]},
            }
            bundle = load_bundle(a.development_bundles / f"{name}-42")
            _, selection = fit_candidates(
                bundle, effective, case, protocol["plan_learning"]["candidate_names"]
            )
            reports[name] = selection
            write_json(a.out / "development-progress.json", reports)
        depth_scores = {}
        for candidate in protocol["plan_learning"]["candidate_names"]:
            values = []
            for selection in reports.values():
                rs = [
                    r
                    for r in selection["records"].values()
                    if r["candidate"] == candidate and r["mae_regression"] <= 0.01
                ]
                values.append(max([r["relative_improvement"] for r in rs], default=-1))
            depth_scores[candidate] = float(np.mean(values))
        best = min(
            protocol["plan_learning"]["candidate_names"], key=lambda d: (-depth_scores[d], d)
        )
        lock = {
            "selected_candidate": best,
            "development_scores": depth_scores,
            "source_lock": source_lock(),
            "protocol_sha256": sha256_file(
                ROOT / "configs/terminal_plan_confirmation_20261005.json"
            ),
            "independent_sources": protocol["confirmatory_datasets"],
            "test_scores_inspected": False,
            "selection_scope": "Taylor/MSFT validation only",
        }
        write_json(a.out / "selection-lock.json", lock)
        write_json(a.out / "development-report.json", reports)
        print(json.dumps({"stage": "development_complete", **lock}), flush=True)
        return
    if not (a.out / "selection-lock.json").exists():
        write_json(a.out / "selection-lock.json", lock)
    cfg = json.loads((ROOT / "configs/ett_m1.json").read_text())
    reports = {}
    for name, meta in protocol["confirmatory_datasets"].items():
        source = a.data / f"{name}.csv"
        if sha256_file(source) != meta["prepared_sha256"]:
            raise ValueError("source differs from preregistration")
        effective = {
            **protocol,
            "objective": {**protocol["objective"], "deadline_seconds": meta["grid_seconds"]},
        }
        for seed in protocol["seeds"]:
            key = f"{name}-{seed}"
            case = a.out / key
            print(json.dumps({"stage": "confirmation_case", "case": key}), flush=True)
            dest = a.out / "models" / key
            if not dest.exists():
                run_experiment(
                    source,
                    dest,
                    {
                        **cfg,
                        "seed": seed,
                        "target": meta["target"],
                        "waits_seconds": [0, meta["grid_seconds"] / 2, meta["grid_seconds"]],
                    },
                    source_kind="user-provided-csv",
                )
            bundle = load_bundle(dest)
            validate_bundle_contract(bundle, cfg, meta, seed)
            if (case / "case-report.json").exists():
                previous = json.loads((case / "case-report.json").read_text())
                if (
                    previous.get("source_lock") != source_lock()
                    or previous.get("raw_sha256") != sha256_file(case / "evaluation-arrays.npz")
                    or previous.get("selected_policy_sha256")
                    != sha256_file(case / "selected-plan.joblib")
                ):
                    raise ValueError(
                        "cached case differs from frozen provenance; use a fresh output directory"
                    )
                with np.load(case / "evaluation-arrays.npz", allow_pickle=False) as data:
                    for field in data.files:
                        data[field]
                joblib.load(case / "selected-plan.joblib")
                reports[key] = previous
                continue
            methods, selection = fit_candidates(
                bundle, effective, case, [lock["selected_candidate"]]
            )
            arrays = {}
            report = {"selection": selection, "conditions": {}}
            origins = bundle.test_origins
            for condition in protocol["conditions"]:
                cname = condition["name"]
                timeline = normalized_timeline(
                    bundle, condition["profile"], condition["arrival_seed"]
                )
                truth = timeline.values[origins + cfg["horizon"], bundle.manifest["target_channel"]]
                arrays[cname + "__truth"] = truth
                arrays[cname + "__origins"] = origins
                report["conditions"][cname] = {}
                for method in [
                    "primary",
                    "forced_plan",
                    "previous_primary",
                    "legacy_joint",
                    "validation_simple",
                    "same_original_simple",
                ]:
                    result = rollout(
                        timeline,
                        origins,
                        bundle.config,
                        *methods[method],
                        bundle.acquisition_cost_proxy,
                    )
                    score = outcomes(bundle, result, truth, effective)
                    for field in [
                        "prediction",
                        "actions",
                        "cost",
                        "wait_seconds",
                        "acquired_count",
                    ]:
                        arrays[f"{cname}__{method}__{field}"] = result[field]
                    for field in ["objective", "native_error"]:
                        arrays[f"{cname}__{method}__{field}"] = score[field]
                    report["conditions"][cname][method] = {
                        **{k: float(v.mean()) for k, v in score.items()},
                        "wait_rate": float((result["wait_seconds"] > 0).mean()),
                        "pull_rate": float((result["cost"] > 0).mean()),
                    }
            save_numeric_arrays(case / "evaluation-arrays.npz", arrays)
            report["source_lock"] = source_lock()
            report["raw_sha256"] = sha256_file(case / "evaluation-arrays.npz")
            report["selected_policy_sha256"] = sha256_file(case / "selected-plan.joblib")
            write_json(case / "case-report.json", report)
            reports[key] = report
            write_json(a.out / "confirmation-progress.json", reports)
    if source_lock() != lock["source_lock"]:
        raise ValueError("source changed during evaluation")
    write_json(a.out / "confirmation-report.json", reports)
    print(json.dumps({"stage": "confirmation_complete", "cases": len(reports)}), flush=True)


if __name__ == "__main__":
    main()
