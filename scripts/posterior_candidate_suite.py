"""Matched-forecast ablations and a frozen cross-dataset research suite.

No result is promoted automatically. The covariance proxy used with the legacy
forecaster is an ablation, not its exact predictive distribution.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from asofcast.bundle import load_bundle
from asofcast.data import sha256_file, write_json
from asofcast.decision_evidence import paired_block_interval
from asofcast.experiment import build_examples
from asofcast.posterior_policy import NORMAL_ABSOLUTE_MEAN, PosteriorPolicy
from asofcast.preprocessing import partition_bounds, split_origins
from asofcast.research_diagnosis import confirmatory_gate, rollout
from asofcast.research_learning import LegacyJoint
from scripts.evaluate_posterior_risk import fit_and_select
from scripts.evaluate_research_diagnosis import normalized_timeline, outcomes

ROOT = Path(__file__).resolve().parents[1]


class TargetAvailabilityPolicy:
    """A stronger simple comparator: do not pay to wait for a known target."""

    def __init__(self, wait_step, target):
        self.wait_step, self.target = wait_step, target

    def choose(self, x, prediction, steps, acquired, costs):
        present = (x[:, -1, self.target, 1] >= 0.5) | acquired[:, self.target]
        return np.where(present, -1, np.where(steps < self.wait_step, -2, self.target)).astype(int)


def select_family(records):
    baseline = records["simple"]
    valid = [
        name
        for name, r in records.items()
        if np.isfinite(r["objective"])
        and np.isfinite(r["native_error"])
        and r["native_error"] <= baseline["native_error"] * 1.01
    ]
    if "simple" not in valid:
        raise ValueError("finite simple comparator required")
    return min(valid, key=lambda n: (records[n]["objective"], n != "simple", n))


def check_result_alignment(origins):
    values = list(origins.values())
    if not values or not all(np.array_equal(values[0], x) for x in values[1:]):
        raise ValueError("paired origins must match exactly")


def source_hashes():
    paths = [
        *ROOT.glob("src/asofcast/*.py"),
        Path(__file__),
        ROOT / "scripts/evaluate_posterior_risk.py",
        ROOT / "scripts/evaluate_research_diagnosis.py",
        ROOT / "scripts/prepare_posterior_sources.py",
    ]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in paths if p.exists()}


def fit_family(bundle, protocol, out):
    reference_dir = out / "reference-fit"
    reference_dir.mkdir()
    reference, ref_selection = fit_and_select(bundle, protocol, reference_dir)
    cfg, raw = bundle.config, bundle.timeline
    costs, target = bundle.acquisition_cost_proxy, bundle.manifest["target_channel"]
    timeline = normalized_timeline(bundle)
    origins = split_origins(len(raw.times), cfg["lookback"], cfg["horizon"], cfg["stride"])
    truth = timeline.values[origins["validation"] + cfg["horizon"], target]
    old = reference["original_validation_simple"][0]
    gaussian, reference_policy = reference["forced_posterior"]
    pairs = {"reference_simple": reference["validation_simple"]}
    records = {
        "reference_simple": ref_selection["simple_candidates"][ref_selection["validation_simple"]]
    }
    same_old = {"original_static": reference["original_validation_simple"]}
    same_new = {"gaussian_static": reference["same_forecast_simple"]}
    for prefix, forecast in [("old", old), ("gaussian", gaussian)]:
        for wait in [0, 1, 2]:
            name = f"{prefix}_target_available_{wait}"
            pair = (forecast, TargetAvailabilityPolicy(wait, target))
            pairs[name] = pair
            result = rollout(timeline, origins["validation"], cfg, *pair, costs)
            score = outcomes(bundle, result, truth, protocol)
            records[name] = {k: float(v.mean()) for k, v in score.items()}
            (same_old if prefix == "old" else same_new)[name] = pair
    best_simple = min(records, key=lambda n: (records[n]["objective"], n))
    all_simple_records = dict(records)
    records["simple"] = records[best_simple]
    pairs["simple"] = pairs[best_simple]
    for collection, name in [
        (same_old, "same_original_simple"),
        (same_new, "same_gaussian_simple"),
    ]:
        scores = {}
        for key, pair in collection.items():
            scores[key] = float(
                outcomes(
                    bundle,
                    rollout(timeline, origins["validation"], cfg, *pair, costs),
                    truth,
                    protocol,
                )["objective"].mean()
            )
        key = min(scores, key=lambda n: (scores[n], n))
        pairs[name] = collection[key]
    policy_cutoff = raw.times[partition_bounds(len(raw.times))["policy"][1] - 1]
    po = origins["policy"]
    po = po[raw.arrivals[po + cfg["horizon"], target] <= policy_cutoff]
    px, py = build_examples(timeline, po, cfg, target)
    flat = px.reshape((-1,) + px.shape[2:])
    y = np.repeat(py, len(cfg["waits_seconds"]))
    level = float(
        np.clip(
            np.abs(old.predict(flat) - y).mean()
            / (NORMAL_ABSOLUTE_MEAN * np.sqrt(gaussian.moments(flat).variance).mean()),
            0.05,
            20.0,
        )
    )
    family_records = {"simple": records["simple"]}
    policy_names = []
    for multiplier in protocol["risk_multipliers"]:
        name = f"policy_only_x{multiplier:g}"
        policy_names.append(name)
        policy = PosteriorPolicy(
            gaussian,
            reference_policy.arrivals,
            cfg["waits_seconds"],
            grid_seconds=raw.grid_seconds,
            weight=protocol["objective"]["cost_weight"],
            delay=protocol["objective"]["delay_weight"],
            risk_scale=level * multiplier,
            samples=protocol["arrival_samples"],
            seed=protocol["arrival_mc_seed"],
        )
        pairs[name] = (old, policy)
        score = outcomes(
            bundle,
            rollout(timeline, origins["validation"], cfg, *pairs[name], costs),
            truth,
            protocol,
        )
        family_records[name] = {k: float(v.mean()) for k, v in score.items()}
    combined = reference["forced_posterior"]
    pairs["combined"] = combined
    score = outcomes(
        bundle, rollout(timeline, origins["validation"], cfg, *combined, costs), truth, protocol
    )
    family_records["combined"] = {k: float(v.mean()) for k, v in score.items()}
    selected = select_family(family_records)
    policy_only = min(policy_names, key=lambda n: (family_records[n]["objective"], n))
    print(
        json.dumps(
            {
                "family_selected": selected,
                "policy_only": policy_only,
                "simple": best_simple,
                "validation_losses": {n: r["objective"] for n, r in family_records.items()},
            }
        ),
        flush=True,
    )
    selection = dict(
        reference=ref_selection,
        strong_simple=best_simple,
        additional_simple_records=all_simple_records,
        family_records=family_records,
        selected=selected,
        policy_only=policy_only,
        policy_only_risk_level=level,
        calibration_scope="policy partition only; proxy risk, not calibrated marginal gain",
    )
    write_json(out / "selection.json", selection)
    methods = {
        "primary": pairs[selected],
        "policy_only": pairs[policy_only],
        "combined": combined,
        "forecast_only": (gaussian, LegacyJoint(bundle)),
        "legacy_joint": reference["legacy_joint"],
        "validation_simple": pairs["simple"],
        "original_validation_simple": reference["original_validation_simple"],
        "same_original_simple": pairs["same_original_simple"],
        "same_gaussian_simple": pairs["same_gaussian_simple"],
    }
    return methods, selection


def evaluate_case(bundle_path, out, protocol, dataset, *, development=False):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    original = json.loads((ROOT / "configs/information_value_20260929.json").read_text())
    if (
        protocol["gate"] != original["gate"]
        or protocol["conditions"] != original["conditions"]
        or protocol["objective"]["cost_weight"] != 0.03
        or protocol["objective"]["delay_weight"] != 0.02
    ):
        raise ValueError("original conditions, weights and gate must be preserved")
    meta = (protocol["development_datasets"] if development else protocol["confirmatory_datasets"])[
        dataset
    ]
    bundle = load_bundle(bundle_path)
    cfg = bundle.config
    grid = meta["grid_seconds"]
    effective = {**protocol, "objective": {**protocol["objective"], "deadline_seconds": grid}}
    if (
        cfg["seed"] not in [42, 43, 44]
        or cfg["lookback"] != 48
        or cfg["horizon"] != 6
        or cfg["stride"] != 4
        or cfg["waits_seconds"] != [0, grid / 2, grid]
        or cfg["target"] != meta["target"]
        or bundle.timeline.grid_seconds != grid
        or bundle.report["source"]["sha256"] != meta["prepared_sha256"]
    ):
        raise ValueError("bundle/source/configuration differs from frozen dataset contract")
    torch.set_num_threads(2)
    out.mkdir(parents=True)
    frozen = source_hashes()
    started = time.monotonic()
    write_json(out / "protocol.json", protocol)
    methods, selection = fit_family(bundle, effective, out)
    origins = bundle.test_origins
    arrays = {}
    runs = {}
    for condition in protocol["conditions"]:
        key = condition["name"]
        timeline = normalized_timeline(bundle, condition["profile"], condition["arrival_seed"])
        truth = timeline.values[origins + cfg["horizon"], bundle.manifest["target_channel"]]
        arrays[key + "__origins"] = origins
        arrays[key + "__truth"] = truth
        arrays[key + "__origin_times"] = timeline.times[origins]
        runs[key] = {}
        for name, pair in methods.items():
            result = rollout(timeline, origins, cfg, *pair, bundle.acquisition_cost_proxy)
            score = outcomes(bundle, result, truth, effective)
            runs[key][name] = {k: float(v.mean()) for k, v in score.items()}
            for field, value in {**result, **score}.items():
                arrays[key + "__" + name + "__" + field] = value
        print(
            json.dumps(
                {
                    "dataset": dataset,
                    "seed": cfg["seed"],
                    "condition": key,
                    "losses": {
                        k: runs[key][k]["objective"]
                        for k in ["primary", "policy_only", "combined", "validation_simple"]
                    },
                }
            ),
            flush=True,
        )
    if frozen != source_hashes():
        raise RuntimeError("source changed during execution")
    with (out / "paired-losses.npz").open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    report = dict(
        schema=protocol["schema"],
        dataset=dataset,
        development_only=development,
        seed=cfg["seed"],
        selection=selection,
        protocol=protocol,
        dataset_metadata=meta,
        source_files=frozen,
        runs=runs,
        costs=bundle.acquisition_cost_proxy.tolist(),
        target_scale=float(bundle.scaler.scale[bundle.manifest["target_channel"]]),
        raw_sha256=sha256_file(out / "paired-losses.npz"),
        bundle_manifest_sha256=sha256_file(Path(bundle_path) / "manifest.json"),
        elapsed_seconds=time.monotonic() - started,
        runtime=dict(
            python=platform.python_version(), numpy=np.__version__, torch=torch.__version__
        ),
        promotion_pending="aggregate all predeclared datasets and seeds; never deploy automatically",
    )
    write_json(out / "report.json", report)
    return report


def aggregate_suite(root, protocol, *, development=False):
    root = Path(root)
    if (root / "aggregate.json").exists():
        raise FileExistsError(root / "aggregate.json")
    datasets = (
        protocol["development_datasets"] if development else protocol["confirmatory_datasets"]
    )
    reports, arrays = {}, {}
    for dataset in datasets:
        for seed in [42, 43, 44]:
            path = root / f"{dataset}-{seed}"
            report = json.loads((path / "report.json").read_text())
            if (
                report["protocol"] != protocol
                or report["dataset"] != dataset
                or report["seed"] != seed
                or report["development_only"] != development
                or sha256_file(path / "paired-losses.npz") != report["raw_sha256"]
            ):
                raise ValueError("report provenance or raw integrity mismatch")
            reports[dataset, seed] = report
            with np.load(path / "paired-losses.npz", allow_pickle=False) as data:
                arrays[dataset, seed] = {k: data[k] for k in data.files}
    gate = protocol["gate"]
    cells = {}
    diagnostics = {}
    for dataset in datasets:
        for condition in protocol["conditions"]:
            name = condition["name"]
            key = dataset + "/" + name
            check_result_alignment(
                {s: arrays[dataset, s][name + "__origins"] for s in [42, 43, 44]}
            )
            comparisons = {}
            diagnostics[key] = {}
            for comparator in [
                "legacy_joint",
                "validation_simple",
                "original_validation_simple",
                "same_original_simple",
                "same_gaussian_simple",
            ]:
                for candidate in ["primary", "policy_only", "combined", "forecast_only"]:
                    delta = np.stack(
                        [
                            arrays[dataset, s][name + "__" + comparator + "__objective"]
                            - arrays[dataset, s][name + "__" + candidate + "__objective"]
                            for s in [42, 43, 44]
                        ]
                    )
                    interval = paired_block_interval(
                        delta,
                        block=gate["block_origins"],
                        repeats=gate["bootstrap_repeats"],
                        seed=gate["bootstrap_seed"],
                    )
                    baseline = np.mean(
                        [
                            reports[dataset, s]["runs"][name][comparator]["objective"]
                            for s in [42, 43, 44]
                        ]
                    )
                    interval["relative_improvement"] = interval["mean"] / max(baseline, 1e-12)
                    interval["sensitivity_block6"] = paired_block_interval(
                        delta,
                        block=6,
                        repeats=gate["bootstrap_repeats"],
                        seed=gate["bootstrap_seed"],
                    )
                    diagnostics[key][candidate + "/" + comparator] = interval
                    if candidate == "primary":
                        comparisons[comparator] = interval
            regressions = [
                max(
                    reports[dataset, s]["runs"][name]["primary"]["native_error"]
                    / max(reports[dataset, s]["runs"][name][c]["native_error"], 1e-12)
                    - 1
                    for c in ["legacy_joint", "validation_simple"]
                )
                for s in [42, 43, 44]
            ]
            cells[key] = dict(
                comparisons=comparisons,
                mae_regressions=regressions,
                origins=len(arrays[dataset, 42][name + "__origins"]),
            )
    expected = [d + "/" + c["name"] for d in datasets for c in protocol["conditions"]]
    numerical = confirmatory_gate(
        cells,
        expected,
        min_relative=gate["minimum_relative_improvement"],
        max_regression=gate["mae_regression_limit"],
    )
    result = dict(
        schema=protocol["schema"],
        development_only=development,
        general_goal_achieved=not development and numerical["passed"],
        gate=numerical,
        cells=cells,
        diagnostics=diagnostics,
        scope="predeclared datasets trained separately; simulated arrivals/costs; not universal or operational superiority",
    )
    write_json(root / "aggregate.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    p = json.loads(args.protocol.read_text())
    if args.aggregate:
        aggregate_suite(args.out, p, development=args.development)
    else:
        evaluate_case(args.bundle, args.out, p, args.dataset, development=args.development)
