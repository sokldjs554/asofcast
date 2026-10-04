"""Frozen development evaluation; ETTh1 results cannot promote a public model.

Run: PYTHONPATH=src:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python
     scripts/evaluate_posterior_risk.py --bundle artifacts/ett-m1 --out artifacts/posterior/seed42
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import sklearn
import torch

from asofcast.bundle import load_bundle
from asofcast.data import sha256_file, write_json
from asofcast.decision_evidence import paired_block_interval
from asofcast.experiment import build_examples
from asofcast.posterior_policy import NORMAL_ABSOLUTE_MEAN, ArrivalSurvival, PosteriorPolicy
from asofcast.posterior_risk import GaussianForecast
from asofcast.preprocessing import partition_bounds, split_origins
from asofcast.research_diagnosis import confirmatory_gate, rollout
from asofcast.research_learning import (
    LegacyJoint,
    SimplePolicy,
    TorchForecast,
    fit_residual_forecast,
)
from scripts.evaluate_research_diagnosis import normalized_timeline, outcomes

ROOT = Path(__file__).resolve().parents[1]


def validate_protocol(protocol):
    original = json.loads((ROOT / "configs/information_value_20260929.json").read_text())
    if (
        protocol.get("schema") != "asofcast.posterior-risk-development.v1"
        or protocol.get("development_only") is not True
        or protocol.get("dataset") != "ETTh1"
        or protocol.get("seeds") != [42, 43, 44]
        or protocol.get("objective") != original["objective"]
        or protocol.get("gate") != original["gate"]
        or protocol.get("conditions") != original["conditions"]
    ):
        raise ValueError(
            "development label and original costs, conditions and gate must be preserved"
        )


def code_hashes():
    paths = [
        *ROOT.glob("src/asofcast/*.py"),
        Path(__file__),
        ROOT / "scripts/evaluate_research_diagnosis.py",
    ]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(paths)}


def fit_and_select(bundle, protocol, out):
    cfg, raw = bundle.config, bundle.timeline
    target, costs = bundle.manifest["target_channel"], bundle.acquisition_cost_proxy
    origins = split_origins(len(raw.times), cfg["lookback"], cfg["horizon"], cfg["stride"])
    bounds = partition_bounds(len(raw.times))
    train_stop = bounds["train"][1]
    cutoff = raw.times[train_stop - 1]
    policy_cutoff = raw.times[bounds["policy"][1] - 1]
    timeline = normalized_timeline(bundle)
    for part, limit in [("train", cutoff), ("policy", policy_cutoff)]:
        o = origins[part]
        origins[part] = o[raw.arrivals[o + cfg["horizon"], target] <= limit]
    train_x, train_y = build_examples(timeline, origins["train"], cfg, target)
    val_x, val_y = build_examples(timeline, origins["validation"], cfg, target)
    train_flat = train_x.reshape((-1,) + train_x.shape[2:])
    val_flat = val_x.reshape((-1,) + val_x.shape[2:])
    train_labels = np.repeat(train_y, len(cfg["waits_seconds"]))
    val_labels = np.repeat(val_y, len(cfg["waits_seconds"]))
    base = TorchForecast(bundle.calibrated)
    forecasts = {"base": base}
    forecast_records = {}
    print(json.dumps({"stage": "legacy_forecasts"}), flush=True)
    for name, metadata in [("upgraded", True), ("value", False)]:
        forecasts[name], forecast_records[name] = fit_residual_forecast(
            base,
            train_flat,
            train_labels,
            val_flat,
            val_labels,
            protocol["forecast"],
            seed=cfg["seed"],
            metadata=metadata,
        )
    known = raw.arrivals[:train_stop] <= cutoff
    var_records = {}
    print(json.dumps({"stage": "var_forecasts"}), flush=True)
    for setting in protocol["var_candidates"]:
        order, ridge = setting["order"], setting["ridge"]
        name = f"var{order}_r{ridge:g}"
        m = GaussianForecast.fit(
            timeline.values[:train_stop],
            known=known,
            order=order,
            ridge=ridge,
            horizon=cfg["horizon"],
            target=target,
            seed=cfg["seed"],
        )
        forecasts[name] = m
        mae = float(np.mean(np.abs(m.predict(val_flat) - val_labels)))
        var_records[name] = dict(
            order=order, ridge=ridge, validation_mae=mae, known_training_transitions=m.training_rows
        )
        m.save(out / f"{name}.npz")
    forecast_name = min(var_records, key=lambda name: (var_records[name]["validation_mae"], name))
    forecaster = forecasts[forecast_name]
    arrivals = ArrivalSurvival.fit(
        raw.times[:train_stop],
        raw.arrivals[:train_stop],
        cutoff=cutoff,
        grid_seconds=raw.grid_seconds,
        lookback=cfg["lookback"],
    )
    with (out / "arrival-survival.npz").open("xb") as handle:
        np.savez_compressed(handle, delays=arrivals.delays)
    policy_x, policy_y = build_examples(timeline, origins["policy"], cfg, target)
    policy_flat = policy_x.reshape((-1,) + policy_x.shape[2:])
    moments = forecaster.moments(policy_flat)
    absolute = np.abs(moments.mean - np.repeat(policy_y, len(cfg["waits_seconds"])))
    calibration = float(
        np.clip(
            absolute.mean() / (NORMAL_ABSOLUTE_MEAN * np.sqrt(moments.variance).mean()), 0.05, 20.0
        )
    )
    print(
        json.dumps(
            {"stage": "simple_comparators", "forecast": forecast_name, "calibration": calibration}
        ),
        flush=True,
    )
    simple_records, simple_pairs = {}, {}
    for name, forecast in forecasts.items():
        print(json.dumps({"stage": "simple_forecaster", "name": name}), flush=True)
        for wait in range(len(cfg["waits_seconds"])):
            for kind in ("commit", "target", "oldest"):
                key = f"{name}__wait{wait}__{kind}"
                policy = SimplePolicy(wait, kind, target)
                result = rollout(timeline, origins["validation"], cfg, forecast, policy, costs)
                score = outcomes(bundle, result, val_y, protocol)
                simple_records[key] = {k: float(v.mean()) for k, v in score.items()}
                simple_pairs[key] = (forecast, policy)
    best_simple = min(simple_records, key=lambda key: (simple_records[key]["objective"], key))
    same_keys = [key for key in simple_records if key.startswith(forecast_name + "__")]
    same_simple = min(same_keys, key=lambda key: (simple_records[key]["objective"], key))
    old_keys = [
        key for key in simple_records if key.split("__")[0] in ("base", "upgraded", "value")
    ]
    old_simple = min(old_keys, key=lambda key: (simple_records[key]["objective"], key))
    policy_records, policy_pairs = {}, {}
    for multiplier in protocol["risk_multipliers"]:
        name = f"posterior_x{multiplier:g}"
        print(json.dumps({"stage": "posterior_validation", "name": name}), flush=True)
        policy = PosteriorPolicy(
            forecaster,
            arrivals,
            cfg["waits_seconds"],
            grid_seconds=raw.grid_seconds,
            weight=protocol["objective"]["cost_weight"],
            delay=protocol["objective"]["delay_weight"],
            risk_scale=calibration * multiplier,
            samples=protocol["arrival_samples"],
            seed=protocol["arrival_mc_seed"],
        )
        result = rollout(timeline, origins["validation"], cfg, forecaster, policy, costs)
        score = outcomes(bundle, result, val_y, protocol)
        policy_records[name] = {k: float(v.mean()) for k, v in score.items()}
        policy_pairs[name] = (forecaster, policy)
    eligible = [
        name
        for name in policy_records
        if policy_records[name]["native_error"]
        <= simple_records[best_simple]["native_error"] * 1.01
    ]
    forced = min(
        eligible or list(policy_records), key=lambda key: (policy_records[key]["objective"], key)
    )
    accepted = (
        forced in eligible
        and policy_records[forced]["objective"] < simple_records[best_simple]["objective"]
    )
    primary = policy_pairs[forced] if accepted else simple_pairs[best_simple]
    selection = dict(
        split="validation",
        seed=cfg["seed"],
        train_cutoff=int(cutoff),
        policy_cutoff=int(policy_cutoff),
        origins={k: len(v) for k, v in origins.items()},
        forecast_candidates=var_records,
        legacy_forecast_selection=forecast_records,
        posterior_forecast=forecast_name,
        gaussian_risk_level_calibration=calibration,
        calibration_warning="heuristic level calibration; not proof of accurate marginal information value",
        simple_candidates=simple_records,
        posterior_candidates=policy_records,
        validation_simple=best_simple,
        same_forecast_simple=same_simple,
        original_validation_simple=old_simple,
        forced_candidate=forced,
        selected=forced if accepted else best_simple,
        selected_posterior=accepted,
    )
    write_json(out / "selection.json", selection)
    methods = {
        "primary": primary,
        "forced_posterior": policy_pairs[forced],
        "validation_simple": simple_pairs[best_simple],
        "same_forecast_simple": simple_pairs[same_simple],
        "original_validation_simple": simple_pairs[old_simple],
        "legacy_joint": (base, LegacyJoint(bundle)),
        "same_forecast_commit": (forecaster, SimplePolicy()),
        "base_commit": (base, SimplePolicy()),
    }
    return methods, selection


def evaluate(bundle_path, out, protocol):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    validate_protocol(protocol)
    bundle = load_bundle(bundle_path)
    cfg = bundle.config
    if (
        cfg["seed"] not in protocol["seeds"]
        or cfg["lookback"] != 48
        or cfg["horizon"] != 6
        or cfg["stride"] != 4
        or cfg["waits_seconds"] != [0, 1800, 3600]
        or cfg["target"] != "OT"
        or bundle.timeline.grid_seconds != 3600
        or bundle.report["source"]["sha256"] != sha256_file(ROOT / "data/raw/ETTh1.csv")
    ):
        raise ValueError("the original ETTh1 configuration and source bytes are required")
    torch.set_num_threads(cfg["cpu_threads"])
    out.mkdir(parents=True)
    write_json(out / "protocol.json", protocol)
    frozen = code_hashes()
    started = time.monotonic()
    print(json.dumps({"stage": "validation_fit", "seed": cfg["seed"]}), flush=True)
    methods, selection = fit_and_select(bundle, protocol, out)
    print(
        json.dumps(
            {
                "stage": "selected",
                "seed": cfg["seed"],
                "primary": selection["selected"],
                "posterior_forecast": selection["posterior_forecast"],
                "elapsed": round(time.monotonic() - started, 1),
            }
        ),
        flush=True,
    )
    reports, arrays = {}, {}
    origins = bundle.test_origins
    for condition in protocol["conditions"]:
        timeline = normalized_timeline(bundle, condition["profile"], condition["arrival_seed"])
        truth = timeline.values[origins + cfg["horizon"], bundle.manifest["target_channel"]]
        prefix = condition["name"]
        reports[prefix] = {}
        arrays[f"{prefix}__origins"] = origins
        arrays[f"{prefix}__truth"] = truth
        arrays[f"{prefix}__origin_times"] = timeline.times[origins]
        for name, (forecast, policy) in methods.items():
            result = rollout(
                timeline, origins, cfg, forecast, policy, bundle.acquisition_cost_proxy
            )
            score = outcomes(bundle, result, truth, protocol)
            reports[prefix][name] = {k: float(v.mean()) for k, v in score.items()}
            for k, v in {**result, **score}.items():
                arrays[f"{prefix}__{name}__{k}"] = v
        print(
            json.dumps(
                {
                    "stage": "evaluated",
                    "condition": prefix,
                    "seed": cfg["seed"],
                    "primary_loss": reports[prefix]["primary"]["objective"],
                    "strong_simple_loss": reports[prefix]["validation_simple"]["objective"],
                    "forced_posterior_loss": reports[prefix]["forced_posterior"]["objective"],
                    "same_forecast_simple_loss": reports[prefix]["same_forecast_simple"][
                        "objective"
                    ],
                    "elapsed": round(time.monotonic() - started, 1),
                }
            ),
            flush=True,
        )
        write_json(out / "progress.json", reports)
    if code_hashes() != frozen:
        raise RuntimeError("source changed during evaluation")
    with (out / "paired-losses.npz").open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    report = dict(
        schema=protocol["schema"],
        development_only=True,
        general_goal_achieved=False,
        scope="Already exposed ETTh1, simulated arrivals/costs; not independent confirmation",
        seed=cfg["seed"],
        protocol=protocol,
        selection=selection,
        runs=reports,
        source_files=frozen,
        bundle_manifest_sha256=sha256_file(Path(bundle_path) / "manifest.json"),
        input_source=bundle.report["source"],
        costs=bundle.acquisition_cost_proxy.tolist(),
        target_scale=float(bundle.scaler.scale[bundle.manifest["target_channel"]]),
        raw_sha256=sha256_file(out / "paired-losses.npz"),
        runtime=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            torch=torch.__version__,
            sklearn=sklearn.__version__,
        ),
        elapsed_seconds=time.monotonic() - started,
    )
    write_json(out / "report.json", report)
    return report


def aggregate(directories, output):
    paths = list(map(Path, directories))
    reports = {}
    arrays = {}
    for path in paths:
        report = json.loads((path / "report.json").read_text())
        validate_protocol(report["protocol"])
        seed = report["seed"]
        if seed in reports:
            raise ValueError("duplicate training seed")
        if sha256_file(path / "paired-losses.npz") != report["raw_sha256"]:
            raise ValueError("raw hash mismatch")
        reports[seed] = report
        with np.load(path / "paired-losses.npz", allow_pickle=False) as data:
            arrays[seed] = {key: data[key] for key in data.files}
    if sorted(reports) != [42, 43, 44]:
        raise ValueError("three distinct training seed results required")
    protocol = reports[42]["protocol"]
    gate = protocol["gate"]
    cells = {}
    diagnostics = {}
    for condition in protocol["conditions"]:
        name = condition["name"]
        key = "ETTh1/" + name
        if not all(
            np.array_equal(arrays[42][name + "__origins"], arrays[s][name + "__origins"])
            for s in reports
        ):
            raise ValueError("paired origins differ")
        comparisons = {}
        diagnostics[key] = {}
        for comparator in [
            "legacy_joint",
            "validation_simple",
            "same_forecast_simple",
            "original_validation_simple",
        ]:
            for candidate in ["primary", "forced_posterior"]:
                delta = np.stack(
                    [
                        arrays[s][f"{name}__{comparator}__objective"]
                        - arrays[s][f"{name}__{candidate}__objective"]
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
                    [reports[s]["runs"][name][comparator]["objective"] for s in reports]
                )
                interval["relative_improvement"] = interval["mean"] / max(baseline, 1e-12)
                interval["sensitivity_block6"] = paired_block_interval(
                    delta, block=6, repeats=gate["bootstrap_repeats"], seed=gate["bootstrap_seed"]
                )
                if candidate == "primary":
                    comparisons[comparator] = interval
                diagnostics[key][candidate + "/" + comparator] = interval
        regressions = [
            max(
                reports[s]["runs"][name]["primary"]["native_error"]
                / reports[s]["runs"][name][c]["native_error"]
                - 1
                for c in ("legacy_joint", "validation_simple")
            )
            for s in [42, 43, 44]
        ]
        cells[key] = dict(comparisons=comparisons, mae_regressions=regressions)
    numeric = confirmatory_gate(cells, list(cells), min_relative=0.01, max_regression=0.01)
    result = dict(
        development_only=True,
        general_goal_achieved=False,
        scope="These are already exposed development data; numeric gate is diagnostic, never promotion",
        numeric_development_gate=numeric,
        cells=cells,
        diagnostics=diagnostics,
        selections={str(s): reports[s]["selection"] for s in reports},
    )
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    write_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/posterior_risk_development_20261002.json"
    )
    parser.add_argument("--aggregate", nargs="+", type=Path)
    args = parser.parse_args()
    if args.aggregate:
        aggregate(args.aggregate, args.out)
    else:
        if args.bundle is None:
            parser.error("--bundle is required unless --aggregate is used")
        evaluate(args.bundle, args.out, json.loads(args.protocol.read_text()))
