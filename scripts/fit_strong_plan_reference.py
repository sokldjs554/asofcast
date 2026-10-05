"""Preserve the entire strong-simple pool, omitting unused posterior-policy ablations."""

from pathlib import Path

import numpy as np

from asofcast.data import write_json
from asofcast.experiment import build_examples
from asofcast.posterior_risk import GaussianForecast
from asofcast.preprocessing import partition_bounds, split_origins
from asofcast.research_diagnosis import rollout
from asofcast.research_learning import (
    LegacyJoint,
    SimplePolicy,
    TorchForecast,
    fit_residual_forecast,
)
from scripts.evaluate_research_diagnosis import normalized_timeline, outcomes
from scripts.posterior_candidate_suite import TargetAvailabilityPolicy
from scripts.run_terminal_plan_confirmation import save_plan_policy


def fit_strong_reference(bundle, protocol, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    cfg, raw = bundle.config, bundle.timeline
    target, costs = bundle.manifest["target_channel"], bundle.acquisition_cost_proxy
    origins = split_origins(len(raw.times), cfg["lookback"], cfg["horizon"], cfg["stride"])
    bounds = partition_bounds(len(raw.times))
    train_stop = bounds["train"][1]
    cutoff = raw.times[train_stop - 1]
    tl = normalized_timeline(bundle)
    po = origins["train"]
    po = po[raw.arrivals[po + cfg["horizon"], target] <= cutoff]
    tx, ty = build_examples(tl, po, cfg, target)
    vx, vy = build_examples(tl, origins["validation"], cfg, target)
    tf, vf = tx.reshape((-1,) + tx.shape[2:]), vx.reshape((-1,) + vx.shape[2:])
    tlabels, vlabels = (
        np.repeat(ty, len(cfg["waits_seconds"])),
        np.repeat(vy, len(cfg["waits_seconds"])),
    )
    base = TorchForecast(bundle.calibrated)
    forecasts, forecast_records, var_records = {"base": base}, {}, {}
    for name, metadata in [("upgraded", True), ("value", False)]:
        forecasts[name], forecast_records[name] = fit_residual_forecast(
            base,
            tf,
            tlabels,
            vf,
            vlabels,
            protocol["forecast"],
            seed=cfg["seed"],
            metadata=metadata,
        )
    known = raw.arrivals[:train_stop] <= cutoff
    for setting in protocol["var_candidates"]:
        order, ridge = setting["order"], setting["ridge"]
        name = f"var{order}_r{ridge:g}"
        model = GaussianForecast.fit(
            tl.values[:train_stop],
            known=known,
            order=order,
            ridge=ridge,
            horizon=cfg["horizon"],
            target=target,
            seed=cfg["seed"],
        )
        forecasts[name] = model
        var_records[name] = {
            "order": order,
            "ridge": ridge,
            "validation_mae": float(np.abs(model.predict(vf) - vlabels).mean()),
            "known_training_transitions": model.training_rows,
        }
        model.save(out / f"{name}.npz")
    gaussian_name = min(var_records, key=lambda n: (var_records[n]["validation_mae"], n))
    pairs, records = {}, {}
    for name, forecast in forecasts.items():
        for wait in range(len(cfg["waits_seconds"])):
            for kind in ["commit", "target", "oldest"]:
                key = f"{name}__wait{wait}__{kind}"
                pair = forecast, SimplePolicy(wait, kind, target)
                result = rollout(tl, origins["validation"], cfg, *pair, costs)
                records[key] = {
                    k: float(v.mean()) for k, v in outcomes(bundle, result, vy, protocol).items()
                }
                pairs[key] = pair
    best = min(records, key=lambda k: (records[k]["objective"], k))
    old_keys = [k for k in records if k.split("__")[0] in ["base", "upgraded", "value"]]
    old = min(old_keys, key=lambda k: (records[k]["objective"], k))
    additional_pairs = {"reference_simple": pairs[best]}
    additional_records = {"reference_simple": records[best]}
    for prefix, forecast in [("old", pairs[old][0]), ("gaussian", forecasts[gaussian_name])]:
        for wait in [0, 1, 2]:
            key = f"{prefix}_target_available_{wait}"
            pair = forecast, TargetAvailabilityPolicy(wait, target)
            result = rollout(tl, origins["validation"], cfg, *pair, costs)
            additional_records[key] = {
                k: float(v.mean()) for k, v in outcomes(bundle, result, vy, protocol).items()
            }
            additional_pairs[key] = pair
    strong = min(additional_records, key=lambda k: (additional_records[k]["objective"], k))
    reference = {
        "validation_simple": best,
        "original_validation_simple": old,
        "posterior_forecast": gaussian_name,
        "legacy_forecast_selection": forecast_records,
        "forecast_candidates": var_records,
        "simple_candidates": records,
    }
    selection = {
        "reference": reference,
        "strong_simple": strong,
        "additional_simple_records": additional_records,
        "scope": "unchanged static 63-candidate and 6 availability-aware pool; unused posterior ablations omitted",
    }
    save_plan_policy(additional_pairs[strong][0], out / "forecaster.joblib")
    save_plan_policy(additional_pairs[strong][1], out / "default-policy.joblib")
    write_json(out / "selection.json", selection)
    return {
        "legacy_joint": (base, LegacyJoint(bundle)),
        "validation_simple": additional_pairs[strong],
    }, selection
