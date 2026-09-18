"""End-to-end experiment: four temporal partitions and explicit evidence export."""
from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict, Field

from asofcast import __version__
from asofcast.acquisition import AcquisitionValueModel, acquisition_features, acquisition_gain_targets
from asofcast.calibration import StalenessCalibratedForecaster, fit_ridge_calibrator
from asofcast.data import load_source, sha256_file, write_json
from asofcast.metrics import random_matched_score, score
from asofcast.models import ArrivalForecaster, BaselineForecaster
from asofcast.policy import GainPolicy, choose_steps, gain_targets, policy_features, select_threshold, validate_waits
from asofcast.preprocessing import TrainScaler, partition_bounds, simulate_arrivals, split_origins
from asofcast.timeline import Timeline
from asofcast.training import predict, seed_everything, train_regressor


class RunConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    lookback: int = Field(default=48, ge=4, le=512)
    horizon: int = Field(default=6, ge=1, le=720)
    stride: int = Field(default=4, ge=1, le=1000)
    waits_seconds: list[float] = Field(default_factory=lambda: [0, 1800, 3600])
    epochs: int = Field(default=25, ge=1, le=5000)
    policy_epochs: int = Field(default=60, ge=1, le=5000)
    batch_size: int = Field(default=128, ge=1, le=4096)
    learning_rate: float = Field(default=.001, gt=0, le=1)
    seed: int = Field(default=42, ge=0, le=2**31-1)
    arrival_seed: int = Field(default=811, ge=0, le=2**31-1)
    delay_cost: float = Field(default=.02, ge=0, le=1000)
    thresholds: list[float] = Field(default_factory=lambda: [0., .01, .02, .05, .1, .2, 1e6])
    calibration_ridges: list[float] = Field(default_factory=lambda: [1., 10., 100., 1000., 10000., 100000.])
    acquisition_epochs: int = Field(default=40, ge=1, le=5000)
    acquisition_cost_weight: float = Field(default=.03, ge=0, le=1000)
    acquisition_pareto_weights: list[float] = Field(default_factory=lambda: [0., .01, .03, .05, .1])
    target: str = 'OT'
    cpu_threads: int = Field(default=2, ge=1, le=16)


def build_examples(timeline: Timeline, origins: np.ndarray, cfg: dict,
                   target_channel: int) -> tuple[np.ndarray, np.ndarray]:
    if len(origins) == 0 or np.max(origins) + cfg['horizon'] >= len(timeline.times):
        raise ValueError('training/evaluation examples require recorded future labels')
    x = np.stack([np.stack([timeline.snapshot(int(origin), wait, cfg['lookback'], cfg['horizon']).features()
                           for wait in cfg['waits_seconds']]) for origin in origins])
    y = timeline.values[origins + cfg['horizon'], target_channel].astype(np.float32)
    return x, y


def build_policy_states(x: np.ndarray, predictions: np.ndarray, waits: list[float]) -> np.ndarray:
    return np.stack([policy_features(x[:, step], predictions[:, step], 1 - wait / waits[-1])
                     for step, wait in enumerate(waits)], axis=1)


def _predict_decisions(model, x: np.ndarray) -> np.ndarray:
    return predict(model, x.reshape((-1,) + x.shape[2:])).reshape(x.shape[:2])


def _acquisition_cost_proxy(raw: Timeline, train_stop: int, cutoff: float) -> np.ndarray:
    arrivals = raw.arrivals[:train_stop]
    event_times = raw.times[:train_stop, None].astype(float)
    known = np.isfinite(arrivals) & (arrivals <= cutoff)
    delays = arrivals - event_times
    medians = np.zeros(arrivals.shape[1], dtype=np.float64)
    fallback = float(raw.grid_seconds)
    for channel in range(arrivals.shape[1]):
        values = delays[known[:, channel], channel]
        medians[channel] = float(np.median(values)) if len(values) else fallback
    maximum = float(np.max(medians))
    normalized = medians / maximum if maximum > 1e-9 else np.zeros_like(medians)
    return (.2 + .8 * normalized).astype(np.float32)


def _counterfactual_predictions(timeline: Timeline, origins: np.ndarray, cfg: dict,
                                model, x: np.ndarray, steps=None) -> tuple[np.ndarray, np.ndarray]:
    n, decisions, _, channels, _ = x.shape
    eligible = x[:, :, -1, :, 1] < .5
    result = np.full((n, decisions, channels), np.nan, dtype=np.float32)
    selected_steps = range(decisions) if steps is None else steps
    for step in selected_steps:
        wait = cfg['waits_seconds'][step]
        for channel in range(channels):
            indices = np.flatnonzero(eligible[:, step, channel])
            if len(indices) == 0:
                continue
            counterfactual_x = np.stack([
                timeline.snapshot(int(origins[i]), wait, cfg['lookback'], cfg['horizon'],
                                  acquired_channels=[channel]).features()
                for i in indices
            ])
            result[indices, step, channel] = predict(model, counterfactual_x)
    return result, eligible


def _predicted_acquisition_gains(model: AcquisitionValueModel, x: np.ndarray,
                                 current_predictions: np.ndarray, eligible: np.ndarray,
                                 cost_proxy: np.ndarray, remaining_fraction: float) -> np.ndarray:
    channels = x.shape[2]
    result = np.full((len(x), channels), -np.inf, dtype=np.float32)
    for channel in range(channels):
        indices = np.flatnonzero(eligible[:, channel])
        if len(indices) == 0:
            continue
        candidate = np.full(len(indices), channel, dtype=np.int64)
        features = acquisition_features(
            x[indices], current_predictions[indices], candidate, cost_proxy, remaining_fraction)
        result[indices, channel] = predict(model, features)
    return result


def _acquisition_metrics(y: np.ndarray, current_predictions: np.ndarray,
                         counterfactual_predictions: np.ndarray, eligible: np.ndarray,
                         predicted_gains: np.ndarray, cost_proxy: np.ndarray,
                         cost_weight: float, target_scale: float) -> tuple[dict, np.ndarray]:
    n, channels = counterfactual_predictions.shape
    if eligible.shape != (n, channels) or predicted_gains.shape != (n, channels):
        raise ValueError('acquisition evaluation schema mismatch')
    current_error = np.abs(current_predictions - y)
    realized_gain = np.full((n, channels), -np.inf, dtype=np.float32)
    for channel in range(channels):
        indices = np.flatnonzero(eligible[:, channel])
        if len(indices):
            realized_gain[indices, channel] = (
                current_error[indices]
                - np.abs(counterfactual_predictions[indices, channel] - y[indices])
            )
    net = np.where(eligible, predicted_gains - cost_weight * cost_proxy[None, :], -np.inf)
    best = np.argmax(net, axis=1)
    best_net = net[np.arange(n), best]
    chosen = np.where(best_net > 0, best, -1).astype(np.int64)
    selected = current_predictions.copy()
    acquired_rows = np.flatnonzero(chosen >= 0)
    if len(acquired_rows):
        selected[acquired_rows] = counterfactual_predictions[acquired_rows, chosen[acquired_rows]]

    oracle_matrix = np.where(eligible, realized_gain, -np.inf)
    oracle_best = np.argmax(oracle_matrix, axis=1)
    oracle_gain = oracle_matrix[np.arange(n), oracle_best]
    oracle_choice = np.where(oracle_gain > 0, oracle_best, -1).astype(np.int64)
    oracle_prediction = current_predictions.copy()
    oracle_rows = np.flatnonzero(oracle_choice >= 0)
    if len(oracle_rows):
        oracle_prediction[oracle_rows] = counterfactual_predictions[oracle_rows, oracle_choice[oracle_rows]]

    native_immediate_error = current_error * target_scale
    native_selected_error = np.abs(selected - y) * target_scale
    native_oracle_error = np.abs(oracle_prediction - y) * target_scale
    selected_gain = native_immediate_error - native_selected_error
    oracle_native_gain = native_immediate_error - native_oracle_error
    positive_oracle = oracle_choice >= 0
    top1 = float(np.mean(chosen[positive_oracle] == oracle_choice[positive_oracle])) if positive_oracle.any() else 0.0
    selected_cost = np.where(chosen >= 0, cost_proxy[np.maximum(chosen, 0)], 0.0)
    finite_pairs = eligible & np.isfinite(predicted_gains) & np.isfinite(realized_gain)
    gain_mae = float(np.mean(np.abs(predicted_gains[finite_pairs] - realized_gain[finite_pairs]))) if finite_pairs.any() else 0.0
    metrics = {
        'n': int(n),
        'immediate_mae': float(np.mean(native_immediate_error)),
        'learned_mae': float(np.mean(native_selected_error)),
        'oracle_mae': float(np.mean(native_oracle_error)),
        'acquisition_rate': float(np.mean(chosen >= 0)),
        'eligible_case_fraction': float(np.mean(eligible.any(axis=1))),
        'mean_cost_proxy': float(np.mean(selected_cost)),
        'mean_realized_gain': float(np.mean(selected_gain)),
        'mean_oracle_gain': float(np.mean(oracle_native_gain)),
        'regret_to_oracle': float(np.mean(oracle_native_gain - selected_gain)),
        'oracle_top1_hit_rate': top1,
        'candidate_gain_mae_standardized': gain_mae,
        'cost_weight': float(cost_weight),
        'scope': 'one-shot origin-slot acquisition; target used only for retrospective evaluation',
    }
    return metrics, chosen


def evaluate(y: np.ndarray, predictions: dict, states: np.ndarray, policy: GainPolicy,
             threshold: float, waits: list[float], scale: float, mean: float,
             policy_model: str = 'calibrated') -> tuple[dict, np.ndarray]:
    native_y = y.astype(float) * scale + mean
    metrics = {}
    for name, pred in predictions.items():
        native_pred = pred.astype(float) * scale + mean
        for step in range(len(waits)):
            suffix = 'immediate' if step == 0 else ('deadline' if step == len(waits)-1 else f'fixed_{int(waits[step])}s')
            metrics[f'{name}_{suffix}'] = score(native_y, native_pred, np.full(len(y), step, dtype=int), waits)
    chosen = choose_steps(states, policy, threshold, waits)
    native_policy = predictions[policy_model].astype(float) * scale + mean
    metrics[f'{policy_model}_learned'] = score(native_y, native_policy, chosen, waits)
    metrics[f'{policy_model}_random_matched'] = random_matched_score(native_y, native_policy, chosen, waits)
    return metrics, chosen


def run_experiment(csv_path: Path, output_dir: Path, config: dict, *, source_kind: str) -> dict:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'refusing to overwrite experiment: {output_dir}')
    started = time.perf_counter()
    cfg = RunConfig.model_validate(config).model_dump()
    waits = validate_waits(cfg['waits_seconds']).tolist()
    raw, source = load_source(csv_path, source_kind=source_kind, arrival_seed=cfg['arrival_seed'])
    if cfg['target'] not in raw.columns:
        raise ValueError('target column is missing')
    if waits[-1] >= cfg['horizon'] * raw.grid_seconds:
        raise ValueError('deadline must be strictly before the fixed target')
    target = raw.columns.index(cfg['target'])
    bounds = partition_bounds(len(raw.times))
    origins = split_origins(len(raw.times), cfg['lookback'], cfg['horizon'], cfg['stride'])
    scaler_cutoff = float(raw.times[bounds['train'][1] - 1])
    scaler = TrainScaler.fit(raw.values, raw.arrivals, scaler_cutoff)
    standardized = Timeline(raw.times, scaler.transform(raw.values), raw.arrivals, raw.columns)
    removed_labels = {}
    for name in ('train', 'policy'):
        cutoff = raw.times[bounds[name][1] - 1]
        candidates = origins[name]
        arrived = raw.arrivals[candidates + cfg['horizon'], target] <= cutoff
        origins[name] = candidates[arrived]
        removed_labels[name] = int((~arrived).sum())
        if len(origins[name]) < 2:
            raise ValueError(f'not enough arrived labels in {name} partition')
    examples = {name: build_examples(standardized, indices, cfg, target) for name, indices in origins.items()}
    torch.set_num_threads(cfg['cpu_threads'])
    seed_everything(cfg['seed'])
    models = {'dlinear': BaselineForecaster(cfg['lookback'], len(raw.columns), target),
              'value_only': ArrivalForecaster(cfg['lookback'], len(raw.columns), target, metadata=False),
              'arrival': ArrivalForecaster(cfg['lookback'], len(raw.columns), target)}
    train_x, train_y = examples['train']
    training_x = train_x.reshape((-1,) + train_x.shape[2:])
    training_y = np.repeat(train_y, len(waits))
    history = {}
    for name, model in models.items():
        history[name] = train_regressor(model, training_x, training_y, epochs=cfg['epochs'],
                                        learning_rate=cfg['learning_rate'], batch_size=cfg['batch_size'],
                                        seed=cfg['seed'])
    if not cfg['calibration_ridges'] or any((not np.isfinite(value) or value <= 0) for value in cfg['calibration_ridges']):
        raise ValueError('calibration_ridges must contain positive finite values')
    val_x, val_y = examples['validation']
    calibration_candidates = []
    for ridge in cfg['calibration_ridges']:
        candidate = StalenessCalibratedForecaster.from_baseline(
            models['dlinear'], lookback=cfg['lookback'], channels=len(raw.columns), target_channel=target)
        fit = fit_ridge_calibrator(candidate, training_x, training_y, ridge=float(ridge))
        val_prediction = _predict_decisions(candidate, val_x)
        val_mae = float(np.mean(np.abs(val_prediction - val_y[:, None])))
        calibration_candidates.append({'ridge': float(ridge), 'validation_mae_standardized': val_mae, **fit})
    selected_record = min(calibration_candidates, key=lambda row: (row['validation_mae_standardized'], row['ridge']))
    calibrated = StalenessCalibratedForecaster.from_baseline(
        models['dlinear'], lookback=cfg['lookback'], channels=len(raw.columns), target_channel=target)
    selected_fit = fit_ridge_calibrator(calibrated, training_x, training_y, ridge=selected_record['ridge'])
    models['calibrated'] = calibrated
    history['calibration'] = {'selected_ridge': selected_record['ridge'], 'fit': selected_fit,
                              'validation_candidates': calibration_candidates}
    policy_x, policy_y = examples['policy']
    policy_pred = _predict_decisions(models['calibrated'], policy_x)
    policy_states = build_policy_states(policy_x, policy_pred, waits)
    features = policy_states[:, :-1].reshape(-1, policy_states.shape[-1])
    gains = gain_targets(policy_pred, policy_y).reshape(-1)
    seed_everything(cfg['seed'] + 1)
    policy = GainPolicy(features.shape[-1])
    policy.fit_scaling(features, gains)
    history['policy'] = train_regressor(policy, features, gains, epochs=cfg['policy_epochs'],
                                        learning_rate=.003, batch_size=cfg['batch_size'], seed=cfg['seed'] + 1)

    if (not cfg['acquisition_pareto_weights']
            or any(not np.isfinite(weight) or weight < 0 for weight in cfg['acquisition_pareto_weights'])):
        raise ValueError('acquisition_pareto_weights must contain nonnegative finite values')
    acquisition_cost_proxy = _acquisition_cost_proxy(
        raw, bounds['train'][1], scaler_cutoff)
    policy_counterfactual, policy_eligible = _counterfactual_predictions(
        standardized, origins['policy'], cfg, models['calibrated'], policy_x)
    acquisition_feature_rows = []
    acquisition_gain_rows = []
    policy_gain_matrix = acquisition_gain_targets(
        policy_pred.reshape(-1),
        policy_counterfactual.reshape(len(policy_y) * len(waits), len(raw.columns)),
        np.repeat(policy_y, len(waits)),
    ).reshape(len(policy_y), len(waits), len(raw.columns))
    for step, wait in enumerate(waits):
        remaining = 1 - wait / waits[-1]
        for channel in range(len(raw.columns)):
            indices = np.flatnonzero(policy_eligible[:, step, channel])
            if len(indices) == 0:
                continue
            acquisition_feature_rows.append(acquisition_features(
                policy_x[indices, step],
                policy_pred[indices, step],
                np.full(len(indices), channel, dtype=np.int64),
                acquisition_cost_proxy,
                remaining,
            ))
            acquisition_gain_rows.append(policy_gain_matrix[indices, step, channel])
    if not acquisition_feature_rows:
        raise ValueError('no eligible active-acquisition examples in policy partition')
    acquisition_x = np.concatenate(acquisition_feature_rows, axis=0)
    acquisition_y = np.concatenate(acquisition_gain_rows, axis=0).astype(np.float32)
    seed_everything(cfg['seed'] + 2)
    acquisition = AcquisitionValueModel(acquisition_x.shape[1])
    acquisition.fit_scaling(acquisition_x, acquisition_y)
    history['acquisition'] = train_regressor(
        acquisition, acquisition_x, acquisition_y,
        epochs=cfg['acquisition_epochs'], learning_rate=.003,
        batch_size=cfg['batch_size'], seed=cfg['seed'] + 2)

    val_pred = _predict_decisions(models['calibrated'], val_x)
    val_states = build_policy_states(val_x, val_pred, waits)
    threshold, validation_table = select_threshold(val_states, val_pred, val_y, policy, waits,
                                                  cfg['delay_cost'], cfg['thresholds'])
    test_x, test_y = examples['test']
    test_predictions = {name: _predict_decisions(model, test_x) for name, model in models.items()}
    test_states = build_policy_states(test_x, test_predictions['calibrated'], waits)
    target_scale, target_mean = float(scaler.scale[target]), float(scaler.mean[target])
    test_metrics, chosen = evaluate(test_y, test_predictions, test_states, policy, threshold, waits,
                                    target_scale, target_mean)

    test_counterfactual, test_eligible_all = _counterfactual_predictions(
        standardized, origins['test'], cfg, models['calibrated'], test_x, steps=[0])
    immediate_predictions = test_predictions['calibrated'][:, 0]
    immediate_eligible = test_eligible_all[:, 0]
    predicted_acquisition_gain = _predicted_acquisition_gains(
        acquisition, test_x[:, 0], immediate_predictions, immediate_eligible,
        acquisition_cost_proxy, 1.0)
    acquisition_policy, acquisition_chosen = _acquisition_metrics(
        test_y, immediate_predictions, test_counterfactual[:, 0], immediate_eligible,
        predicted_acquisition_gain, acquisition_cost_proxy,
        cfg['acquisition_cost_weight'], target_scale)
    acquisition_pareto = []
    for weight in cfg['acquisition_pareto_weights']:
        row, _ = _acquisition_metrics(
            test_y, immediate_predictions, test_counterfactual[:, 0], immediate_eligible,
            predicted_acquisition_gain, acquisition_cost_proxy, float(weight), target_scale)
        acquisition_pareto.append({
            'cost_weight': float(weight),
            'mae': row['learned_mae'],
            'acquisition_rate': row['acquisition_rate'],
            'mean_cost_proxy': row['mean_cost_proxy'],
            'mean_realized_gain': row['mean_realized_gain'],
        })

    outage_arrivals = simulate_arrivals(raw.times, len(raw.columns), cfg['arrival_seed'], profile='outage')
    shifted = Timeline(raw.times, standardized.values, outage_arrivals, raw.columns)
    shift_x, shift_y = build_examples(shifted, origins['test'], cfg, target)
    shift_pred = {name: _predict_decisions(model, shift_x) for name, model in models.items()}
    shift_states = build_policy_states(shift_x, shift_pred['calibrated'], waits)
    shift_metrics, _ = evaluate(shift_y, shift_pred, shift_states, policy, threshold, waits,
                                target_scale, target_mean)
    partitions = {}
    for name, indices in origins.items():
        partitions[name] = {'samples': len(indices), 'first_origin_index': int(indices[0]),
                            'last_origin_index': int(indices[-1]),
                            'earliest_input_time': int(raw.times[indices[0] - cfg['lookback'] + 1]),
                            'latest_target_time': int(raw.times[indices[-1] + cfg['horizon']]),
                            'label_cutoff': int(raw.times[bounds[name][1]-1]) if name in ('train','policy') else None}
    run_id = hashlib.sha256((source['sha256'] + json.dumps(cfg, sort_keys=True)).encode()).hexdigest()[:16]
    error = np.abs(test_predictions['calibrated'] - test_y[:, None]) * target_scale
    report = {'version': __version__, 'run_id': run_id, 'status': 'local_experiment_completed',
              'source': source, 'config': cfg, 'partitions': partitions,
              'removed_unarrived_training_labels': removed_labels,
              'calibration_selection': {'split': 'validation', 'ridge': selected_record['ridge'],
                                        'objective': 'mean standardized MAE across decision times',
                                        'validation_candidates': calibration_candidates},
              'policy_selection': {'split': 'validation', 'forecaster': 'calibrated', 'threshold': threshold,
                                   'objective': 'normalized MAE + delay_cost * mean_wait/deadline',
                                   'validation_candidates': validation_table},
              'test_metrics': test_metrics, 'outage_test_metrics': shift_metrics,
              'acquisition_policy': acquisition_policy,
              'acquisition_pareto': acquisition_pareto,
              'acquisition_cost_proxy': {name: float(acquisition_cost_proxy[i])
                                         for i, name in enumerate(raw.columns)},
              'wait_feasibility': {'mean_error_reduction_full_wait_native': float((error[:,0]-error[:,-1]).mean()),
                                   'fraction_helped_by_full_wait': float(np.mean(error[:,-1] < error[:,0])),
                                   'hindsight_best_mae_native': float(error.min(axis=1).mean()),
                                   'hindsight_warning': 'uses future outcomes; not deployable'},
              'parameters': {**{name: sum(p.numel() for p in model.parameters()) for name, model in models.items()},
                             'acquisition': sum(p.numel() for p in acquisition.parameters())},
              'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                          'numpy': np.__version__, 'platform': platform.platform(), 'device': 'cpu',
                          'threads': cfg['cpu_threads']},
              'duration_seconds': time.perf_counter() - started,
              'paper_score_reproduced': False, 'cloud_deployed': False, 'remote_ci_run': False,
              'inference_optimized': False, 'mlflow_executed': False,
              'limitations': ['M1 scalar fixed-horizon experiment, not original paper protocol',
                              'Synthetic arrival timestamps, not real transport telemetry',
                              'Myopic one-step gain policy is not an optimal stopping algorithm',
                              'Active acquisition is a simulated origin-slot pull with train-derived relative cost, not real hardware control',
                              'Logical decision deadline excludes network and model runtime',
                              'Raw benchmark labels available retrospectively, including simulated drops']}
    if source_kind == 'synthetic':
        report['limitations'].insert(0, 'Measurements are synthetic too; this is a pipeline validation, not ETT evidence')
    output_dir.mkdir(parents=True)
    for name, model in models.items():
        torch.save(model.state_dict(), output_dir / f'{name}.pt')
    torch.save(policy.state_dict(), output_dir / 'policy.pt')
    torch.save(acquisition.state_dict(), output_dir / 'acquisition.pt')
    write_json(output_dir / 'config.json', cfg)
    write_json(output_dir / 'scaler.json', scaler.to_dict())
    write_json(output_dir / 'training_history.json', history)
    write_json(output_dir / 'report.json', report)
    np.savez_compressed(output_dir / 'replay.npz', times=raw.times, values=raw.values,
                        arrivals=raw.arrivals, columns=np.array(raw.columns), test_origins=origins['test'])
    result_rows = []
    for i, origin in enumerate(origins['test']):
        for k, wait in enumerate(waits):
            result_rows.append({'origin_index': int(origin), 'origin_time': int(raw.times[origin]),
                                'target_time': int(raw.times[origin] + cfg['horizon'] * raw.grid_seconds),
                                'wait_seconds': wait, 'target_actual': float(test_y[i]*target_scale+target_mean),
                                'dlinear': float(test_predictions['dlinear'][i,k]*target_scale+target_mean),
                                'value_only': float(test_predictions['value_only'][i,k]*target_scale+target_mean),
                                'arrival': float(test_predictions['arrival'][i,k]*target_scale+target_mean),
                                'calibrated': float(test_predictions['calibrated'][i,k]*target_scale+target_mean),
                                'policy_selected': bool(chosen[i] == k)})
    pd.DataFrame(result_rows).to_csv(output_dir / 'test_predictions.csv', index=False)
    files = {p.name: sha256_file(p) for p in sorted(output_dir.iterdir()) if p.is_file()}
    write_json(output_dir / 'manifest.json', {'bundle_version': 2, 'run_id': run_id,
               'target_channel': target, 'channels': list(raw.columns), 'policy_features': features.shape[-1],
               'calibration_features': len(raw.columns) * 4 + 1,
               'acquisition_features': acquisition_x.shape[1],
               'acquisition_cost_proxy': acquisition_cost_proxy.tolist(),
               'files': files, 'status': 'complete'})
    return report
