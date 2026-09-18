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
              'wait_feasibility': {'mean_error_reduction_full_wait_native': float((error[:,0]-error[:,-1]).mean()),
                                   'fraction_helped_by_full_wait': float(np.mean(error[:,-1] < error[:,0])),
                                   'hindsight_best_mae_native': float(error.min(axis=1).mean()),
                                   'hindsight_warning': 'uses future outcomes; not deployable'},
              'parameters': {name: sum(p.numel() for p in model.parameters()) for name, model in models.items()},
              'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                          'numpy': np.__version__, 'platform': platform.platform(), 'device': 'cpu',
                          'threads': cfg['cpu_threads']},
              'duration_seconds': time.perf_counter() - started,
              'paper_score_reproduced': False, 'cloud_deployed': False, 'remote_ci_run': False,
              'inference_optimized': False, 'mlflow_executed': False,
              'limitations': ['M1 scalar fixed-horizon experiment, not original paper protocol',
                              'Synthetic arrival timestamps, not real transport telemetry',
                              'Myopic one-step gain policy is not an optimal stopping algorithm',
                              'Logical decision deadline excludes network and model runtime',
                              'Raw benchmark labels available retrospectively, including simulated drops']}
    if source_kind == 'synthetic':
        report['limitations'].insert(0, 'Measurements are synthetic too; this is a pipeline validation, not ETT evidence')
    output_dir.mkdir(parents=True)
    for name, model in models.items():
        torch.save(model.state_dict(), output_dir / f'{name}.pt')
    torch.save(policy.state_dict(), output_dir / 'policy.pt')
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
               'files': files, 'status': 'complete'})
    return report
