"""Learn one-step expected error reduction, then stop causally before a deadline.

This myopic policy is a baseline, not an optimal stopping algorithm. Future
forecasts and outcomes create training labels but are never policy inputs.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from asofcast.training import predict


def gain_targets(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    predictions, targets = np.asarray(predictions), np.asarray(targets)
    if predictions.ndim != 2 or targets.shape != (len(predictions),) or predictions.shape[1] < 2:
        raise ValueError('predictions [examples, decisions] and scalar targets required')
    errors = np.abs(predictions - targets[:, None])
    return (errors[:, :-1] - errors[:, 1:]).astype(np.float32)


def policy_features(snapshot_features: np.ndarray, prediction: np.ndarray,
                    remaining_fraction: float) -> np.ndarray:
    x = np.asarray(snapshot_features, dtype=np.float32)
    p = np.asarray(prediction, dtype=np.float32)
    if x.ndim != 4 or x.shape[-1] != 4 or p.shape != (len(x),):
        raise ValueError('one CURRENT snapshot and prediction per example required')
    if not 0 <= remaining_fraction <= 1 or not np.isfinite(x).all() or not np.isfinite(p).all():
        raise ValueError('invalid current state')
    recent_availability = x[:, -min(4, x.shape[1]):, :, 1].mean(axis=1)
    mean_values = x[..., 0].mean(axis=1)
    return np.concatenate((x[:, -1].reshape(len(x), -1), recent_availability, mean_values,
                           p[:, None], np.full((len(x), 1), remaining_fraction, dtype=np.float32)),
                          axis=1).astype(np.float32)


class GainPolicy(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.register_buffer('mean', torch.zeros(features))
        self.register_buffer('scale', torch.ones(features))
        self.register_buffer('gain_mean', torch.tensor(0.0))
        self.register_buffer('gain_scale', torch.tensor(1.0))
        self.net = nn.Sequential(nn.Linear(features, 32), nn.Tanh(), nn.Linear(32, 1))

    def fit_scaling(self, x: np.ndarray, gains: np.ndarray) -> None:
        x, gains = np.asarray(x, dtype=np.float32), np.asarray(gains, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != len(self.mean) or gains.shape != (len(x),) or len(x) < 1:
            raise ValueError('invalid policy training shape')
        if not np.isfinite(x).all() or not np.isfinite(gains).all():
            raise ValueError('policy training values must be finite')
        with torch.no_grad():
            self.mean.copy_(torch.from_numpy(x.mean(axis=0)))
            self.scale.copy_(torch.from_numpy(np.maximum(x.std(axis=0), .01)))
            self.gain_mean.fill_(float(gains.mean()))
            self.gain_scale.fill_(max(float(gains.std()), .01))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / self.scale).squeeze(-1) * self.gain_scale + self.gain_mean


def validate_waits(waits: list[float] | np.ndarray) -> np.ndarray:
    waits = np.asarray(waits, dtype=float)
    if waits.ndim != 1 or len(waits) < 2 or not np.isfinite(waits).all():
        raise ValueError('finite one-dimensional decision grid required')
    if waits[0] != 0 or (np.diff(waits) <= 0).any():
        raise ValueError('decision grid must start at zero and strictly increase')
    return waits


def choose_steps(states: np.ndarray, policy: nn.Module, threshold: float,
                 waits: list[float] | np.ndarray) -> np.ndarray:
    waits = validate_waits(waits)
    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 3 or states.shape[1] != len(waits) or not np.isfinite(states).all():
        raise ValueError('states must be finite [examples, decisions, features]')
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError('threshold must be finite and nonnegative')
    chosen = np.full(len(states), len(waits) - 1, dtype=np.int64)
    active = np.ones(len(states), dtype=bool)
    for step in range(len(waits) - 1):
        indices = np.flatnonzero(active)
        if len(indices) == 0:
            break
        gains = predict(policy, states[indices, step])
        incremental_cost = threshold * (waits[step + 1] - waits[step]) / waits[-1]
        stopped = indices[gains <= incremental_cost]
        chosen[stopped] = step
        active[stopped] = False
    return chosen


def select_threshold(validation_states: np.ndarray, validation_predictions: np.ndarray,
                     validation_targets: np.ndarray, policy: nn.Module, waits: list[float],
                     delay_cost: float, candidates: list[float]) -> tuple[float, list[dict]]:
    from asofcast.metrics import score
    if not np.isfinite(delay_cost) or delay_cost < 0 or not candidates:
        raise ValueError('invalid validation selection configuration')
    table = []
    for threshold in candidates:
        steps = choose_steps(validation_states, policy, threshold, waits)
        metrics = score(validation_targets, validation_predictions, steps, waits)
        objective = metrics['mae'] + delay_cost * metrics['mean_wait_seconds'] / waits[-1]
        table.append({'threshold': float(threshold), 'objective': objective, **metrics})
    best = min(table, key=lambda item: (item['objective'], item['mean_wait_seconds'], item['threshold']))
    return best['threshold'], table
