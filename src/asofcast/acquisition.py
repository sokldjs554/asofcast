"""Causal value-of-information helpers for active sensor acquisition.

The learned model predicts error reduction from the CURRENT snapshot only.
Future targets create offline labels but never enter deployable features.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from asofcast.policy import policy_features
from asofcast.training import predict


def acquisition_gain_targets(current_predictions: np.ndarray,
                             counterfactual_predictions: np.ndarray,
                             targets: np.ndarray) -> np.ndarray:
    current = np.asarray(current_predictions, dtype=np.float32)
    counterfactual = np.asarray(counterfactual_predictions, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    if (current.ndim != 1 or y.shape != current.shape
            or counterfactual.ndim != 2 or counterfactual.shape[0] != len(current)):
        raise ValueError('current [N], counterfactual [N,C] and targets [N] are required')
    if not np.isfinite(current).all() or not np.isfinite(counterfactual).all() or not np.isfinite(y).all():
        raise ValueError('acquisition gain labels require finite arrays')
    before = np.abs(current - y)
    after = np.abs(counterfactual - y[:, None])
    return (before[:, None] - after).astype(np.float32)


def acquisition_features(snapshot_features: np.ndarray, prediction: np.ndarray,
                         channel_indices: np.ndarray, cost_proxy: np.ndarray,
                         remaining_fraction: float) -> np.ndarray:
    x = np.asarray(snapshot_features, dtype=np.float32)
    p = np.asarray(prediction, dtype=np.float32)
    channels = np.asarray(channel_indices)
    costs = np.asarray(cost_proxy, dtype=np.float32)
    if x.ndim != 4 or x.shape[-1] != 4 or p.shape != (len(x),):
        raise ValueError('current snapshots [N,L,C,4] and one prediction per row are required')
    n_channels = x.shape[2]
    if (channels.shape != (len(x),) or not np.issubdtype(channels.dtype, np.integer)
            or (channels < 0).any() or (channels >= n_channels).any()):
        raise ValueError('one valid integer candidate channel per row is required')
    if costs.shape != (n_channels,) or not np.isfinite(costs).all() or (costs < 0).any():
        raise ValueError('finite nonnegative cost proxy per channel is required')
    if not 0 <= remaining_fraction <= 1:
        raise ValueError('remaining fraction must lie in [0,1]')
    base = policy_features(x, p, remaining_fraction)
    selected = x[np.arange(len(x)), -1, channels, :]
    one_hot = np.zeros((len(x), n_channels), dtype=np.float32)
    one_hot[np.arange(len(x)), channels] = 1.0
    selected_cost = costs[channels, None]
    result = np.concatenate((base, selected, one_hot, selected_cost), axis=1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('acquisition features must be finite')
    return result


class AcquisitionValueModel(nn.Module):
    """Small MLP estimating one-shot absolute-error reduction per candidate sensor."""

    def __init__(self, features: int):
        super().__init__()
        if features < 1:
            raise ValueError('positive feature count required')
        self.register_buffer('mean', torch.zeros(features))
        self.register_buffer('scale', torch.ones(features))
        self.register_buffer('gain_mean', torch.tensor(0.0))
        self.register_buffer('gain_scale', torch.tensor(1.0))
        self.net = nn.Sequential(
            nn.Linear(features, 48),
            nn.GELU(),
            nn.Linear(48, 1),
        )

    def fit_scaling(self, x: np.ndarray, gains: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32)
        gains = np.asarray(gains, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != len(self.mean) or gains.shape != (len(x),) or len(x) < 1:
            raise ValueError('invalid acquisition training shape')
        if not np.isfinite(x).all() or not np.isfinite(gains).all():
            raise ValueError('acquisition training values must be finite')
        with torch.no_grad():
            self.mean.copy_(torch.from_numpy(x.mean(axis=0)))
            self.scale.copy_(torch.from_numpy(np.maximum(x.std(axis=0), .01)))
            self.gain_mean.fill_(float(gains.mean()))
            self.gain_scale.fill_(max(float(gains.std()), .01))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / self.scale).squeeze(-1) * self.gain_scale + self.gain_mean


def predict_candidate_gains(model: nn.Module, snapshot_features: np.ndarray,
                            prediction: float, cost_proxy: np.ndarray,
                            eligible: np.ndarray, remaining_fraction: float) -> np.ndarray:
    x = np.asarray(snapshot_features, dtype=np.float32)
    costs = np.asarray(cost_proxy, dtype=np.float32)
    eligible = np.asarray(eligible, dtype=bool)
    if x.ndim != 3 or x.shape[-1] != 4:
        raise ValueError('one current snapshot [L,C,4] is required')
    channels = x.shape[1]
    if costs.shape != (channels,) or eligible.shape != (channels,):
        raise ValueError('candidate schema mismatch')
    result = np.full(channels, -np.inf, dtype=np.float32)
    indices = np.flatnonzero(eligible)
    if len(indices) == 0:
        return result
    repeated = np.repeat(x[None, ...], len(indices), axis=0)
    predictions = np.full(len(indices), float(prediction), dtype=np.float32)
    features = acquisition_features(repeated, predictions, indices, costs, remaining_fraction)
    result[indices] = predict(model, features)
    return result


def select_joint_action(*, wait_gain: float, wait_cost: float,
                        acquisition_gains: np.ndarray, acquisition_costs: np.ndarray,
                        eligible: np.ndarray, acquisition_cost_weight: float) -> dict:
    gains = np.asarray(acquisition_gains, dtype=float)
    costs = np.asarray(acquisition_costs, dtype=float)
    eligible = np.asarray(eligible, dtype=bool)
    if gains.ndim != 1 or costs.shape != gains.shape or eligible.shape != gains.shape:
        raise ValueError('one gain/cost/eligibility value per channel is required')
    if (not np.isfinite(wait_gain) or not np.isfinite(wait_cost)
            or wait_cost < 0 or not np.isfinite(acquisition_cost_weight)
            or acquisition_cost_weight < 0 or not np.isfinite(costs).all() or (costs < 0).any()):
        raise ValueError('invalid joint action costs')
    wait_net = float(wait_gain - wait_cost)
    acquisition_net = np.full(len(gains), -np.inf, dtype=float)
    finite_gain = np.isfinite(gains)
    usable = eligible & finite_gain
    acquisition_net[usable] = gains[usable] - acquisition_cost_weight * costs[usable]
    if usable.any():
        best_channel = int(np.argmax(acquisition_net))
        best_net = float(acquisition_net[best_channel])
    else:
        best_channel = None
        best_net = float('-inf')
    if best_channel is not None and best_net > 0 and best_net >= wait_net:
        return {
            'action': 'ACQUIRE',
            'channel': best_channel,
            'net_utility': best_net,
            'wait_net_utility': wait_net,
        }
    if wait_net > 0:
        return {
            'action': 'WAIT',
            'channel': None,
            'net_utility': wait_net,
            'wait_net_utility': wait_net,
        }
    return {
        'action': 'COMMIT',
        'channel': None,
        'net_utility': max(0.0, best_net if np.isfinite(best_net) else 0.0),
        'wait_net_utility': wait_net,
    }
