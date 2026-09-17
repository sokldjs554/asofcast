"""Low-capacity causal correction for a strong DLinear forecast.

The calibrator keeps the trained decomposition forecast intact and learns only a
small ridge-regularized residual from information available in the current
snapshot: latest values, their one-slot deltas, data age, arrival flag, and the
base prediction. It deliberately does not receive future arrivals or targets.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from asofcast.models import BaselineForecaster
from asofcast.training import predict


class StalenessCalibratedForecaster(nn.Module):
    def __init__(self, lookback: int, channels: int, target_channel: int):
        super().__init__()
        if lookback < 2 or channels < 1 or not 0 <= target_channel < channels:
            raise ValueError('invalid calibrator dimensions')
        self.base = BaselineForecaster(lookback, channels, target_channel)
        self.channels = channels
        features = channels * 4 + 1
        self.register_buffer('feature_mean', torch.zeros(features))
        self.register_buffer('feature_scale', torch.ones(features))
        self.correction = nn.Linear(features, 1)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    @classmethod
    def from_baseline(cls, baseline: BaselineForecaster, *, lookback: int,
                      channels: int, target_channel: int) -> 'StalenessCalibratedForecaster':
        model = cls(lookback, channels, target_channel)
        model.base = copy.deepcopy(baseline)
        model.eval()
        return model

    def calibration_features(self, x: torch.Tensor, base_prediction: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] < 2 or x.shape[2] != self.channels or x.shape[3] != 4:
            raise ValueError('calibrator expects [batch, lookback>=2, channels, 4]')
        if base_prediction.shape != (len(x),):
            raise ValueError('one base prediction per example required')
        current_values = x[:, -1, :, 0]
        one_step_delta = current_values - x[:, -2, :, 0]
        current_age = x[:, -1, :, 2]
        current_observed = x[:, -1, :, 1]
        return torch.cat((current_values, one_step_delta, current_age,
                          current_observed, base_prediction[:, None]), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        features = self.calibration_features(x, base)
        normalized = (features - self.feature_mean) / self.feature_scale
        return base + self.correction(normalized).squeeze(-1)


def fit_ridge_calibrator(model: StalenessCalibratedForecaster, x: np.ndarray,
                         y: np.ndarray, *, ridge: float) -> dict:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if x.ndim != 4 or y.shape != (len(x),) or len(x) < 2:
        raise ValueError('nonempty [N,L,C,4] inputs and one target per example required')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(ridge) or ridge <= 0:
        raise ValueError('finite training data and positive ridge are required')

    base = predict(model.base, x)
    with torch.inference_mode():
        features = model.calibration_features(torch.from_numpy(x), torch.from_numpy(base)).cpu().numpy()
    mean = features.mean(axis=0, dtype=np.float64)
    scale = np.maximum(features.std(axis=0, dtype=np.float64), 1e-5)
    design = (features.astype(np.float64) - mean) / scale
    design = np.concatenate((np.ones((len(design), 1)), design), axis=1)
    residual = y.astype(np.float64) - base.astype(np.float64)
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ residual)

    with torch.no_grad():
        model.feature_mean.copy_(torch.from_numpy(mean.astype(np.float32)))
        model.feature_scale.copy_(torch.from_numpy(scale.astype(np.float32)))
        model.correction.bias.copy_(torch.tensor([coefficients[0]], dtype=torch.float32))
        model.correction.weight.copy_(torch.from_numpy(coefficients[1:].astype(np.float32))[None, :])
    model.eval()
    fitted = predict(model, x)
    return {
        'ridge': float(ridge),
        'train_mae_before': float(np.mean(np.abs(base - y))),
        'train_mae_after': float(np.mean(np.abs(fitted - y))),
        'feature_count': int(features.shape[1]),
    }
