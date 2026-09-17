"""Independently written decomposition baseline and arrival-aware residual model.

DLinear algorithm reference: Zeng et al., AAAI 2023, arXiv:2205.13504.
This implementation is not a claim of reproducing the paper's benchmark scores.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def decompose(x: Tensor, kernel_size: int) -> tuple[Tensor, Tensor]:
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError('moving-average kernel must be positive and odd')
    padding = (kernel_size - 1) // 2
    trend = F.avg_pool1d(F.pad(x.transpose(1, 2), (padding, padding), mode='replicate'),
                        kernel_size=kernel_size, stride=1).transpose(1, 2)
    return x - trend, trend


class DLinear(nn.Module):
    def __init__(self, lookback: int, channels: int, horizon: int, kernel_size: int = 25):
        super().__init__()
        if min(lookback, channels, horizon, kernel_size) < 1 or kernel_size % 2 == 0:
            raise ValueError('positive dimensions and odd kernel required')
        self.kernel_size = kernel_size
        self.seasonal = nn.Linear(lookback, horizon)
        self.trend = nn.Linear(lookback, horizon)
        # Uniform temporal initialization preserves a constant input exactly.
        nn.init.constant_(self.seasonal.weight, 1.0 / lookback)
        nn.init.constant_(self.trend.weight, 1.0 / lookback)
        nn.init.zeros_(self.seasonal.bias)
        nn.init.zeros_(self.trend.bias)

    def forward(self, x: Tensor) -> Tensor:
        seasonal, trend = decompose(x, self.kernel_size)
        return (self.seasonal(seasonal.transpose(1, 2)) +
                self.trend(trend.transpose(1, 2))).transpose(1, 2)


class BaselineForecaster(nn.Module):
    """Single fixed future target, using value channels only."""
    def __init__(self, lookback: int, channels: int, target_channel: int):
        super().__init__()
        if not 0 <= target_channel < channels:
            raise ValueError('target channel outside input')
        self.target_channel = target_channel
        self.dlinear = DLinear(lookback, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.dlinear(x[..., 0])[:, 0, self.target_channel]


class ArrivalForecaster(nn.Module):
    """Small cross-channel residual on top of a decomposition forecast.

    Temporal projections are shared across channels and feature types. Values,
    availability, log-age and known-prefix state all enter the residual branch.
    """
    def __init__(self, lookback: int, channels: int, target_channel: int,
                 metadata: bool = True):
        super().__init__()
        self.base = BaselineForecaster(lookback, channels, target_channel)
        self.metadata = metadata
        self.temporal = nn.Linear(lookback, 8)
        self.fusion = nn.Sequential(nn.Linear(channels * 4 * 8, 48), nn.GELU(), nn.Linear(48, 1))
        nn.init.normal_(self.fusion[-1].weight, std=.01)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, x: Tensor) -> Tensor:
        base = self.base(x)
        if not self.metadata:
            x = torch.cat((x[..., :1], torch.zeros_like(x[..., 1:])), dim=-1)
        projected = self.temporal(x.permute(0, 2, 3, 1)).flatten(1)
        return base + self.fusion(projected).squeeze(-1)
