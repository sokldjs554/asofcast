"""Train-cutoff scaling, strictly chronological partitions, and synthetic arrivals."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TrainScaler:
    mean: NDArray[np.float64]
    scale: NDArray[np.float64]
    cutoff: float
    counts: NDArray[np.int64]

    @classmethod
    def fit(cls, values: NDArray, arrivals: NDArray, cutoff: float) -> TrainScaler:
        v, a = np.asarray(values, dtype=float), np.asarray(arrivals, dtype=float)
        if v.ndim != 2 or v.shape != a.shape or not np.isfinite(v).all() or not np.isfinite(cutoff):
            raise ValueError('finite values, cutoff and matching arrays are required')
        if np.isnan(a).any():
            raise ValueError('NaN arrival time')
        visible = a <= cutoff
        counts = visible.sum(axis=0)
        if (counts == 0).any():
            raise ValueError('every channel needs an arrived training observation')
        mean = np.where(visible, v, 0.0).sum(axis=0) / counts
        variance = np.where(visible, (v - mean) ** 2, 0.0).sum(axis=0) / counts
        scale = np.sqrt(variance)
        scale = np.where(scale < 1e-8, 1.0, scale)
        return cls(mean, scale, float(cutoff), counts)

    def transform(self, values: NDArray) -> NDArray[np.float64]:
        return (np.asarray(values, dtype=float) - self.mean) / self.scale

    def inverse(self, values: NDArray) -> NDArray[np.float64]:
        return np.asarray(values, dtype=float) * self.scale + self.mean

    def to_dict(self) -> dict:
        return {'mean': self.mean.tolist(), 'scale': self.scale.tolist(),
                'cutoff': self.cutoff, 'counts': self.counts.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> TrainScaler:
        mean = np.asarray(data['mean'], dtype=float)
        scale = np.asarray(data['scale'], dtype=float)
        counts = np.asarray(data['counts'], dtype=np.int64)
        if mean.ndim != 1 or mean.shape != scale.shape or mean.shape != counts.shape:
            raise ValueError('invalid scaler dimensions')
        if not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any() or (counts < 1).any():
            raise ValueError('invalid scaler statistics')
        if not np.isfinite(data['cutoff']):
            raise ValueError('invalid scaler cutoff')
        return cls(mean, scale, float(data['cutoff']), counts)


def partition_bounds(n: int) -> dict[str, tuple[int, int]]:
    cuts = [0, int(n * .55), int(n * .75), int(n * .85), n]
    return dict(zip(('train', 'policy', 'validation', 'test'),
                    zip(cuts[:-1], cuts[1:], strict=True), strict=True))


def split_origins(n: int, lookback: int, horizon: int, stride: int = 1) -> dict[str, NDArray]:
    if min(n, lookback, horizon, stride) < 1:
        raise ValueError('all split parameters must be positive')
    result = {name: np.arange(lo + lookback - 1, hi - horizon, stride, dtype=np.int64)
              for name, (lo, hi) in partition_bounds(n).items()}
    if any(len(indices) < 2 for indices in result.values()):
        raise ValueError('series too short for four purged temporal partitions')
    return result


def simulate_arrivals(times: NDArray, channels: int, seed: int = 42,
                      profile: str = 'mixed') -> NDArray[np.float64]:
    """One global synthetic schedule. Raw values are deliberately not an argument."""
    times = np.asarray(times, dtype=np.int64)
    if len(times) < 2 or channels < 1 or profile not in ('mixed', 'outage'):
        raise ValueError('invalid arrival simulation configuration')
    rng = np.random.default_rng(seed)
    shape = (len(times), channels)
    u = rng.random(shape)
    delays = rng.lognormal(np.log(900), 1.0, shape)
    delays[u < .30] = 0.0
    long = u > .90
    delays[long] = rng.uniform(7200, 28800, long.sum())
    for start in range(0, len(times), 96):
        if rng.random() < (.20 if profile == 'mixed' else .65):
            channel = int(rng.integers(channels))
            width = min(12, len(times) - start)
            delays[start:start + width, channel] += rng.uniform(7200, 21600)
    delays[u > .995] = np.inf
    if profile == 'outage':
        delays = delays * 2.5
    return times[:, None].astype(float) + delays
