"""Immutable observation log and frozen-origin, arrival-filtered snapshots.

A received observation may fill a later missing slot, never an earlier slot.
Snapshot construction reads no measurements beyond the original forecast origin.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def _readonly(array: NDArray, dtype=None) -> NDArray:
    out = np.array(array, dtype=dtype, copy=True)
    out.flags.writeable = False
    return out


@dataclass(frozen=True)
class Snapshot:
    origin_time: int
    decision_time: float
    target_time: int
    values: NDArray[np.float64]
    observed: NDArray[np.bool_]
    known: NDArray[np.bool_]
    age_seconds: NDArray[np.float64]
    source_times: NDArray[np.int64]
    grid_seconds: int

    def features(self) -> NDArray[np.float32]:
        cap = len(self.values) + 1
        age = np.log1p(np.minimum(self.age_seconds / self.grid_seconds, cap)) / np.log1p(cap)
        return np.stack((self.values, self.observed, age, self.known), axis=-1).astype(np.float32)


@dataclass(frozen=True)
class Timeline:
    times: NDArray[np.int64]
    values: NDArray[np.float64]
    arrivals: NDArray[np.float64]
    columns: tuple[str, ...]

    def __post_init__(self) -> None:
        t, v, a = np.asarray(self.times), np.asarray(self.values), np.asarray(self.arrivals)
        if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all():
            raise ValueError('times must be a finite one-dimensional series of length >= 2')
        if not np.equal(t, np.floor(t)).all():
            raise ValueError('times must be integer seconds')
        gaps = np.diff(t)
        if (gaps <= 0).any() or not np.all(gaps == gaps[0]):
            raise ValueError('times must be strictly increasing on a regular grid')
        if v.ndim != 2 or v.shape[0] != len(t) or v.shape != a.shape or v.shape[1] < 1:
            raise ValueError('values and arrivals must have matching [time, channel] shape')
        if not np.isfinite(v).all():
            raise ValueError('raw values must be finite; missing arrivals use +infinity')
        if np.isnan(a).any() or (a < t[:, None]).any():
            raise ValueError('arrivals must not precede event time or contain NaN')
        if len(self.columns) != v.shape[1] or len(set(self.columns)) != len(self.columns):
            raise ValueError('unique channel names must match value columns')
        if any(not isinstance(c, str) or not c for c in self.columns):
            raise ValueError('channel names must be nonempty strings')
        object.__setattr__(self, 'times', _readonly(t, np.int64))
        object.__setattr__(self, 'values', _readonly(v, np.float64))
        object.__setattr__(self, 'arrivals', _readonly(a, np.float64))
        object.__setattr__(self, 'columns', tuple(self.columns))

    @property
    def grid_seconds(self) -> int:
        return int(self.times[1] - self.times[0])

    def snapshot(self, origin_index: int, wait_seconds: float, lookback: int,
                 horizon: int, acquired_channels=()) -> Snapshot:
        if not isinstance(origin_index, (int, np.integer)) or not isinstance(lookback, (int, np.integer)):
            raise ValueError('origin and lookback must be integers')
        if not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError('horizon must be a positive integer')
        if lookback < 1 or origin_index < lookback - 1 or origin_index >= len(self.times):
            raise ValueError('query falls outside the timeline')
        if not np.isfinite(wait_seconds) or wait_seconds < 0 or wait_seconds >= horizon * self.grid_seconds:
            raise ValueError('wait must be finite, nonnegative and strictly before the target')
        acquired = list(acquired_channels)
        if (any(isinstance(channel, bool) or not isinstance(channel, (int, np.integer)) for channel in acquired)
                or len(set(acquired)) != len(acquired)
                or any(channel < 0 or channel >= self.values.shape[1] for channel in acquired)):
            raise ValueError('acquired_channels must contain unique valid channel indices')
        start = origin_index - lookback + 1
        stop = origin_index + 1
        origin = int(self.times[origin_index])
        decision = origin + float(wait_seconds)
        values = self.values[start:stop]
        available = np.array(self.arrivals[start:stop] <= decision, copy=True)
        if acquired:
            # Active acquisition is an explicit pull of the frozen origin slot only.
            # It never exposes an event after the original forecast origin.
            available[-1, acquired] = True
        positions = np.arange(lookback)[:, None]
        last = np.maximum.accumulate(np.where(available, positions, -1), axis=0)
        known = last >= 0
        safe = np.maximum(last, 0)
        columns = np.arange(values.shape[1])[None, :]
        imputed = np.where(known, values[safe, columns], 0.0)
        source_times = np.where(known, self.times[start:stop][safe], -1)
        ages = np.where(known, decision - source_times, np.inf)
        return Snapshot(origin, decision, origin + horizon * self.grid_seconds,
                        _readonly(imputed), _readonly(available), _readonly(known),
                        _readonly(ages), _readonly(source_times), self.grid_seconds)
