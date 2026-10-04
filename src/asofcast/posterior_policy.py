"""Gaussian-risk sensor subsets and train-estimated arrival lookahead.

The risk is conditional on a fitted model, not an empirical performance claim.
The caller supplies no outcome labels or actual future arrival timestamps.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from asofcast.policy import validate_waits

NORMAL_ABSOLUTE_MEAN = np.sqrt(2.0 / np.pi)


def best_sensor_subset(
    variance, covariance, cross, eligible, costs, observation_noise, *, weight=0.03, risk_scale=1.0
):
    variance, covariance, cross, costs, noise = (
        np.asarray(x, dtype=np.float64)
        for x in (variance, covariance, cross, costs, observation_noise)
    )
    eligible = np.asarray(eligible)
    if (
        variance.ndim != 1
        or costs.ndim != 1
        or not 1 <= len(costs) <= 10
        or covariance.shape != (len(variance), len(costs), len(costs))
        or cross.shape != (len(variance), len(costs))
        or eligible.shape != cross.shape
        or eligible.dtype != bool
        or noise.shape != costs.shape
        or any(not np.isfinite(x).all() for x in (variance, covariance, cross, costs, noise))
        or (variance < 0).any()
        or (costs < 0).any()
        or (noise < 0).any()
        or not np.isfinite(weight)
        or weight < 0
        or not np.isfinite(risk_scale)
        or risk_scale <= 0
    ):
        raise ValueError(
            "matching finite Gaussian moments, legal masks and nonnegative costs required"
        )
    factor = NORMAL_ABSOLUTE_MEAN * risk_scale
    best = factor * np.sqrt(variance)
    masks = np.zeros_like(eligible)
    for size in range(1, len(costs) + 1):
        for subset in combinations(range(len(costs)), size):
            indices = np.asarray(subset)
            rows = np.flatnonzero(eligible[:, indices].all(axis=1))
            if not len(rows):
                continue
            c = covariance[rows][:, indices][:, :, indices].copy()
            c[:, np.arange(size), np.arange(size)] += noise[indices]
            eig, vectors = np.linalg.eigh((c + c.swapaxes(1, 2)) * 0.5)
            if (eig < -1e-8).any():
                raise ValueError("sensor covariance is not positive semidefinite")
            projected = np.einsum("nji,nj->ni", vectors, cross[rows][:, indices])
            reduction = np.sum(projected**2 / np.maximum(eig, 1e-12), axis=1)
            if np.any(reduction > variance[rows] + 1e-7 * np.maximum(1, variance[rows])):
                raise ValueError("joint moments imply negative target conditional variance")
            candidate = (
                factor * np.sqrt(np.maximum(variance[rows] - reduction, 0.0))
                + weight * costs[indices].sum()
            )
            improve = candidate < best[rows] - 1e-12
            winners = rows[improve]
            best[winners] = candidate[improve]
            masks[winners] = False
            masks[winners[:, None], indices[None, :]] = True
    return best, masks


class ArrivalSurvival:
    def __init__(self, delays):
        d = np.asarray(delays, dtype=np.float64)
        if d.ndim != 2 or min(d.shape) < 1 or np.isnan(d).any() or (d < 0).any():
            raise ValueError("nonnegative [training events, channels] delays required")
        self.delays = np.sort(d, axis=0)
        self.delays.flags.writeable = False

    @classmethod
    def fit(cls, times, arrivals, *, cutoff, grid_seconds, lookback):
        times, arrivals = np.asarray(times), np.asarray(arrivals, dtype=np.float64)
        if (
            times.ndim != 1
            or len(times) < 2
            or not np.isfinite(times).all()
            or arrivals.ndim != 2
            or arrivals.shape[0] != len(times)
            or np.isnan(arrivals).any()
            or (arrivals < times[:, None]).any()
            or (np.diff(times) <= 0).any()
            or not np.isfinite(cutoff)
            or not np.isfinite(grid_seconds)
            or grid_seconds <= 0
            or not isinstance(lookback, int)
            or lookback < 1
        ):
            raise ValueError("ordered events and valid train cutoff/grid required")
        mature = times <= cutoff - lookback * grid_seconds
        if not mature.any():
            raise ValueError("no sufficiently mature training cohort")
        known = arrivals[mature] <= cutoff
        return cls(np.where(known, arrivals[mature] - times[mature, None], np.inf))

    def probability(self, age, additional_seconds):
        age = np.asarray(age, dtype=np.float64)
        if (
            not np.isfinite(age).all()
            or (age < 0).any()
            or not np.isfinite(additional_seconds)
            or additional_seconds < 0
        ):
            raise ValueError("finite nonnegative ages and additional wait required")
        result = np.zeros(age.shape + (self.delays.shape[1],), dtype=float)
        for channel in range(self.delays.shape[1]):
            values = self.delays[:, channel]
            before = np.searchsorted(values, age, side="right")
            after = np.searchsorted(values, age + additional_seconds, side="right")
            remaining = len(values) - before
            result[..., channel] = np.divide(
                after - before, remaining, out=np.zeros(age.shape), where=remaining > 0
            )
        return result


class PosteriorPolicy:
    def __init__(
        self,
        forecaster,
        arrivals,
        waits,
        *,
        grid_seconds,
        weight=0.03,
        delay=0.02,
        risk_scale=1.0,
        samples=8,
        seed=20261002,
    ):
        self.forecaster, self.arrivals = forecaster, arrivals
        self.waits = validate_waits(waits)
        if (
            len(self.waits) < 2
            or self.waits[0] != 0
            or not np.isfinite(grid_seconds)
            or grid_seconds <= 0
            or self.waits[-1] >= forecaster.horizon * grid_seconds
            or arrivals.delays.shape[1] != forecaster.channels
            or any(not np.isfinite(v) or v < 0 for v in (weight, delay))
            or not np.isfinite(risk_scale)
            or risk_scale <= 0
            or not isinstance(samples, int)
            or not 1 <= samples <= 64
        ):
            raise ValueError("valid horizon, arrival model, cost weights and samples required")
        self.grid_seconds, self.weight, self.delay = grid_seconds, weight, delay
        self.risk_scale, self.samples, self.seed = risk_scale, samples, seed

    def choose(self, x, prediction, steps, acquired, costs):
        x = self.forecaster._check(x)
        steps, acquired, prediction = (
            np.asarray(steps),
            np.asarray(acquired),
            np.asarray(prediction),
        )
        n, length, channels = x.shape[:3]
        if (
            steps.shape != (n,)
            or not np.issubdtype(steps.dtype, np.integer)
            or (steps < 0).any()
            or (steps >= len(self.waits)).any()
            or acquired.shape != (n, channels)
            or acquired.dtype != bool
            or prediction.shape != (n,)
            or not np.isfinite(prediction).all()
        ):
            raise ValueError("valid decision steps, acquired mask and predictions required")
        observed = x[..., 1].astype(bool)
        current = self.forecaster.covariance(observed)
        best, subset = best_sensor_subset(
            *current,
            ~observed[:, -1] & ~acquired,
            costs,
            self.forecaster.observation_noise,
            weight=self.weight,
            risk_scale=self.risk_scale,
        )
        selected = np.where(subset.any(1), subset.argmax(1), -1).astype(int)
        uniforms = np.random.default_rng(self.seed).random((self.samples, length, channels))
        for step in np.unique(steps):
            rows = np.flatnonzero(steps == step)
            age = (length - 1 - np.arange(length)) * self.grid_seconds + self.waits[step]
            for later in range(int(step) + 1, len(self.waits)):
                additional = self.waits[later] - self.waits[step]
                probability = self.arrivals.probability(age, additional)
                future = observed[rows, None] | (uniforms[None] < probability[None, None])
                flat = future.reshape((-1, length, channels))
                moments = self.forecaster.covariance(flat)
                best_future, _ = best_sensor_subset(
                    *moments,
                    ~flat[:, -1],
                    costs,
                    self.forecaster.observation_noise,
                    weight=self.weight,
                    risk_scale=self.risk_scale,
                )
                estimate = best_future.reshape((len(rows), self.samples)).mean(1)
                estimate += self.delay * additional / self.waits[-1]
                improve = estimate < best[rows] - 1e-12
                winners = rows[improve]
                best[winners] = estimate[improve]
                selected[winners] = -2
        return selected
