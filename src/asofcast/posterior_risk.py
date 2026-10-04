"""Research-only linear Gaussian uncertainty, not certified real-process risk.

Restored from the preceding execution transcript. Inference receives only a
causal snapshot; unknown values, outcomes and future arrivals are never read.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge


@dataclass(frozen=True)
class PosteriorMoments:
    mean: np.ndarray
    variance: np.ndarray
    origin_covariance: np.ndarray
    cross: np.ndarray


def _covariance(value, size, name):
    value = np.asarray(value, dtype=np.float64)
    if (
        value.shape != (size, size)
        or not np.isfinite(value).all()
        or not np.allclose(value, value.T, rtol=0, atol=1e-10)
        or np.linalg.eigvalsh(value).min() < -1e-10
    ):
        raise ValueError(f"{name} must be a finite positive semidefinite covariance")
    return value.copy()


class GaussianForecast:
    """Centered VAR companion state, missing-observation filter, fixed horizon."""

    def __init__(
        self,
        transition,
        noise,
        center,
        prior,
        horizon,
        target,
        observation_noise,
        *,
        training_rows=0,
    ):
        self.center = np.asarray(center, dtype=np.float64).copy()
        self.transition = np.asarray(transition, dtype=np.float64).copy()
        if (
            self.center.ndim != 1
            or not len(self.center)
            or not np.isfinite(self.center).all()
            or self.transition.ndim != 2
            or not np.isfinite(self.transition).all()
            or self.transition.shape[0] != self.transition.shape[1]
            or self.transition.shape[0] < len(self.center)
            or self.transition.shape[0] % len(self.center)
        ):
            raise ValueError("finite companion transition and channel center required")
        self.channels, self.dim = len(self.center), len(self.transition)
        if (
            not isinstance(horizon, (int, np.integer))
            or isinstance(horizon, bool)
            or horizon < 1
            or not isinstance(target, (int, np.integer))
            or isinstance(target, bool)
            or not 0 <= target < self.channels
        ):
            raise ValueError("positive integer horizon and valid target required")
        self.horizon, self.target = int(horizon), int(target)
        self.noise = _covariance(noise, self.dim, "process noise")
        self.prior = _covariance(prior, self.dim, "prior")
        self.observation_noise = np.asarray(observation_noise, dtype=np.float64).copy()
        if (
            self.observation_noise.shape != (self.channels,)
            or not np.isfinite(self.observation_noise).all()
            or (self.observation_noise < 0).any()
        ):
            raise ValueError("nonnegative observation noise per channel required")
        self.training_rows = int(training_rows)
        self.projection = np.linalg.matrix_power(self.transition, self.horizon)[self.target]
        future = np.zeros_like(self.noise)
        for _ in range(self.horizon):
            future = self.transition @ future @ self.transition.T + self.noise
        self.future_noise = max(float(future[self.target, self.target]), 0.0)

    @classmethod
    def fit(
        cls, values, *, order=2, ridge=10.0, horizon=6, target=0, seed=42, known=None, block=42
    ):
        """Use only passed training values and transitions known by caller cutoff."""
        v = np.asarray(values, dtype=np.float64)
        if (
            v.ndim != 2
            or min(v.shape) < 1
            or not np.isfinite(v).all()
            or not isinstance(order, int)
            or isinstance(order, bool)
            or order < 1
            or not isinstance(block, int)
            or block < 1
            or not np.isfinite(ridge)
            or ridge <= 0
            or len(v) < max(32, order + 2)
        ):
            raise ValueError("finite training matrix, positive order/ridge/block required")
        observed = np.ones(v.shape, bool) if known is None else np.asarray(known)
        if observed.shape != v.shape or observed.dtype != bool or not observed.any(axis=0).all():
            raise ValueError("boolean cutoff-known mask with observed channels required")
        center = np.array([v[observed[:, c], c].mean() for c in range(v.shape[1])])
        z = v - center
        origins = np.arange(order, len(v))
        usable = np.all(observed[origins], axis=1)
        for lag in range(1, order + 1):
            usable &= np.all(observed[origins - lag], axis=1)
        origins = origins[usable]
        if len(origins) < max(24, order * v.shape[1] + 2):
            raise ValueError("too few complete training transitions known by cutoff")
        x = np.concatenate([z[origins - lag] for lag in range(1, order + 1)], axis=1)
        y = z[origins]
        groups = origins // block
        keys = np.unique(groups)
        chosen = np.random.default_rng(seed).choice(keys, size=len(keys), replace=True)
        rows = np.concatenate([np.flatnonzero(groups == key) for key in chosen])
        regression = Ridge(alpha=ridge, fit_intercept=False, solver="cholesky").fit(
            x[rows], y[rows]
        )
        channels, dim = v.shape[1], x.shape[1]
        transition = np.zeros((dim, dim))
        transition[:channels] = regression.coef_
        if order > 1:
            transition[channels:, :-channels] = np.eye(dim - channels)
        radius = float(np.max(np.abs(np.linalg.eigvals(transition))))
        gamma = min(1.0, 0.995 / max(radius, 1e-12))
        for lag in range(order):
            transition[:channels, lag * channels : (lag + 1) * channels] *= gamma ** (lag + 1)
        residual = y[rows] - x[rows] @ transition[:channels].T
        covariance = np.atleast_2d(np.cov(residual, rowvar=False, ddof=1))
        covariance = (
            0.95 * covariance + 0.05 * np.diag(np.diag(covariance)) + np.eye(channels) * 1e-9
        )
        noise = np.zeros((dim, dim))
        noise[:channels, :channels] = covariance
        prior = np.atleast_2d(np.cov(x[rows], rowvar=False, ddof=1)) + np.eye(dim) * 1e-9
        return cls(
            transition,
            noise,
            center,
            prior,
            horizon,
            target,
            np.diag(covariance) * 1e-8,
            training_rows=len(origins),
        )

    def _check(self, x):
        x = np.asarray(x, dtype=np.float64)
        if (
            x.ndim != 4
            or x.shape[2:] != (self.channels, 4)
            or min(x.shape[:2]) < 1
            or not np.isfinite(x).all()
            or not np.isin(x[..., 1], [0.0, 1.0]).all()
        ):
            raise ValueError("finite [N,L,C,4] snapshots with boolean observation mask required")
        return x

    def _filter(self, observed, values=None):
        observed = np.asarray(observed)
        if (
            observed.ndim != 3
            or observed.shape[-1] != self.channels
            or min(observed.shape[:2]) < 1
            or observed.dtype != bool
        ):
            raise ValueError("nonempty boolean [N,L,C] observation mask required")
        n, length, _ = observed.shape
        p = np.broadcast_to(self.prior, (n, self.dim, self.dim)).copy()
        m = None if values is None else np.zeros((n, self.dim), dtype=float)
        a = self.transition
        for step in range(length):
            if step:
                p = a[None] @ p @ a.T[None] + self.noise
                if m is not None:
                    m = m @ a.T
            for channel in range(self.channels):
                rows = np.flatnonzero(observed[:, step, channel])
                if not len(rows):
                    continue
                denominator = p[rows, channel, channel] + self.observation_noise[channel]
                usable = denominator > 1e-14
                rows, denominator = rows[usable], denominator[usable]
                column = p[rows, :, channel].copy()
                gain = column / denominator[:, None]
                if m is not None:
                    innovation = (
                        values[rows, step, channel] - self.center[channel] - m[rows, channel]
                    )
                    m[rows] += gain * innovation[:, None]
                p[rows] -= gain[:, :, None] * column[:, None, :]
            p = (p + p.swapaxes(1, 2)) * 0.5
        return m, p

    def covariance(self, observed):
        _, p = self._filter(observed)
        variance = np.einsum("d,ndk,k->n", self.projection, p, self.projection) + self.future_noise
        cross = np.einsum("d,ndc->nc", self.projection, p[:, :, : self.channels])
        return (
            np.maximum(variance, self.future_noise),
            p[:, : self.channels, : self.channels],
            cross,
        )

    def moments(self, x):
        x = self._check(x)
        m, p = self._filter(x[..., 1].astype(bool), x[..., 0])
        mean = m @ self.projection + self.center[self.target]
        variance = np.einsum("d,ndk,k->n", self.projection, p, self.projection) + self.future_noise
        cross = np.einsum("d,ndc->nc", self.projection, p[:, :, : self.channels])
        return PosteriorMoments(
            mean,
            np.maximum(variance, self.future_noise),
            p[:, : self.channels, : self.channels],
            cross,
        )

    def predict(self, x):
        return self.moments(x).mean

    def save(self, path):
        with Path(path).open("xb") as handle:
            np.savez_compressed(
                handle,
                transition=self.transition,
                noise=self.noise,
                center=self.center,
                prior=self.prior,
                horizon=self.horizon,
                target=self.target,
                observation_noise=self.observation_noise,
                training_rows=self.training_rows,
            )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(
                data["transition"],
                data["noise"],
                data["center"],
                data["prior"],
                int(data["horizon"]),
                int(data["target"]),
                data["observation_noise"],
                training_rows=int(data["training_rows"]),
            )
