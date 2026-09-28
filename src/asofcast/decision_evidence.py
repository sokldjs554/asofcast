"""Offline, cost-aware decision comparisons and conditional uncertainty estimates."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _costs(costs, channels, weight):
    costs = np.asarray(costs, dtype=float)
    if (costs.shape != (channels,) or not np.isfinite(costs).all() or (costs < 0).any()
            or not np.isfinite(weight) or weight < 0):
        raise ValueError('finite nonnegative sensor costs and cost weight required')
    return costs


def choose_gain_actions(gains, eligible, costs, weight):
    gains, eligible = np.asarray(gains, dtype=float), np.asarray(eligible, dtype=bool)
    if gains.ndim != 2 or gains.shape != eligible.shape or gains.shape[1] < 1:
        raise ValueError('matching [origins, sensors] gain and eligibility required')
    costs = _costs(costs, gains.shape[1], weight)
    if not np.isfinite(gains[eligible]).all():
        raise ValueError('eligible gains must be finite')
    net = np.where(eligible, gains - weight * costs[None, :], -np.inf)
    best = net.argmax(axis=1)
    return np.where(net[np.arange(len(net)), best] > 0, best, -1).astype(np.int64)


def _outcome_inputs(y, current, counterfactual, actions, eligible, costs, weight, scale):
    y, current = np.asarray(y, dtype=float), np.asarray(current, dtype=float)
    counterfactual = np.asarray(counterfactual, dtype=float)
    actions, eligible = np.asarray(actions), np.asarray(eligible, dtype=bool)
    if (y.ndim != 1 or len(y) < 1 or current.shape != y.shape
            or counterfactual.ndim != 2 or counterfactual.shape[0] != len(y)
            or eligible.shape != counterfactual.shape or counterfactual.shape[1] < 1
            or actions.shape != y.shape or not np.issubdtype(actions.dtype, np.integer)
            or (actions < -1).any() or (actions >= counterfactual.shape[1]).any()
            or not np.isfinite(y).all() or not np.isfinite(current).all()
            or not np.isfinite(scale) or scale <= 0):
        raise ValueError('invalid outcome shapes, actions, or finite values')
    costs = _costs(costs, counterfactual.shape[1], weight)
    rows = np.flatnonzero(actions >= 0)
    if not eligible[rows, actions[rows]].all():
        raise ValueError('selected acquisition must be eligible')
    return y, current, counterfactual, actions, eligible, costs, rows


def action_losses(y, current, counterfactual, actions, eligible, costs, weight, scale):
    y, current, cf, actions, eligible, costs, rows = _outcome_inputs(
        y, current, counterfactual, actions, eligible, costs, weight, scale)
    selected = current.copy()
    selected[rows] = cf[rows, actions[rows]]
    if not np.isfinite(selected).all():
        raise ValueError('selected counterfactual must be finite')
    error = np.abs(selected - y)
    paid = np.zeros(len(y))
    paid[rows] = costs[actions[rows]]
    return {'native_error': error * scale, 'cost': paid,
            'objective': error + weight * paid, 'acquired': (actions >= 0).astype(float)}


def random_matched_losses(y, current, counterfactual, actions, eligible, costs, weight, scale):
    y, current, cf, actions, eligible, costs, rows = _outcome_inputs(
        y, current, counterfactual, actions, eligible, costs, weight, scale)
    result = action_losses(y, current, cf, np.full(len(y), -1), eligible, costs, weight, scale)
    if len(rows):
        allowed = eligible[rows]
        if not np.isfinite(cf[rows][allowed]).all():
            raise ValueError('all eligible random outcomes must be finite')
        count = allowed.sum(axis=1)
        error = np.where(allowed, np.abs(cf[rows] - y[rows, None]), 0.).sum(axis=1) / count
        paid = np.where(allowed, costs[None, :], 0.).sum(axis=1) / count
        result['native_error'][rows] = error * scale
        result['cost'][rows] = paid
        result['objective'][rows] = error + weight * paid
        result['acquired'][rows] = 1.
    return result


@dataclass(frozen=True)
class RidgeGainModel:
    """Independent regularized sensor regressions; no target is accepted at inference."""

    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray

    @classmethod
    def fit(cls, features, gains, eligible, *, ridge):
        x, y = np.asarray(features, dtype=float), np.asarray(gains, dtype=float)
        eligible = np.asarray(eligible, dtype=bool)
        if (x.ndim != 3 or min(x.shape) < 1 or y.shape != x.shape[:2]
                or eligible.shape != y.shape or not np.isfinite(x).all()
                or not np.isfinite(y[eligible]).all() or not np.isfinite(ridge) or ridge <= 0):
            raise ValueError('finite feature cube, eligible labels and positive ridge required')
        _, channels, features_count = x.shape
        mean = np.zeros((channels, features_count))
        scale = np.ones_like(mean)
        coefficients = np.zeros((channels, features_count + 1))
        for channel in range(channels):
            subset = x[eligible[:, channel], channel]
            if len(subset) == 0:
                continue
            mean[channel] = subset.mean(axis=0)
            scale[channel] = np.maximum(subset.std(axis=0), .01)
            design = np.column_stack([np.ones(len(subset)), (subset - mean[channel]) / scale[channel]])
            penalty = np.eye(features_count + 1) * ridge
            penalty[0, 0] = 0.
            coefficients[channel] = np.linalg.solve(
                design.T @ design + penalty, design.T @ y[eligible[:, channel], channel])
        return cls(mean, scale, coefficients)

    def predict(self, features):
        x = np.asarray(features, dtype=float)
        if x.ndim != 3 or x.shape[1:] != self.mean.shape or not np.isfinite(x).all():
            raise ValueError('finite feature cube matching fitted sensor schema required')
        normalized = (x - self.mean) / self.scale
        return np.einsum('ncf,cf->nc', normalized, self.coefficients[:, 1:]) + self.coefficients[:, 0]


def select_validation(losses):
    if not losses:
        raise ValueError('nonempty validation candidates required')
    shape = None
    means = {}
    for name, loss in losses.items():
        loss = np.asarray(loss, dtype=float)
        if loss.ndim != 1 or len(loss) < 1 or not np.isfinite(loss).all():
            raise ValueError('finite per-origin validation loss required')
        if shape is not None and loss.shape != shape:
            raise ValueError('all candidates must use the same validation cases')
        shape = loss.shape
        means[name] = float(loss.mean())
    return min(means, key=lambda name: (means[name], name))


def paired_block_interval(improvements, *, block, repeats, seed):
    delta = np.asarray(improvements, dtype=float)
    if (delta.ndim != 2 or min(delta.shape) < 1 or not np.isfinite(delta).all()
            or not isinstance(block, int) or not 1 <= block <= delta.shape[1]
            or not isinstance(repeats, int) or repeats < 2):
        raise ValueError('finite [training seed, chronological origin] improvements and valid block required')
    # Average the same origins across seeds BEFORE sampling blocks. Three seeds
    # do not turn one sequence of observed targets into three independent series.
    series = delta.mean(axis=0)
    n = len(series)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(repeats, (n + block - 1) // block))
    indices = ((starts[:, :, None] + np.arange(block)) % n).reshape(repeats, -1)[:, :n]
    sampled_means = series[indices].mean(axis=1)
    lower, upper = np.quantile(sampled_means, [.025, .975])
    return {'mean': float(series.mean()), 'lower95': float(lower), 'upper95': float(upper),
            'origins': n, 'training_seeds': len(delta), 'block_origins': block,
            'bootstrap_repeats': repeats, 'seed_improvements': delta.mean(axis=1).tolist(),
            'scope': 'conditional on these training seeds; circular blocks over shared target origins'}


def promotion_gate(cells, expected_keys, max_mae_regression, *, expected_seed_count=3):
    reasons = []
    if not expected_keys or len(set(expected_keys)) != len(expected_keys):
        raise ValueError('nonempty unique expected cells required')
    if not np.isfinite(max_mae_regression) or max_mae_regression < 0:
        raise ValueError('finite nonnegative MAE regression limit required')
    if not isinstance(expected_seed_count, int) or expected_seed_count < 3:
        raise ValueError('at least three distinct training seeds required')
    for key in expected_keys:
        if key not in cells:
            reasons.append(f'{key}: missing result')
            continue
        cell = cells[key]
        for name in ('mlp', 'simple'):
            comparison = cell.get('comparisons', {}).get(name, {})
            lower = comparison.get('lower95', float('nan'))
            seeds = np.asarray(comparison.get('seed_improvements', []), dtype=float)
            if not np.isfinite(lower) or lower <= 0:
                reasons.append(f'{key} vs {name}: interval does not establish improvement')
            if (seeds.shape != (expected_seed_count,) or not np.isfinite(seeds).all()
                    or (seeds <= 0).any()):
                reasons.append(f'{key} vs {name}: not every training seed improves')
        regressions = np.asarray(cell.get('mae_relative_regressions', []), dtype=float)
        if (regressions.shape != (expected_seed_count,) or not np.isfinite(regressions).all()
                or (regressions > max_mae_regression).any()):
            reasons.append(f'{key}: MAE safety requirement failed')
    return {'passed': not reasons, 'reasons': reasons,
            'meaning': 'candidate promotion gate within this fixed benchmark; not universal superiority'}
