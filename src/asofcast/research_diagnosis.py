"""Research-only headroom and causal sequential replay; outcomes never enter policy calls."""
from __future__ import annotations

import numpy as np

from asofcast.decision_evidence import action_losses, choose_gain_actions
from asofcast.policy import validate_waits


def oracle_acquisition(y, current, counterfactual, eligible, costs, weight):
    """Retrospective best single pull or commit, NOT an implementable policy."""
    y, current, cf = map(lambda v: np.asarray(v, dtype=float), (y, current, counterfactual))
    gains = np.abs(current - y)[:, None] - np.abs(cf - y[:, None])
    choice = choose_gain_actions(gains, eligible, costs, weight)
    return {**action_losses(y, current, cf, choice, eligible, costs, weight, 1.),
            'choice': choice}


def multiscale_features(x, *, metadata=True):
    """Only values and metadata in the supplied causal snapshot; no time-series lookup."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 4 or x.shape[-1] != 4 or x.shape[1] < 4 or not np.isfinite(x).all():
        raise ValueError('finite [N,L>=4,C,4] snapshots required')
    values = x[..., 0]
    latest = values[:, -1]
    features = [latest]
    for window in (1, 4, 12, 24):
        lag = min(window, x.shape[1] - 1)
        recent = values[:, -min(window + 1, x.shape[1]):]
        features.extend((latest - values[:, -lag - 1], recent.mean(1), recent.std(1)))
    if metadata:
        features.extend((x[:, -1, :, 1:].reshape(len(x), -1),
                         x[:, -min(12, x.shape[1]):, :, 1].mean(1)))
    return np.concatenate(features, axis=1).astype(np.float32)


def snapshot_batch(timeline, origins, cfg, steps, acquired):
    return np.stack([timeline.snapshot(
        int(origin), cfg['waits_seconds'][int(step)], cfg['lookback'], cfg['horizon'],
        acquired_channels=np.flatnonzero(mask).tolist()).features()
        for origin, step, mask in zip(origins, steps, acquired, strict=True)])


def rollout(timeline, origins, cfg, forecaster, policy, costs):
    """WAIT=-2, COMMIT=-1, ACQUIRE=channel. Each policy sees only current state.

    The immutable timeline contains future records, but is never passed to a
    policy. Final labels are deliberately absent from the return value too.
    """
    origins = np.asarray(origins)
    waits = validate_waits(cfg['waits_seconds'])
    costs = np.asarray(costs, dtype=float)
    n, channels = len(origins), len(timeline.columns)
    if (n == 0 or origins.ndim != 1 or not np.issubdtype(origins.dtype, np.integer)
            or costs.shape != (channels,) or not np.isfinite(costs).all() or (costs < 0).any()
            or waits[-1] >= cfg['horizon'] * timeline.grid_seconds):
        raise ValueError('valid origins, sensor costs and deadline required')
    acquired = np.zeros((n, channels), dtype=bool)
    steps = np.zeros(n, dtype=int)
    active = np.ones(n, dtype=bool)
    prediction = np.zeros(n)
    actions = np.full((n, channels + len(waits)), -9, dtype=np.int16)
    paid = np.zeros(n)
    for turn in range(actions.shape[1]):
        rows = np.flatnonzero(active)
        if not len(rows):
            break
        x = snapshot_batch(timeline, origins[rows], cfg, steps[rows], acquired[rows])
        p = np.asarray(forecaster.predict(x), dtype=float)
        if p.shape != (len(rows),) or not np.isfinite(p).all():
            raise ValueError('one finite prediction per active origin required')
        chosen = np.asarray(policy.choose(x, p, steps[rows].copy(), acquired[rows].copy(), costs.copy()))
        if (chosen.shape != (len(rows),) or not np.issubdtype(chosen.dtype, np.integer)
                or (chosen < -2).any() or (chosen >= channels).any()):
            raise ValueError('invalid sequential action')
        # Match the service: at the deadline no further WAIT, but a zero-latency
        # simulated pull is allowed. The final decision never exceeds deadline.
        waiting = chosen == -2
        if (steps[rows[waiting]] >= len(waits) - 1).any():
            raise ValueError('WAIT beyond deadline')
        pulls = np.flatnonzero(chosen >= 0)
        if len(pulls):
            channel = chosen[pulls]
            if ((x[pulls, -1, channel, 1] >= .5) | acquired[rows[pulls], channel]).any():
                raise ValueError('acquisition must be eligible and not previously paid')
            acquired[rows[pulls], channel] = True
            paid[rows[pulls]] += costs[channel]
        steps[rows[waiting]] += 1
        stopped = chosen == -1
        prediction[rows[stopped]] = p[stopped]
        active[rows[stopped]] = False
        actions[rows, turn] = chosen
    if active.any():
        raise ValueError('policy did not commit within finite legal action budget')
    return {'prediction': prediction, 'cost': paid, 'wait_seconds': waits[steps],
            'actions': actions, 'acquired_count': acquired.sum(1),
            'target_times': timeline.times[origins] + cfg['horizon'] * timeline.grid_seconds}


def score_rollout(result, y, scale, *, weight=.03, delay=.02, deadline=3600):
    y = np.asarray(y, dtype=float)
    if (y.shape != result['prediction'].shape or not np.isfinite(y).all()
            or not np.isfinite(scale) or scale <= 0 or deadline <= 0):
        raise ValueError('matching finite outcomes and positive scale required')
    error = np.abs(result['prediction'] - y)
    return {'objective': error + weight * result['cost'] + delay * result['wait_seconds'] / deadline,
            'native_error': error * scale, 'cost': result['cost'],
            'wait_seconds': result['wait_seconds'], 'acquired_count': result['acquired_count'],
            'deadline_violation': (result['wait_seconds'] > deadline).astype(float)}


def confirmatory_gate(cells, expected, *, min_relative=.01, max_regression=.01):
    if not expected or len(set(expected)) != len(expected):
        raise ValueError('distinct expected cells required')
    reasons = []
    for key in expected:
        if key not in cells:
            reasons.append(f'{key}: missing result')
            continue
        for comparator in ('legacy_joint', 'validation_simple'):
            record = cells[key].get('comparisons', {}).get(comparator, {})
            lower = record.get('lower95', np.nan)
            relative = record.get('relative_improvement', np.nan)
            seed_values = np.asarray(record.get('seed_improvements', []), dtype=float)
            if not np.isfinite(lower) or lower <= 0:
                reasons.append(f'{key}/{comparator}: confidence interval does not establish improvement')
            if not np.isfinite(relative) or relative < min_relative:
                reasons.append(f'{key}/{comparator}: practical improvement below {min_relative:.1%}')
            if seed_values.shape != (3,) or not np.isfinite(seed_values).all() or (seed_values <= 0).any():
                reasons.append(f'{key}/{comparator}: not every seed improves')
        regressions = np.asarray(cells[key].get('mae_regressions', []), dtype=float)
        if regressions.shape != (3,) or not np.isfinite(regressions).all() or (regressions > max_regression).any():
            reasons.append(f'{key}: MAE regression exceeds limit')
    return {'passed': not reasons, 'reasons': reasons,
            'scope': 'all predeclared new-dataset conditions; not universal superiority'}
