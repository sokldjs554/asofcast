"""Research policies with block resampling, bounded backups, and a simple fallback.

Ensemble disagreement is a heuristic penalty, NOT a statistical confidence
interval. Only a separately held-out paired evaluation can establish improvement.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from asofcast.research_diagnosis import snapshot_batch
from asofcast.research_learning import decision_features


def continuation_targets(gains, next_net, next_eligible):
    """Gross immediate improvement plus the estimated net value of continuation."""
    gains, next_net, next_eligible = np.asarray(gains), np.asarray(next_net), np.asarray(next_eligible)
    if next_net.shape != next_eligible.shape or gains.shape != (len(next_net),):
        raise ValueError('matching successor rows required')
    return gains + np.maximum(0., np.where(next_eligible, next_net, -np.inf).max(axis=1))


def action_penalties(steps, waits, costs, weight=.03, delay=.02):
    waits, steps = np.asarray(waits), np.asarray(steps)
    result = np.broadcast_to(weight * np.asarray(costs), (len(steps), len(costs))).copy()
    next_steps = np.minimum(steps + 1, len(waits) - 1)
    return np.column_stack((result, delay * (waits[next_steps] - waits[steps]) / waits[-1]))


class GainEnsemble:
    def __init__(self, members):
        self.members = members

    @classmethod
    def fit(cls, features, targets, eligible, groups, settings, *, seed):
        groups = np.asarray(groups)
        _, inverse = np.unique(groups, return_inverse=True)
        blocks = inverse // settings['block_origins']
        block_rows = [np.flatnonzero(blocks == b) for b in np.unique(blocks)]
        rng = np.random.default_rng(seed)
        members = []
        for member in range(settings['members']):
            rows = np.concatenate([block_rows[b] for b in rng.integers(len(block_rows), size=len(block_rows))])
            models = []
            for action in range(targets.shape[1]):
                selected = rows[eligible[rows, action]]
                if len(selected) < 2:
                    models.append(None)
                    continue
                model = HistGradientBoostingRegressor(
                    loss='squared_error', max_iter=settings['iterations'],
                    max_leaf_nodes=settings['leaves'], min_samples_leaf=settings['min_leaf'],
                    l2_regularization=settings['ridge'], early_stopping=False,
                    random_state=seed + member)
                model.fit(features[selected], targets[selected, action])
                models.append(model)
            members.append(models)
        return cls(members)

    def net(self, features, penalties, *, uncertainty):
        estimates = np.stack([np.column_stack([
            np.zeros(len(features)) if model is None else model.predict(features)
            for model in models]) for models in self.members])
        return estimates.mean(0) - uncertainty * estimates.std(0) - penalties


def training_transitions(timeline, origins, cfg, target, forecaster, costs, *, seed,
                         weight=.03, delay=.02, metadata=True):
    """Training-only transitions. Deployed policies never receive this timeline.

    Each origin occurs with an empty and a sampled acquired subset at each wait.
    A successor retains all acquired channels, including through WAIT.
    """
    origins = np.asarray(origins)
    channels, n = len(costs), len(origins)
    rng = np.random.default_rng(seed)
    parts = []
    for step in range(len(cfg['waits_seconds'])):
        for augment in (False, True):
            steps = np.full(n, step, dtype=int)
            acquired = np.zeros((n, channels), dtype=bool)
            x = snapshot_batch(timeline, origins, cfg, steps, acquired)
            if augment:
                acquired = (rng.uniform(size=(n, channels)) < rng.uniform(size=(n, 1))) & (x[:, -1, :, 1] < .5)
                x = snapshot_batch(timeline, origins, cfg, steps, acquired)
            current = forecaster.predict(x)
            features = decision_features(x, current, steps, acquired, cfg['waits_seconds'], metadata=metadata)
            eligible = np.column_stack(((x[:, -1, :, 1] < .5) & ~acquired,
                                        steps < len(cfg['waits_seconds']) - 1))
            gains = np.zeros((n, channels + 1))
            successors = []
            y = timeline.values[origins + cfg['horizon'], target]
            for action in range(channels + 1):
                rows = np.flatnonzero(eligible[:, action])
                if not len(rows):
                    successors.append(None)
                    continue
                next_steps, next_acquired = steps[rows].copy(), acquired[rows].copy()
                if action == channels:
                    next_steps += 1
                else:
                    next_acquired[:, action] = True
                nx = snapshot_batch(timeline, origins[rows], cfg, next_steps, next_acquired)
                npred = forecaster.predict(nx)
                gains[rows, action] = np.abs(current[rows] - y[rows]) - np.abs(npred - y[rows])
                successors.append(dict(rows=rows, features=decision_features(
                    nx, npred, next_steps, next_acquired, cfg['waits_seconds'], metadata=metadata),
                    eligible=np.column_stack(((nx[:, -1, :, 1] < .5) & ~next_acquired,
                                               next_steps < len(cfg['waits_seconds']) - 1)),
                    penalties=action_penalties(next_steps, cfg['waits_seconds'], costs, weight, delay)))
            parts.append(dict(features=features, gains=gains, eligible=eligible,
                              groups=origins, successors=successors))
    return parts


def fit_backups(parts, waits, settings, *, seed):
    """Fitted finite-depth values; no hindsight maximization over true outcomes.

    Each backup uses predicted successor values, not realized best future actions.
    This does not establish optimality and may still overestimate continuation.
    """
    features = np.concatenate([p['features'] for p in parts])
    eligible = np.concatenate([p['eligible'] for p in parts])
    groups = np.concatenate([p['groups'] for p in parts])
    fitted, teacher = {}, None
    for depth in range(1, max(settings['depths']) + 1):
        targets = []
        for part in parts:
            gains = part['gains'].copy()
            if teacher is not None:
                for action, successor in enumerate(part['successors']):
                    if successor is not None:
                        rows = successor['rows']
                        net = teacher.net(successor['features'], successor['penalties'], uncertainty=1.)
                        gains[rows, action] = continuation_targets(
                            gains[rows, action], net, successor['eligible'])
            targets.append(gains)
        teacher = GainEnsemble.fit(features, np.concatenate(targets), eligible, groups,
                                   settings, seed=seed + depth * 100)
        if depth in settings['depths']:
            fitted[depth] = teacher
    return fitted


class InformationPolicy:
    def __init__(self, ensemble, waits, *, uncertainty=1., margin=0., metadata=True,
                 weight=.03, delay=.02):
        self.ensemble, self.waits = ensemble, waits
        self.uncertainty, self.margin, self.metadata = uncertainty, margin, metadata
        self.weight, self.delay = weight, delay

    def choose(self, x, prediction, steps, acquired, costs):
        features = decision_features(x, prediction, steps, acquired, self.waits, metadata=self.metadata)
        net = self.ensemble.net(features, action_penalties(steps, self.waits, costs, self.weight, self.delay),
                                uncertainty=self.uncertainty) - self.margin
        eligible = np.column_stack(((x[:, -1, :, 1] < .5) & ~acquired, steps < len(self.waits) - 1))
        net = np.where(eligible, net, -np.inf)
        chosen = net.argmax(1)
        return np.where(net[np.arange(len(x)), chosen] > 0,
                        np.where(chosen == len(costs), -2, chosen), -1)


def select_candidate(scores):
    """Validation selection always admits the same strong simple comparator."""
    simple = scores['simple']
    limit = np.mean(simple['native_error']) * 1.01
    valid = [name for name, score in scores.items()
             if np.isfinite(score['objective']).all() and np.isfinite(score['native_error']).all()
             and np.mean(score['native_error']) <= limit]
    return min(valid, key=lambda name: (np.mean(scores[name]['objective']), name != 'simple', name))
