"""Low-capacity research candidates; train/policy/validation supplied explicitly."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from asofcast.acquisition import acquisition_features
from asofcast.policy import policy_features
from asofcast.research_diagnosis import multiscale_features, snapshot_batch
from asofcast.training import predict


class TorchForecast:
    def __init__(self, model):
        self.model = model

    def predict(self, x):
        return predict(self.model, x).astype(float)


class ResidualForecast:
    def __init__(self, base, tree, weight, metadata):
        self.base, self.tree, self.weight, self.metadata = base, tree, weight, metadata

    def predict(self, x):
        p = self.base.predict(x)
        if self.weight == 0:
            return p
        features = np.column_stack((multiscale_features(x, metadata=self.metadata), p))
        return p + self.weight * self.tree.predict(features)


def fit_residual_forecast(base, train_x, train_y, val_x, val_y, settings, *, seed, metadata=True):
    before, val_before = base.predict(train_x), base.predict(val_x)
    features = np.column_stack((multiscale_features(train_x, metadata=metadata), before))
    val_features = np.column_stack((multiscale_features(val_x, metadata=metadata), val_before))
    records, fitted = [], {}
    for leaves in settings['leaf_candidates']:
        tree = HistGradientBoostingRegressor(
            loss=settings['loss'], max_iter=settings['iterations'], max_leaf_nodes=leaves,
            min_samples_leaf=settings['min_samples_leaf'], l2_regularization=settings['l2_regularization'],
            early_stopping=False, random_state=seed)
        tree.fit(features, np.asarray(train_y) - before)
        residual = tree.predict(val_features)
        fitted[leaves] = tree
        for weight in settings['weights']:
            records.append({'leaves': leaves, 'weight': weight,
                            'validation_mae': float(np.abs(val_before + weight * residual - val_y).mean())})
    best = min(records, key=lambda r: (r['validation_mae'], r['weight'], r['leaves']))
    return (ResidualForecast(base, fitted[best['leaves']], best['weight'], metadata),
            {**best, 'split': 'validation', 'metadata': metadata, 'candidates': records})


def decision_features(x, p, steps, acquired, waits, *, metadata=True):
    return np.column_stack((multiscale_features(x, metadata=metadata), p,
                            1 - np.asarray(waits)[steps] / waits[-1], acquired.astype(float)))


def transition_examples(timeline, origins, cfg, target, forecaster, *, seed, metadata=True):
    """Offline labels for one-step gains, including post-acquisition states.

    The caller supplies policy-partition origins whose labels have arrived by
    its cutoff. Future outcomes are used ONLY here, never by TreePolicy.choose.
    """
    origins = np.asarray(origins)
    n, channels = len(origins), len(timeline.columns)
    rng = np.random.default_rng(seed)
    xs, ys, masks = [], [], []
    for step in range(len(cfg['waits_seconds'])):
        for augmented in (False, True):
            steps = np.full(n, step, dtype=int)
            acquired = np.zeros((n, channels), dtype=bool)
            empty = snapshot_batch(timeline, origins, cfg, steps, acquired)
            if augmented:
                probability = rng.uniform(size=(n, 1))
                acquired = (rng.uniform(size=(n, channels)) < probability) & (empty[:, -1, :, 1] < .5)
            x = snapshot_batch(timeline, origins, cfg, steps, acquired) if augmented else empty
            current = forecaster.predict(x)
            target_y = timeline.values[origins + cfg['horizon'], target]
            gains = np.zeros((n, channels + 1))
            eligible = np.column_stack((x[:, -1, :, 1] < .5,
                                         np.full(n, step < len(cfg['waits_seconds']) - 1)))
            for channel in range(channels + 1):
                rows = np.flatnonzero(eligible[:, channel])
                if not len(rows):
                    continue
                after_acquired, after_steps = acquired[rows].copy(), steps[rows].copy()
                if channel == channels:
                    after_steps += 1
                else:
                    after_acquired[:, channel] = True
                cf = snapshot_batch(timeline, origins[rows], cfg, after_steps, after_acquired)
                after = forecaster.predict(cf)
                gains[rows, channel] = np.abs(current[rows] - target_y[rows]) - np.abs(after - target_y[rows])
            xs.append(decision_features(x, current, steps, acquired, cfg['waits_seconds'], metadata=metadata))
            ys.append(gains)
            masks.append(eligible)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(masks)


class TreePolicy:
    def __init__(self, models, waits, *, weight=.03, delay=.02, margin=0., metadata=True):
        self.models, self.waits = models, np.asarray(waits, dtype=float)
        self.weight, self.delay, self.margin, self.metadata = weight, delay, margin, metadata

    @classmethod
    def fit(cls, features, gains, eligible, waits, settings, *, seed, metadata=True):
        models = []
        for channel in range(gains.shape[1]):
            rows = eligible[:, channel]
            if not rows.any():
                models.append(None)
                continue
            model = HistGradientBoostingRegressor(
                loss=settings['loss'], max_iter=settings['iterations'],
                max_leaf_nodes=settings['max_leaf_nodes'], min_samples_leaf=settings['min_samples_leaf'],
                l2_regularization=settings['l2_regularization'], early_stopping=False, random_state=seed)
            model.fit(features[rows], gains[rows, channel])
            models.append(model)
        return cls(models, waits, metadata=metadata)

    def choose(self, x, prediction, steps, acquired, costs):
        features = decision_features(x, prediction, steps, acquired, self.waits, metadata=self.metadata)
        n, channels = len(x), len(costs)
        net = np.full((n, channels + 1), -np.inf)
        eligible = np.column_stack(((x[:, -1, :, 1] < .5) & ~acquired, steps < len(self.waits) - 1))
        for channel, model in enumerate(self.models):
            rows = np.flatnonzero(eligible[:, channel])
            if model is None or not len(rows):
                continue
            penalty = (self.weight * costs[channel] if channel < channels else
                       self.delay * (self.waits[steps[rows] + 1] - self.waits[steps[rows]]) / self.waits[-1])
            net[rows, channel] = model.predict(features[rows]) - penalty - self.margin
        best = net.argmax(1)
        return np.where(net[np.arange(n), best] > 0, np.where(best == channels, -2, best), -1)


class LegacyJoint:
    def __init__(self, bundle):
        self.bundle = bundle

    def choose(self, x, prediction, steps, acquired, costs):
        bundle = self.bundle
        waits = np.asarray(bundle.config['waits_seconds'])
        channels = len(costs)
        net = np.full((len(x), channels + 1), -np.inf)
        for step in np.unique(steps):
            rows = np.flatnonzero(steps == step)
            remaining = 1 - waits[step] / waits[-1]
            for channel in range(channels):
                features = acquisition_features(x[rows], prediction[rows],
                                                 np.full(len(rows), channel), costs, remaining)
                net[rows, channel] = predict(bundle.acquisition, features) - bundle.config['acquisition_cost_weight'] * costs[channel]
            if step < len(waits) - 1:
                features = policy_features(x[rows], prediction[rows], remaining)
                net[rows, channels] = predict(bundle.policy, features) - bundle.report['policy_selection']['threshold'] * (waits[step + 1] - waits[step]) / waits[-1]
        net[:, :channels] = np.where((x[:, -1, :, 1] < .5) & ~acquired, net[:, :channels], -np.inf)
        best = net.argmax(1)
        return np.where(net[np.arange(len(x)), best] > 0, np.where(best == channels, -2, best), -1)


class SimplePolicy:
    def __init__(self, wait_step=0, kind='commit', target=0, seed=42):
        self.wait_step, self.kind, self.target = wait_step, kind, target
        self.rng = np.random.default_rng(seed)

    def choose(self, x, prediction, steps, acquired, costs):
        result = np.where(steps < self.wait_step, -2, -1)
        eligible = (x[:, -1, :, 1] < .5) & ~acquired
        for row in np.flatnonzero((result == -1) & ~acquired.any(1) & eligible.any(1)):
            if self.kind == 'target' and eligible[row, self.target]:
                result[row] = self.target
            elif self.kind == 'oldest':
                result[row] = np.argmax(np.where(eligible[row], x[row, -1, :, 2], -np.inf))
            elif self.kind == 'random':
                result[row] = self.rng.choice(np.flatnonzero(eligible[row]))
        return result.astype(int)
