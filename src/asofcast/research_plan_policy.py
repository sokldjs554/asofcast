"""Research-only conditional terminal-plan values; no outcome access at inference.

Offline labels include realized future arrivals and measurements. Inference
uses only a current causal snapshot, prediction, step and acquired mask.
Receding choices always execute one legal action through the existing rollout.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

from asofcast.research_diagnosis import snapshot_batch
from asofcast.research_learning import decision_features


def terminal_plan_examples(
    timeline,
    origins,
    cfg,
    target,
    forecaster,
    costs,
    *,
    seed,
    weight=0.03,
    delay=0.02,
    augment=True,
):
    """Expected net-gain targets for waiting then pulling at most two sensors.

    Caller must supply only policy-partition origins with mature target labels.
    Pulls already naturally received at the terminal time are free. Existing
    acquisitions persist, while only newly requested missing origin slots are paid.
    """
    origins = np.asarray(origins, dtype=int)
    costs = np.asarray(costs, dtype=float)
    channels, waits = len(costs), np.asarray(cfg["waits_seconds"], dtype=float)
    if (
        not len(origins)
        or costs.shape != (len(timeline.columns),)
        or not np.isfinite(costs).all()
        or (costs < 0).any()
        or waits[0] != 0
        or (np.diff(waits) <= 0).any()
    ):
        raise ValueError("nonempty origins, nonnegative costs and increasing waits required")
    subsets = [()] + [(c,) for c in range(channels)] + list(combinations(range(channels), 2))
    plans = [(step, subset) for step in range(len(waits)) for subset in subsets]
    rng = np.random.default_rng(seed)
    truth = timeline.values[origins + cfg["horizon"], target]
    features, labels, masks = [], [], []
    for step in range(len(waits)):
        empty = np.zeros((len(origins), channels), dtype=bool)
        start = snapshot_batch(timeline, origins, cfg, np.full(len(origins), step), empty)
        missing = start[:, -1, :, 1] < 0.5
        states = [empty]
        if augment:
            states.append((rng.random(empty.shape) < rng.random((len(origins), 1))) & missing)
        for acquired in states:
            current_x = (
                start
                if not acquired.any()
                else snapshot_batch(timeline, origins, cfg, np.full(len(origins), step), acquired)
            )
            current = forecaster.predict(current_x)
            eligible = np.zeros((len(origins), len(plans)), dtype=bool)
            gains = np.zeros_like(eligible, dtype=float)
            for terminal in range(step, len(waits)):
                terminal_x = snapshot_batch(
                    timeline, origins, cfg, np.full(len(origins), terminal), acquired
                )
                for j, (terminal_step, subset) in enumerate(plans):
                    if terminal_step != terminal or (terminal == step and not subset):
                        continue
                    rows = np.flatnonzero((current_x[:, -1, subset, 1] < 0.5).all(1))
                    if not len(rows):
                        continue
                    after_acquired = acquired[rows].copy()
                    after_acquired[:, subset] = True
                    after_x = (
                        terminal_x[rows]
                        if not subset
                        else snapshot_batch(
                            timeline,
                            origins[rows],
                            cfg,
                            np.full(len(rows), terminal),
                            after_acquired,
                        )
                    )
                    payment = (
                        (terminal_x[rows][:, -1, list(subset), 1] < 0.5) * costs[list(subset)]
                    ).sum(1)
                    gains[rows, j] = (
                        np.abs(current[rows] - truth[rows])
                        - np.abs(forecaster.predict(after_x) - truth[rows])
                        - weight * payment
                        - delay * (waits[terminal] - waits[step]) / waits[-1]
                    )
                    eligible[rows, j] = True
            features.append(
                decision_features(current_x, current, np.full(len(origins), step), acquired, waits)
            )
            labels.append(gains)
            masks.append(eligible)
    return np.concatenate(features), np.concatenate(labels), np.concatenate(masks), plans


class PlanRegressors:
    """One masked regression per plan; unavailable plans carry no observations."""

    def __init__(self, models):
        self.models = models

    def predict(self, features):
        return np.column_stack(
            [
                model.predict(features) if model is not None else np.full(len(features), -np.inf)
                for model in self.models
            ]
        )


class TerminalPlanPolicy:
    """Shared low-capacity regression of net benefit, with explicit COMMIT option."""

    def __init__(self, model, plans, waits, *, margin=0):
        self.model, self.plans = model, plans
        self.waits, self.margin = np.asarray(waits, dtype=float), margin

    @classmethod
    def fit(cls, features, gains, eligible, plans, waits, settings, *, seed):
        if gains.shape != eligible.shape or gains.shape[1] != len(plans):
            raise ValueError("aligned finite plan targets required")
        settings = dict(settings)
        algorithm = settings.pop("algorithm", "extra")
        if algorithm == "hist":
            models = []
            for j in range(len(plans)):
                rows = eligible[:, j]
                if not rows.any():
                    models.append(None)
                    continue
                regressor = HistGradientBoostingRegressor(**settings, random_state=seed)
                regressor.fit(features[rows], gains[rows, j])
                models.append(regressor)
            model = PlanRegressors(models)
        elif algorithm == "extra":
            model = ExtraTreesRegressor(**settings, random_state=seed, n_jobs=1)
            model.fit(features, np.where(eligible, gains, 0.0))
        else:
            raise ValueError("unknown plan regression algorithm")
        return cls(model, plans, waits)

    def choose(self, x, prediction, steps, acquired, costs):
        f = decision_features(x, prediction, steps, acquired, self.waits)
        gains = np.asarray(self.model.predict(f), dtype=float)
        missing = (x[:, -1, :, 1] < 0.5) & ~acquired
        for j, (terminal, subset) in enumerate(self.plans):
            legal = (terminal >= steps) & missing[:, subset].all(1)
            legal &= (terminal > steps) | bool(subset)
            gains[~legal, j] = -np.inf
        best = gains.argmax(1)
        result = np.full(len(x), -1, dtype=int)
        for row in np.flatnonzero(gains[np.arange(len(x)), best] > self.margin):
            terminal, subset = self.plans[best[row]]
            result[row] = -2 if terminal > steps[row] else subset[0]
        return result
