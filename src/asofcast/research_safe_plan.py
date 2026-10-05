"""Research-only, honest calibration of executable complete action plans.

A router receives an initial causal snapshot only. It commits to one complete
strategy; evaluation outcomes enter only policy fitting and held-out calibration.
The uncertainty penalty is a heuristic safeguard, not a coverage guarantee.
"""

from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from asofcast.policy import validate_waits
from asofcast.research_diagnosis import snapshot_batch
from asofcast.research_learning import decision_features


def initial_features(timeline, origins, cfg, forecaster):
    origins = np.asarray(origins, dtype=int)
    acquired = np.zeros((len(origins), len(timeline.columns)), dtype=bool)
    steps = np.zeros(len(origins), dtype=int)
    x = snapshot_batch(timeline, origins, cfg, steps, acquired)
    return decision_features(x, forecaster.predict(x), steps, acquired, cfg["waits_seconds"])


def execute_plans(timeline, origins, cfg, forecaster, costs, plans):
    """Execute each preselected terminal time and requested subset exactly once.

    Plan selection must precede this function and must not use timeline outcomes.
    Pulls reveal frozen-origin slots only; sensors received naturally are free.
    """
    origins = np.asarray(origins)
    waits = validate_waits(cfg["waits_seconds"])
    costs = np.asarray(costs, dtype=float)
    n, channels = (len(origins), len(timeline.columns))
    if (
        n == 0
        or origins.ndim != 1
        or (not np.issubdtype(origins.dtype, np.integer))
        or (costs.shape != (channels,))
        or (not np.isfinite(costs).all())
        or (costs < 0).any()
        or (len(plans) != n)
        or (waits[-1] >= cfg["horizon"] * timeline.grid_seconds)
    ):
        raise ValueError("valid origins, costs and aligned plans required")
    steps = np.empty(n, dtype=int)
    requested = np.zeros((n, channels), dtype=bool)
    for row, (terminal, subset) in enumerate(plans):
        if (
            isinstance(terminal, bool)
            or not isinstance(terminal, (int, np.integer))
            or terminal < 0
            or (terminal >= len(waits))
            or (len(set(subset)) != len(subset))
            or any(
                isinstance(c, bool)
                or not isinstance(c, (int, np.integer))
                or c < 0
                or (c >= channels)
                for c in subset
            )
        ):
            raise ValueError("illegal terminal plan")
        steps[row] = terminal
        requested[row, list(subset)] = True
    passive = snapshot_batch(timeline, origins, cfg, steps, np.zeros_like(requested))
    acquired = requested & (passive[:, -1, :, 1] < 0.5)
    after = (
        passive if not acquired.any() else snapshot_batch(timeline, origins, cfg, steps, acquired)
    )
    prediction = np.asarray(forecaster.predict(after), dtype=float)
    if prediction.shape != (n,) or not np.isfinite(prediction).all():
        raise ValueError("finite plan predictions required")
    actions = np.full((n, channels + len(waits)), -9, dtype=np.int16)
    for row in range(n):
        events = [-2] * int(steps[row]) + np.flatnonzero(acquired[row]).tolist() + [-1]
        actions[row, : len(events)] = events
    return {
        "prediction": prediction,
        "cost": (acquired * costs).sum(1),
        "wait_seconds": waits[steps],
        "actions": actions,
        "acquired_count": acquired.sum(1),
        "target_times": timeline.times[origins] + cfg["horizon"] * timeline.grid_seconds,
    }


def mature_policy_origins(timeline, origins, cfg, target_channel, cutoff):
    """Labels must have arrived by the fitting cutoff in this scenario."""
    origins = np.asarray(origins)
    return origins[timeline.arrivals[origins + cfg["horizon"], target_channel] <= cutoff]


class ValueTree:
    """Greedy complete-plan reward splits, not a globally optimal policy tree."""

    def __init__(self, max_depth, min_samples_leaf):
        self.max_depth, self.min_samples_leaf = (max_depth, min_samples_leaf)
        self.nodes = []

    def fit(self, features, gains):
        self.feature_count = features.shape[1]

        def build(rows, depth):
            node = len(self.nodes)
            self.nodes.append(None)
            parent = gains[rows].sum(0)
            best_value = float(parent.max())
            best = None
            if depth < self.max_depth and len(rows) >= 2 * self.min_samples_leaf:
                positions = np.unique(
                    np.linspace(
                        self.min_samples_leaf, len(rows) - self.min_samples_leaf, 17, dtype=int
                    )
                )
                for feature in range(self.feature_count):
                    order = rows[np.argsort(features[rows, feature], kind="stable")]
                    values = features[order, feature]
                    cumulative = np.cumsum(gains[order], axis=0)
                    for pos in positions:
                        if values[pos - 1] >= values[pos]:
                            continue
                        left = cumulative[pos - 1]
                        value = float(left.max() + (parent - left).max())
                        if value > best_value + 1e-10 * len(rows):
                            threshold = float(values[pos - 1])
                            best_value, best = (
                                value,
                                (feature, threshold, order[:pos], order[pos:]),
                            )
            if best is None:
                self.nodes[node] = ("leaf",)
            else:
                feature, threshold, left, right = best
                self.nodes[node] = (
                    feature,
                    threshold,
                    build(left, depth + 1),
                    build(right, depth + 1),
                )
            return node

        build(np.arange(len(features)), 0)
        return self

    def apply(self, features):
        features = np.asarray(features)
        if (
            features.ndim != 2
            or features.shape[1] != self.feature_count
            or (not np.isfinite(features).all())
        ):
            raise ValueError("matching finite tree features required")
        result = np.empty(len(features), dtype=int)

        def visit(node, rows):
            definition = self.nodes[node]
            if definition[0] == "leaf":
                result[rows] = node
                return
            feature, threshold, left, right = definition
            mask = features[rows, feature] <= threshold
            visit(left, rows[mask])
            visit(right, rows[~mask])

        visit(0, np.arange(len(features)))
        return result


class HonestPlanRouter:
    """Choose a plan on earlier origins and calibrate it on later distinct ones."""

    def __init__(self, tree, leaf_records, diagnostics):
        self.tree, self.leaf_records, self.diagnostics = (tree, leaf_records, diagnostics)

    @classmethod
    def fit(cls, features, gains, origins, settings, *, seed, mae_gains=None):
        f, g, origins = (np.asarray(features), np.asarray(gains), np.asarray(origins))
        if (
            f.ndim != 2
            or g.ndim != 2
            or len(f) != len(g)
            or (len(f) != len(origins))
            or (g.shape[1] < 2)
            or (not np.isfinite(f).all())
            or (not np.isfinite(g).all())
            or (not np.array_equal(g[:, 0], np.zeros(len(g))))
        ):
            raise ValueError("finite aligned gains with zero default advantage required")
        priority = settings.get("mae_priority", 0.0)
        if not np.isfinite(priority) or priority < 0:
            raise ValueError("nonnegative MAE priority required")
        reward = g
        if priority:
            mae = np.asarray(mae_gains)
            if (
                mae.shape != g.shape
                or not np.isfinite(mae).all()
                or (not np.equal(mae[:, 0], 0).all())
            ):
                raise ValueError("aligned finite MAE advantage required")
            reward = g + priority * mae
        unique = np.unique(origins)
        cut = int(len(unique) * settings.get("fit_fraction", 0.6))
        if cut < 2 or cut >= len(unique) - 1:
            raise ValueError("distinct chronological fitting and calibration origins required")
        cutoff = unique[cut]
        train = origins < cutoff - settings.get("purge_observations", 54)
        calibration = origins >= cutoff
        if len(np.unique(origins[train])) < 2:
            raise ValueError("purge leaves insufficient fitting origins")
        if settings.get("algorithm", "regression_tree") == "value_tree":
            tree = ValueTree(settings["max_depth"], settings["min_samples_leaf"])
        else:
            tree = DecisionTreeRegressor(
                max_depth=settings["max_depth"],
                min_samples_leaf=settings["min_samples_leaf"],
                random_state=seed,
            )
        tree.fit(f[train], reward[train])
        fit_leaves, cal_leaves = (tree.apply(f[train]), tree.apply(f[calibration]))
        records = {}
        cal_origins = origins[calibration]
        for leaf in np.unique(fit_leaves):
            chosen = int(reward[train][fit_leaves == leaf].mean(0).argmax())
            rows = cal_leaves == leaf
            ids = np.unique(cal_origins[rows])
            per_origin = np.array(
                [g[calibration][rows & (cal_origins == o), chosen].mean() for o in ids]
            )
            blocks = [v for v in np.array_split(per_origin, max(1, len(ids) // 6)) if len(v)]
            means = np.array([v.mean() for v in blocks])
            mean = float(per_origin.mean()) if len(ids) else 0.0
            se = float(means.std(ddof=1) / np.sqrt(len(means))) if len(means) >= 2 else float("inf")
            supported = len(ids) >= settings.get("minimum_calibration_origins", 12)
            records[int(leaf)] = {
                "chosen": chosen,
                "mean": mean,
                "standard_error": se,
                "origins": len(ids),
                "supported": supported,
            }
            if not np.isfinite(se):
                records[int(leaf)]["standard_error"] = None
        return cls(
            tree,
            records,
            {
                "fitting_origin_max": int(origins[train].max()),
                "calibration_origin_min": int(origins[calibration].min()),
                "fitting_origins": len(np.unique(origins[train])),
                "calibration_origins": len(np.unique(cal_origins)),
                "leaves": {str(k): v for k, v in records.items()},
                "scope": "origin-block heuristic, not confidence coverage",
            },
        )

    def select(self, features, *, strength=0.0, margin=0.0):
        if not np.isfinite(strength) or strength < 0 or (not np.isfinite(margin)) or (margin < 0):
            raise ValueError("nonnegative finite calibration strength and margin required")
        leaves = self.tree.apply(features)
        choices = np.zeros(len(features), dtype=int)
        for leaf in np.unique(leaves):
            r = self.leaf_records[int(leaf)]
            penalty = (
                0.0
                if strength == 0
                else strength * r["standard_error"]
                if r["standard_error"] is not None
                else float("inf")
            )
            if r["supported"] and r["mean"] - penalty > margin:
                choices[leaves == leaf] = r["chosen"]
        return choices

    def raw_select(self, features):
        """Uncalibrated fitting-period choice, retained only as a diagnostic ablation."""
        leaves = self.tree.apply(features)
        return np.array([self.leaf_records[int(leaf)]["chosen"] for leaf in leaves], dtype=int)


def strategy_bank(channels, wait_count=3):
    from itertools import combinations

    subsets = [()] + [(c,) for c in range(channels)] + list(combinations(range(channels), 2))
    return (
        [("default", 0, ())]
        + [("plan", step, subset) for step in range(wait_count) for subset in subsets]
        + [("oldest", step, ()) for step in range(wait_count)]
    )


def execute_strategies(timeline, origins, cfg, forecaster, costs, bank, choices, default_policy):
    """Route already selected indices; future outcomes never select a strategy."""
    from asofcast.research_diagnosis import rollout
    from asofcast.research_learning import SimplePolicy

    origins, choices = (np.asarray(origins), np.asarray(choices))
    if (
        choices.shape != origins.shape
        or not np.issubdtype(choices.dtype, np.integer)
        or (choices < 0).any()
        or (choices >= len(bank)).any()
    ):
        raise ValueError("aligned legal strategy indices required")
    output = None
    for choice in np.unique(choices):
        rows = np.flatnonzero(choices == choice)
        kind, step, subset = bank[int(choice)]
        if kind == "default":
            result = rollout(timeline, origins[rows], cfg, forecaster, default_policy, costs)
        elif kind == "oldest":
            result = rollout(
                timeline, origins[rows], cfg, forecaster, SimplePolicy(step, "oldest"), costs
            )
        elif kind == "plan":
            result = execute_plans(
                timeline, origins[rows], cfg, forecaster, costs, [(step, subset)] * len(rows)
            )
        else:
            raise ValueError("unknown strategy kind")
        if output is None:
            output = {
                k: np.empty((len(origins),) + v.shape[1:], dtype=v.dtype) for k, v in result.items()
            }
        for k, v in result.items():
            output[k][rows] = v
    if output is None:
        raise ValueError("nonempty strategy rollout required")
    return output


class GainEnsemble:
    def __init__(self, models):
        self.models = models

    def predict(self, features):
        return np.column_stack(
            [np.zeros(len(features))] + [m.predict(features) for m in self.models]
        )


class HonestGainRouter:
    """Flexible gain ranking with held-out calibration of the selected winner."""

    def __init__(self, model, edges, records, diagnostics):
        self.model, self.edges, self.records, self.diagnostics = (
            model,
            edges,
            records,
            diagnostics,
        )

    @classmethod
    def fit(cls, features, gains, origins, settings, *, seed, mae_gains=None):
        from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

        f, g, origins = (np.asarray(features), np.asarray(gains), np.asarray(origins))
        if (
            f.ndim != 2
            or g.ndim != 2
            or len(f) != len(g)
            or (len(f) != len(origins))
            or (g.shape[1] < 2)
            or (not np.isfinite(f).all())
            or (not np.isfinite(g).all())
            or (not np.equal(g[:, 0], 0).all())
        ):
            raise ValueError("finite aligned zero-default gain arrays required")
        unique = np.unique(origins)
        cut = int(len(unique) * settings.get("fit_fraction", 0.6))
        if cut < 2 or cut >= len(unique) - 1:
            raise ValueError("distinct chronological origins required")
        cutoff = unique[cut]
        train = origins < cutoff - settings.get("purge_observations", 54)
        calibration = origins >= cutoff
        if len(np.unique(origins[train])) < 2:
            raise ValueError("insufficient fitting origins after purge")
        priority = settings.get("mae_priority", 0.0)
        if not np.isfinite(priority) or priority < 0:
            raise ValueError("nonnegative finite MAE priority required")
        reward = g
        if priority:
            m = np.asarray(mae_gains)
            if m.shape != g.shape or not np.isfinite(m).all() or (not np.equal(m[:, 0], 0).all()):
                raise ValueError("aligned MAE gains required")
            reward = g + priority * m
        if settings.get("algorithm", "extra") == "hist":
            models = []
            for j in range(1, g.shape[1]):
                tree = HistGradientBoostingRegressor(
                    max_leaf_nodes=settings.get("max_leaf_nodes", 7),
                    max_iter=80,
                    min_samples_leaf=32,
                    l2_regularization=10.0,
                    max_bins=32,
                    early_stopping=False,
                    random_state=seed,
                )
                tree.fit(f[train], reward[train, j])
                models.append(tree)
            model = GainEnsemble(models)
        else:
            model = ExtraTreesRegressor(
                n_estimators=settings["n_estimators"],
                max_depth=settings["max_depth"],
                min_samples_leaf=settings["min_samples_leaf"],
                max_features=0.7,
                random_state=seed,
                n_jobs=1,
            )
            model.fit(f[train], reward[train])
        scores = model.predict(f[calibration])
        scores[:, 0] = 0
        chosen = scores.argmax(1)
        best = scores[np.arange(len(scores)), chosen]
        active = chosen != 0
        edges = (
            np.unique(
                np.quantile(
                    best[active], np.linspace(0, 1, settings.get("calibration_bins", 3) + 1)[1:-1]
                )
            )
            if active.any()
            else np.empty(0)
        )
        records = {}
        cal_origins = origins[calibration]
        bins = np.searchsorted(edges, best, side="right")
        actual = g[calibration][np.arange(len(scores)), chosen]
        for b in range(len(edges) + 1):
            rows = active & (bins == b)
            ids = np.unique(cal_origins[rows])
            per_origin = np.array([actual[rows & (cal_origins == o)].mean() for o in ids])
            block_means = np.array(
                [v.mean() for v in np.array_split(per_origin, max(1, len(ids) // 6)) if len(v)]
            )
            se = (
                float(block_means.std(ddof=1) / np.sqrt(len(block_means)))
                if len(block_means) >= 2
                else None
            )
            records[b] = {
                "mean": float(per_origin.mean()) if len(ids) else 0.0,
                "standard_error": se,
                "origins": len(ids),
                "supported": len(ids) >= settings.get("minimum_calibration_origins", 12),
            }
        diagnostics = {
            "fitting_origin_max": int(origins[train].max()),
            "calibration_origin_min": int(cal_origins.min()),
            "fitting_origins": len(np.unique(origins[train])),
            "calibration_origins": len(np.unique(cal_origins)),
            "bins": records,
            "scope": "calibration of selected plan on held-out origin blocks; heuristic penalty, not guaranteed coverage",
        }
        return cls(model, edges, records, diagnostics)

    def select(self, features, *, strength=0.0, margin=0.0):
        if not np.isfinite(strength) or strength < 0 or (not np.isfinite(margin)) or (margin < 0):
            raise ValueError("nonnegative finite calibration strength and margin required")
        scores = self.model.predict(features)
        scores[:, 0] = 0
        chosen = scores.argmax(1)
        best = scores[np.arange(len(scores)), chosen]
        bins = np.searchsorted(self.edges, best, side="right")
        result = np.zeros(len(features), dtype=int)
        for b, r in self.records.items():
            penalty = (
                0.0
                if strength == 0
                else strength * r["standard_error"]
                if r["standard_error"] is not None
                else float("inf")
            )
            if r["supported"] and r["mean"] - penalty > margin:
                rows = (bins == b) & (chosen != 0)
                result[rows] = chosen[rows]
        return result
