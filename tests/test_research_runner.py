import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from asofcast.bundle import load_bundle
from asofcast.data import write_synthetic_csv
from asofcast.experiment import run_experiment
from asofcast.preprocessing import partition_bounds
from asofcast.timeline import Timeline

pytest.importorskip('sklearn')


@pytest.fixture(scope='module')
def small_bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp('research')
    source = root / 'source.csv'
    write_synthetic_csv(source, n=720, seed=15)
    run_experiment(source, root / 'bundle', {'lookback': 12, 'epochs': 2,
                    'policy_epochs': 2, 'acquisition_epochs': 2}, source_kind='synthetic')
    return load_bundle(root / 'bundle')


def settings():
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / 'configs/research_diagnosis_20260929.json').read_text())
    protocol['forecast'].update(leaf_candidates=[3], iterations=2, min_samples_leaf=4)
    protocol['policy'].update(iterations=2, min_samples_leaf=4, margins=[0, .03])
    return protocol


def test_entire_selection_is_invariant_to_poisoned_final_test(small_bundle):
    from scripts.evaluate_research_diagnosis import fit_research
    bundle = small_bundle
    first = fit_research(bundle, settings())
    values = bundle.timeline.values.copy()
    values[partition_bounds(len(values))['test'][0]:] += 10000
    poisoned = replace(bundle, timeline=Timeline(bundle.timeline.times, values,
                        bundle.timeline.arrivals, bundle.timeline.columns))
    second = fit_research(poisoned, settings())
    assert first['selection'] == second['selection']
    for key in first['selection_probe']:
        np.testing.assert_array_equal(first['selection_probe'][key], second['selection_probe'][key])


def test_full_evaluation_returns_all_ablation_and_sequential_scores(small_bundle):
    from scripts.evaluate_research_diagnosis import evaluate_research, fit_research
    protocol = settings()
    fitted = fit_research(small_bundle, protocol)
    report, arrays = evaluate_research(small_bundle, fitted, protocol, 'outage', 1811)
    assert set(report['methods']) >= {'combined', 'policy_only', 'forecast_only',
                                     'legacy_joint', 'validation_simple', 'value_features'}
    for name in report['methods']:
        assert np.isfinite(arrays[name + '__objective']).all()
        assert not arrays[name + '__deadline_violation'].any()
    assert report['selection'] == fitted['selection']
