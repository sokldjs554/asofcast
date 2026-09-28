from dataclasses import replace

import numpy as np
import pytest

from asofcast.bundle import load_bundle
from asofcast.data import write_synthetic_csv
from asofcast.experiment import run_experiment
from asofcast.preprocessing import partition_bounds
from asofcast.timeline import Timeline
from scripts import evaluate_model_evidence as audit


@pytest.fixture(scope='module')
def evidence_bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp('evidence-bundle')
    source = root / 'source.csv'
    write_synthetic_csv(source, n=720, seed=9)
    destination = root / 'model'
    run_experiment(source, destination, {'lookback': 12, 'epochs': 2, 'policy_epochs': 3,
                                        'acquisition_epochs': 3}, source_kind='synthetic')
    return load_bundle(destination)


def test_candidate_fit_and_validation_choice_are_invariant_to_test_targets(evidence_bundle):
    bundle = evidence_bundle
    fitted = audit.fit_choices(bundle, [1., 10.], .03)
    values = bundle.timeline.values.copy()
    values[partition_bounds(len(values))['test'][0]:] += 10000
    poisoned = replace(bundle, timeline=Timeline(bundle.timeline.times, values,
                                                bundle.timeline.arrivals, bundle.timeline.columns))
    again = audit.fit_choices(poisoned, [1., 10.], .03)
    assert fitted.selection == again.selection
    np.testing.assert_array_equal(fitted.ridge.coefficients, again.ridge.coefficients)
    np.testing.assert_array_equal(fitted.prior, again.prior)


def test_real_model_evaluation_keeps_origins_and_all_scores_finite(evidence_bundle):
    fitted = audit.fit_choices(evidence_bundle, [1., 10.], .03)
    report, arrays = audit.evaluate_condition(evidence_bundle, fitted, 'outage', 1811, .03)
    np.testing.assert_array_equal(arrays['origins'], evidence_bundle.test_origins)
    for method in ('mlp', 'ridge', 'simple', 'commit', 'random_mlp', 'random_ridge'):
        assert report['acquisition'][method]['n'] == len(evidence_bundle.test_origins)
        assert np.isfinite(arrays[f'acquisition__{method}__objective']).all()
    assert report['selection'] == fitted.selection
    assert set(report['forecast']) >= {'dlinear', 'calibrated', 'persistence'}
    assert set(report['wait']) >= {'learned', 'validation_fixed', 'random_matched'}


def test_dataset_integrity_rejects_wrong_bytes_without_replacing_file(tmp_path):
    source = tmp_path / 'ETTh1.csv'
    source.write_bytes(b'wrong but existing source')
    with pytest.raises(ValueError, match='integrity'):
        audit.fetch_dataset({'name': 'ETTh1', 'blob_sha1': '0' * 40}, tmp_path)
    assert source.read_bytes() == b'wrong but existing source'


def test_protocol_cannot_count_one_training_seed_as_three_independent_repeats():
    import json
    protocol = json.loads((audit.ROOT / 'configs/model_evidence_20260928.json').read_text())
    protocol['seeds'] = [42, 42, 42]
    with pytest.raises(ValueError, match='distinct'):
        audit.validate_protocol(protocol)
