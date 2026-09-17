import json
import shutil

import pytest


@pytest.fixture(scope='module')
def original(tmp_path_factory):
    from asofcast.data import write_synthetic_csv
    from asofcast.experiment import run_experiment
    root = tmp_path_factory.mktemp('bundle-schema')
    csv = root / 'source.csv'
    write_synthetic_csv(csv, n=720, seed=3)
    run_experiment(csv, root / 'bundle', {'lookback':12, 'epochs':1, 'policy_epochs':1}, source_kind='synthetic')
    return root / 'bundle'


@pytest.mark.parametrize('field,value', [
    ('channels', ['renamed']), ('target_channel', 100), ('policy_features', 3),
])
def test_bundle_rejects_inconsistent_manifest_schema(original, tmp_path, field, value):
    from asofcast.bundle import load_bundle
    bundle = tmp_path / 'copy'
    shutil.copytree(original, bundle)
    path = bundle / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='schema|channel|target|feature'):
        load_bundle(bundle)


def test_bundle_rejects_out_of_range_replay_origins(original, tmp_path):
    import numpy as np
    from asofcast.bundle import load_bundle
    from asofcast.data import sha256_file
    bundle = tmp_path / 'copy'
    shutil.copytree(original, bundle)
    path = bundle / 'replay.npz'
    with np.load(path, allow_pickle=False) as data:
        values = {k: data[k].copy() for k in data.files}
    values['test_origins'][0] = -1
    np.savez_compressed(path, **values)
    manifest_path = bundle / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['files']['replay.npz'] = sha256_file(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='origin'):
        load_bundle(bundle)


def test_serving_forecaster_prefers_calibrated_model(original):
    from asofcast.bundle import load_bundle, select_serving_forecaster
    bundle = load_bundle(original)
    model, name = select_serving_forecaster(bundle)
    assert model is bundle.calibrated
    assert name == 'staleness_calibrated_dlinear'
