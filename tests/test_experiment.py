from pathlib import Path

import numpy as np
import pytest
import torch


def small_config():
    return {'lookback': 12, 'horizon': 6, 'stride': 4, 'waits_seconds': [0, 1800, 3600],
            'epochs': 2, 'policy_epochs': 3, 'batch_size': 64, 'learning_rate': .003,
            'seed': 42, 'arrival_seed': 811, 'delay_cost': .02,
            'thresholds': [0., .02, .1, 1e6], 'calibration_ridges': [1.0, 100.0],
            'acquisition_epochs': 4, 'acquisition_cost_weight': .03,
            'acquisition_pareto_weights': [0., .02, .05], 'target': 'OT', 'cpu_threads': 2}


def test_real_training_pipeline_exports_loadable_verified_bundle(tmp_path):
    from asofcast.data import write_synthetic_csv
    from asofcast.experiment import run_experiment
    from asofcast.bundle import load_bundle
    source = tmp_path / 'synthetic.csv'
    write_synthetic_csv(source, n=720, seed=12)
    out = tmp_path / 'run'
    report = run_experiment(source, out, small_config(), source_kind='synthetic')
    assert report['source']['kind'] == 'synthetic'
    assert report['status'] == 'local_experiment_completed'
    assert report['paper_score_reproduced'] is False
    assert report['cloud_deployed'] is False
    assert report['partitions']['test']['samples'] > 0
    assert report['policy_selection']['split'] == 'validation'
    assert 'calibrated_learned' in report['test_metrics']
    assert report['test_metrics']['calibrated_learned']['mae'] >= 0
    assert report['calibration_selection']['split'] == 'validation'
    assert report['calibration_selection']['ridge'] in small_config()['calibration_ridges']
    acquisition = report['acquisition_policy']
    for key in ('immediate_mae', 'learned_mae', 'oracle_mae', 'acquisition_rate',
                'mean_cost_proxy', 'mean_realized_gain', 'mean_oracle_gain',
                'regret_to_oracle', 'oracle_top1_hit_rate'):
        assert np.isfinite(acquisition[key])
    assert 0 <= acquisition['acquisition_rate'] <= 1
    assert acquisition['oracle_mae'] <= acquisition['immediate_mae'] + 1e-6
    assert acquisition['regret_to_oracle'] >= -1e-6
    assert len(report['acquisition_pareto']) == len(small_config()['acquisition_pareto_weights'])
    assert all(np.isfinite(row['mae']) and np.isfinite(row['mean_cost_proxy'])
               for row in report['acquisition_pareto'])
    assert (out / 'acquisition.pt').stat().st_size > 100
    assert 'acquisition' in report['parameters']
    bundle = load_bundle(out)
    assert bundle.report['run_id'] == report['run_id']
    assert bundle.arrival is not None and bundle.policy is not None
    assert bundle.calibrated is not None
    from asofcast.bundle import select_serving_forecaster
    serving, serving_name = select_serving_forecaster(bundle)
    assert serving is bundle.calibrated
    assert serving_name == 'staleness_calibrated_dlinear'
    assert (out / 'test_predictions.csv').stat().st_size > 100
    assert all(torch.isfinite(v).all() for v in bundle.arrival.state_dict().values())


def test_tampered_weights_rejected(tmp_path):
    from asofcast.data import write_synthetic_csv
    from asofcast.experiment import run_experiment
    from asofcast.bundle import load_bundle
    source = tmp_path / 'source.csv'
    write_synthetic_csv(source, n=720, seed=12)
    out = tmp_path / 'run'
    run_experiment(source, out, small_config(), source_kind='synthetic')
    with (out / 'arrival.pt').open('ab') as f: f.write(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        load_bundle(out)


def test_ett_provenance_cannot_be_assigned_to_generated_values(tmp_path):
    from asofcast.data import write_synthetic_csv, load_source
    path = tmp_path / 'fake-ETTh1.csv'
    write_synthetic_csv(path, n=720, seed=1)
    with pytest.raises(ValueError, match='ETT'):
        load_source(path, source_kind='ett', arrival_seed=1)


def test_invalid_csv_not_sorted_or_filled_silently(tmp_path):
    from asofcast.data import load_source
    path = tmp_path / 'bad.csv'
    path.write_text('date,OT\n2024-01-02,1\n2024-01-01,2\n', encoding='utf-8')
    with pytest.raises(ValueError):
        load_source(path, source_kind='user-provided-csv', arrival_seed=1)


def test_wrong_source_kind_and_existing_output_rejected(tmp_path):
    from asofcast.data import write_synthetic_csv, load_source
    from asofcast.experiment import run_experiment
    source = tmp_path / 'source.csv'
    write_synthetic_csv(source, n=720, seed=1)
    with pytest.raises(ValueError): load_source(source, source_kind='real-factory', arrival_seed=1)
    out = tmp_path / 'exists'
    out.mkdir()
    with pytest.raises(FileExistsError):
        run_experiment(source, out, small_config(), source_kind='synthetic')
