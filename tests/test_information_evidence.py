"""Check published numerical claims against the archived per-origin outcomes."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize('report_name,loss_name', [
    ('report.json', 'paired-losses.npz'),
    ('component-report.json', 'component-losses.npz'),
])
def test_published_information_losses_include_every_paid_action_and_wait(report_name, loss_name):
    root = Path(__file__).resolve().parents[1] / 'docs/assets/information-value'
    assert (root / report_name).is_file(), 'publish the completed evidence before merging'
    report = json.loads((root / report_name).read_text())
    primary = json.loads((root / 'report.json').read_text())
    provenance = json.loads((root / 'provenance.json').read_text())
    path = root / loss_name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == report['raw_losses_sha256']
    assert len(report['runs']) == 18
    with np.load(path, allow_pickle=False) as arrays:
        for prefix, run in report['runs'].items():
            dataset, seed, _ = prefix.split('__')
            bundle = primary['bundles'][f'{dataset}-{seed}']
            target = bundle['manifest']['target_channel']
            scale = provenance['scalers'][f'{dataset}-{seed}']['scale'][target]
            deadline = bundle['config']['waits_seconds'][-1]
            y = arrays[prefix + '__target_standardized']
            for method, metrics in run['methods'].items():
                key = f'{prefix}__{method}__'
                error = np.abs(arrays[key + 'prediction'] - y)
                objective = error + .03 * arrays[key + 'cost'] + .02 * arrays[key + 'wait_seconds'] / deadline
                np.testing.assert_allclose(arrays[key + 'objective'], objective, atol=1e-12)
                np.testing.assert_allclose(arrays[key + 'native_error'], error * scale, atol=1e-10)
                assert np.max(arrays[key + 'wait_seconds']) <= deadline
                for metric, mean in metrics.items():
                    if metric == 'origins':
                        assert mean == len(y)
                        continue
                    np.testing.assert_allclose(mean, arrays[key + metric].mean(), atol=1e-12)
