from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def test_ett_split_has_no_future_targets_or_normalizer_leakage():
    from asofcast.reference_eval import prepare_ett
    values = np.arange(17420 * 7, dtype=np.float64).reshape(-1, 7)
    first = prepare_ett(values)
    changed = values.copy(); changed[8640:] += 10000
    second = prepare_ett(changed)
    np.testing.assert_array_equal(first['mean'], second['mean'])
    np.testing.assert_array_equal(first['scale'], second['scale'])
    assert first['windows']['train'][0].shape == (8209, 336, 7)
    assert first['windows']['val'][1].shape == (2785, 96, 7)
    assert first['windows']['test'][1].shape == (2785, 96, 7)
    expected = (values[11520] - first['mean']) / first['scale']
    np.testing.assert_allclose(first['windows']['test'][1][0, 0], expected, rtol=1e-6)


def test_reference_weight_mapping_preserves_parameters_and_output():
    from asofcast.reference_eval import copy_reference_weights
    from asofcast.models import DLinear
    source = SimpleNamespace(Linear_Seasonal=torch.nn.Linear(336, 96),
                             Linear_Trend=torch.nn.Linear(336, 96))
    model = DLinear(336, 7, 96)
    copy_reference_weights(source, model)
    torch.testing.assert_close(model.seasonal.weight, source.Linear_Seasonal.weight)
    torch.testing.assert_close(model.trend.bias, source.Linear_Trend.bias)
    x = torch.randn(2, 336, 7)
    assert model(x).shape == (2, 96, 7)


def test_unpinned_reference_code_is_rejected_before_execution(tmp_path):
    from asofcast.reference_eval import load_official_model
    source = tmp_path / 'DLinear.py'
    marker = tmp_path / 'executed'
    source.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    with pytest.raises(ValueError, match='identity'):
        load_official_model(source)
    assert not marker.exists()


@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_invalid_measurements_rejected(value):
    from asofcast.reference_eval import prepare_ett
    data = np.zeros((14400, 7)); data[0, 0] = value
    with pytest.raises(ValueError, match='finite'):
        prepare_ett(data)


def test_failed_reference_download_invalidates_previous_success(monkeypatch, tmp_path):
    from asofcast import reference_eval
    out = tmp_path / 'evidence'
    out.mkdir()
    (out / 'reference-report.json').write_text('{"status":"verified"}')
    def unavailable(_path):
        raise OSError('controlled unavailable source')
    monkeypatch.setattr(reference_eval, 'download_ett', unavailable)
    with pytest.raises(OSError):
        reference_eval.evaluate_reference(tmp_path / 'source.csv', tmp_path / 'ref.py', out)
    import json
    report = json.loads((out / 'reference-report.json').read_text())
    assert report['status'] == 'failed'
    assert report['gate']['passed'] is False
