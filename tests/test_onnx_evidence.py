import copy

import numpy as np
import pytest


def valid_report():
    samples = {'torch': [0.2, 0.3], 'onnxruntime': [0.1, 0.2]}
    summary = lambda v: {'p50_ms': float(np.quantile(v, .5)), 'p95_ms': float(np.quantile(v, .95)), 'repeats': len(v)}
    return {'status': 'verified', 'schema_version': 2,
            'gate': {'passed': True, 'finite_outputs': True, 'matching_shapes': True,
                     'max_abs_drift_lte_1e-4': True},
            'max_abs_drift_standardized': 0.0,
            'parity': {label: {'passed': True, 'max_abs_drift_standardized': 0.0}
                       for label in ('1', '16', 'all')},
            'measurement': {'torch_inference_mode': True, 'torch_grad_enabled': False,
                            'intra_op_threads': 2, 'rounds': 1, 'repeats_per_round': 2},
            'rounds': [{'round': 1, 'samples_ms': samples}],
            'torch_latency': summary(samples['torch']),
            'onnxruntime_latency': summary(samples['onnxruntime'])}


def test_contract_accepts_recalculated_evidence():
    from asofcast.onnx_contract import validate_onnx_report
    validate_onnx_report(valid_report())


@pytest.mark.parametrize('change', ['failed', 'gate', 'nan', 'shape', 'mode', 'samples', 'summary'])
def test_contract_rejects_false_or_inconsistent_verification(change):
    from asofcast.onnx_contract import validate_onnx_report
    r = copy.deepcopy(valid_report())
    if change == 'failed': r['status'] = 'failed'
    elif change == 'gate': r['gate']['passed'] = False
    elif change == 'nan': r['max_abs_drift_standardized'] = float('nan')
    elif change == 'shape': r['parity']['16']['passed'] = False
    elif change == 'mode': r['measurement']['torch_grad_enabled'] = True
    elif change == 'samples': r['rounds'][0]['samples_ms']['torch'][0] = float('inf')
    elif change == 'summary': r['torch_latency']['p95_ms'] = 9.0
    with pytest.raises(ValueError):
        validate_onnx_report(r)
