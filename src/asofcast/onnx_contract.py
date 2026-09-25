"""Independent artifact checks; do not trust a report's 'verified' label alone."""
from __future__ import annotations

import numpy as np


def validate_onnx_report(report: dict) -> None:
    try:
        if report['status'] != 'verified' or report['schema_version'] != 2:
            raise ValueError('report is not verified schema v2')
        for key in ('passed', 'finite_outputs', 'matching_shapes', 'max_abs_drift_lte_1e-4'):
            if report['gate'].get(key) is not True:
                raise ValueError(f'failed gate: {key}')
        for key in ('1', '16', 'all'):
            parity = report['parity'][key]
            drift = parity['max_abs_drift_standardized']
            if parity['passed'] is not True or not np.isfinite(drift) or not 0 <= drift <= 1e-4:
                raise ValueError(f'invalid parity batch: {key}')
        drift = report['max_abs_drift_standardized']
        if not np.isfinite(drift) or not 0 <= drift <= 1e-4:
            raise ValueError('invalid maximum drift')
        meta = report['measurement']
        if meta['torch_inference_mode'] is not True or meta['torch_grad_enabled'] is not False:
            raise ValueError('timing does not match serving inference mode')
        if meta['intra_op_threads'] < 1 or len(report['rounds']) != meta['rounds'] or meta['rounds'] < 1:
            raise ValueError('invalid measurement counts')
        for engine in ('torch', 'onnxruntime'):
            samples = []
            for round_ in report['rounds']:
                values = round_['samples_ms'][engine]
                if len(values) != meta['repeats_per_round'] or len(values) < 2:
                    raise ValueError('missing timing samples')
                if not np.isfinite(values).all() or min(values) < 0:
                    raise ValueError('nonfinite or negative timing samples')
                samples.extend(values)
            summary = report[engine + '_latency']
            if summary['repeats'] != len(samples):
                raise ValueError('pooled sample count mismatch')
            for key, quantile in [('p50_ms', .5), ('p95_ms', .95)]:
                actual = summary[key]
                if not np.isfinite(actual) or not np.isclose(actual, np.quantile(samples, quantile), rtol=1e-10, atol=1e-12):
                    raise ValueError(f'{engine} {key} differs from raw samples')
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError(f'incomplete ONNX evidence: {exc}') from exc
