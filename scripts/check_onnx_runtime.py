"""Required real-ORT measurements plus labelled corruption canaries.

This script fails, rather than skipping, when native ONNX dependencies are absent.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from asofcast.cli import main as cli
from asofcast.onnx_contract import validate_onnx_report
from asofcast.onnx_eval import _optional_runtime, evaluate_onnx


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    native = _optional_runtime()
    measurements = []
    for repeat in range(1, 4):
        path = args.out / f'onnx-{repeat}.json'
        result = evaluate_onnx(args.artifacts, path, repeats=300, threads=2, warmup=30, rounds=3)
        validate_onnx_report(result)
        measurements.append({'report': path.name, 'torch': result['torch_latency'],
                             'onnxruntime': result['onnxruntime_latency']})
    canaries = []
    for mode in ('drift', 'nan', 'inf', 'shape', 'runtime_error'):
        class CorruptedSession:
            def __init__(self, *a, **kw):
                self.session = native.InferenceSession(*a, **kw)
            def get_providers(self):
                return self.session.get_providers()
            def run(self, *a, **kw):
                if mode == 'runtime_error':
                    raise RuntimeError('CANARY: injected runtime failure')
                values = self.session.run(*a, **kw)
                value = values[0].copy()
                if mode == 'drift': value += 1.0
                elif mode == 'nan': value[:] = np.nan
                elif mode == 'inf': value[:] = np.inf
                elif mode == 'shape': value = value[:, None]
                return [value]
        wrapper = SimpleNamespace(InferenceSession=CorruptedSession,
                                   SessionOptions=native.SessionOptions,
                                   ExecutionMode=native.ExecutionMode,
                                   GraphOptimizationLevel=native.GraphOptimizationLevel,
                                   __version__=native.__version__)
        path = args.out / 'canaries' / f'{mode}.json'
        with patch('asofcast.onnx_eval._optional_runtime', return_value=wrapper):
            code = cli(['onnx-eval', '--artifacts', str(args.artifacts),
                        '--out', str(path), '--repeats', '2'])
        report = json.loads(path.read_text())
        if code == 0 or report['status'] != 'failed' or report['gate']['passed'] is not False:
            raise RuntimeError(f'corruption canary incorrectly succeeded: {mode}')
        canaries.append({'fault_injection': mode, 'exit_code': code,
                         'report': str(path.relative_to(args.out)), 'blocked': True})
    summary = {'status': 'passed', 'source_commit': os.getenv('GITHUB_SHA'),
               'real_onnxruntime_executed': True, 'measurements': measurements,
               'canaries': canaries,
               'canary_scope': 'real export/runtime outputs deliberately corrupted by a test adapter; not model-quality evidence'}
    (args.out / 'runtime-verification.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
