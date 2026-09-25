"""Fault injection tests exercise the real exporter/evaluator/CLI control flow.
The external runtime is a test double; these are NOT real ONNX speed evidence.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from asofcast import onnx_eval
from asofcast.cli import main


@pytest.fixture
def runtime_case(monkeypatch, tmp_path):
    from asofcast.data import write_synthetic_csv
    from asofcast.experiment import run_experiment
    from asofcast.bundle import load_bundle, select_serving_forecaster
    path = tmp_path / 'bundle'
    csv = tmp_path / 'source.csv'
    write_synthetic_csv(csv, n=400, seed=9)
    run_experiment(csv, path, {'lookback': 12, 'epochs': 1, 'policy_epochs': 1,
                              'acquisition_epochs': 1}, source_kind='synthetic')
    bundle = load_bundle(path)
    model, _ = select_serving_forecaster(bundle)
    state = SimpleNamespace(mode='ok', calls=[], options=None)
    class Session:
        def __init__(self, *args, **kwargs):
            state.options = kwargs.get('sess_options')
        def get_providers(self):
            return ['CPUExecutionProvider']
        def run(self, names, feeds):
            if state.mode == 'runtime_error':
                raise Exception('injected native runtime failure')
            with torch.inference_mode():
                output = model(torch.from_numpy(feeds['features'])).cpu().numpy().copy()
            if state.mode == 'drift':
                output += 1.0
            elif state.mode == 'nan':
                output[:] = np.nan
            elif state.mode == 'inf':
                output[:] = np.inf
            elif state.mode == 'shape':
                output = output[:, None]
            return [output]
    class Options:
        def add_session_config_entry(self, *args):
            pass
    ort = SimpleNamespace(InferenceSession=Session, SessionOptions=Options,
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL=0),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL=99),
        __version__='FAULT_INJECTION_NOT_REAL_ORT')
    def export(forecaster, sample, out, **kwargs):
        Path(out).write_bytes(b'TEST DOUBLE NOT ONNX')
    def select(bundle):
        real_model, name = select_serving_forecaster(bundle)
        def hook(_m, _inp, out):
            state.calls.append((torch.is_grad_enabled(), torch.is_inference_mode_enabled(),
                                bool(out.requires_grad)))
        real_model.register_forward_hook(hook)
        return real_model, name
    monkeypatch.setattr(onnx_eval, '_optional_runtime', lambda: ort)
    monkeypatch.setattr(onnx_eval, 'select_serving_forecaster', select)
    monkeypatch.setattr(torch.onnx, 'export', export)
    return path, state


def invoke(path, out, *extra):
    return main(['onnx-eval', '--artifacts', str(path), '--out', str(out),
                 '--repeats', '2', *extra])


def test_timing_matches_serving_inference_mode(runtime_case, tmp_path):
    path, state = runtime_case
    assert invoke(path, tmp_path / 'report.json') == 0
    assert state.calls
    assert all(call == (False, True, False) for call in state.calls)


@pytest.mark.parametrize('mode', ['drift', 'nan', 'inf', 'shape', 'runtime_error'])
def test_bad_runtime_output_fails_cli_and_never_verifies(runtime_case, tmp_path, mode):
    path, state = runtime_case
    state.mode = mode
    out = tmp_path / 'report.json'
    out.write_text('{"status":"verified","old":true}')
    assert invoke(path, out) != 0
    report = json.loads(out.read_text(), parse_constant=lambda x: pytest.fail(x))
    assert report['status'] == 'failed'
    assert report['gate']['passed'] is False
    assert 'old' not in report
    assert 'torch_latency' not in report


def test_valid_output_has_explicit_conditions_and_rounds(runtime_case, tmp_path):
    path, state = runtime_case
    out = tmp_path / 'nested' / 'report.json'
    old_threads = torch.get_num_threads()
    assert invoke(path, out, '--threads', '1', '--warmup', '2', '--rounds', '3') == 0
    result = json.loads(out.read_text())
    assert result['status'] == 'verified' and result['gate']['passed'] is True
    assert result['measurement']['torch_inference_mode'] is True
    assert result['measurement']['intra_op_threads'] == 1
    assert result['measurement']['warmup_per_engine'] == 2
    assert len(result['rounds']) == 3
    assert result['torch_latency']['repeats'] == 6
    assert torch.get_num_threads() == old_threads
    assert state.options.intra_op_num_threads == 1


def test_dependency_failure_overwrites_stale_verified_report(monkeypatch, tmp_path):
    def missing():
        raise RuntimeError('onnx dependency unavailable')
    monkeypatch.setattr(onnx_eval, '_optional_runtime', missing)
    out = tmp_path / 'report.json'
    out.write_text('{"status":"verified"}')
    assert invoke(tmp_path / 'missing', out) != 0
    assert json.loads(out.read_text())['status'] == 'failed'
