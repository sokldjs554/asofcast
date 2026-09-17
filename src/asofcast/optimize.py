"""Measure PyTorch FP32 vs CPU dynamic INT8; keep slower candidates unpromoted."""
from __future__ import annotations

import copy
import io
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn

from asofcast.bundle import load_bundle
from asofcast.data import write_json
from asofcast.experiment import build_examples, build_policy_states
from asofcast.metrics import score
from asofcast.policy import choose_steps
from asofcast.timeline import Timeline
from asofcast.training import predict


def dynamic_int8(model: nn.Module) -> tuple[nn.Module,list[str]]:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        candidate = torch.ao.quantization.quantize_dynamic(copy.deepcopy(model).cpu().eval(),
                                                          {nn.Linear},dtype=torch.qint8)
    notes = sorted({str(w.message) for w in captured})
    return candidate,notes


def measure_latency(model: nn.Module, sample: torch.Tensor, *, warmup: int = 30,
                    repeats: int = 200) -> dict:
    if warmup < 0 or repeats < 2:
        raise ValueError('invalid timing configuration')
    model.eval()
    milliseconds = []
    with torch.inference_mode():
        for _ in range(warmup): model(sample)
        for _ in range(repeats):
            start = time.perf_counter_ns()
            model(sample)
            milliseconds.append((time.perf_counter_ns()-start)/1e6)
    return {'p50_ms':float(np.quantile(milliseconds,.5)), 'p95_ms':float(np.quantile(milliseconds,.95)),
            'p99_ms':float(np.quantile(milliseconds,.99)), 'repeats':repeats, 'warmup':warmup,
            'batch_size':len(sample), 'scope':'model_forward_only',
            'threads':torch.get_num_threads()}


def _weight_bytes(model: nn.Module) -> int:
    buffer = io.BytesIO()
    torch.save(model.state_dict(),buffer)
    return buffer.tell()


def benchmark_bundle(directory: Path, output: Path, *, repeats: int = 300) -> dict:
    bundle = load_bundle(directory)
    torch.set_num_threads(bundle.config['cpu_threads'])
    raw = bundle.timeline
    timeline = Timeline(raw.times,bundle.scaler.transform(raw.values),raw.arrivals,raw.columns)
    target = bundle.manifest['target_channel']
    x,y = build_examples(timeline,bundle.test_origins,bundle.config,target)
    flat = x.reshape((-1,)+x.shape[2:])
    candidate,notes = dynamic_int8(bundle.arrival)
    fp = predict(bundle.arrival,flat).reshape(x.shape[:2])
    qp = predict(candidate,flat).reshape(x.shape[:2])
    waits = bundle.config['waits_seconds']
    threshold = bundle.report['policy_selection']['threshold']
    fp_steps = choose_steps(build_policy_states(x,fp,waits),bundle.policy,threshold,waits)
    qp_steps = choose_steps(build_policy_states(x,qp,waits),bundle.policy,threshold,waits)
    scale,mean = bundle.scaler.scale[target],bundle.scaler.mean[target]
    fp_quality = score(y*scale+mean,fp*scale+mean,fp_steps,waits)
    qp_quality = score(y*scale+mean,qp*scale+mean,qp_steps,waits)
    sample = torch.from_numpy(flat[:1].copy())
    fp_latency = measure_latency(bundle.arrival,sample,repeats=repeats)
    qp_latency = measure_latency(candidate,sample,repeats=repeats)
    action_change = float(np.mean(fp_steps != qp_steps))
    gate = {'p95_improved':qp_latency['p95_ms'] < fp_latency['p95_ms'],
            'mae_within_one_percent':qp_quality['mae'] <= fp_quality['mae']*1.01+1e-8,
            'action_changes_within_five_percent':action_change <= .05}
    result = {'run_id':bundle.report['run_id'], 'source_kind':bundle.report['source']['kind'],
              'method':'torch.ao dynamic quantization of Linear layers; CPU',
              'torch_version':torch.__version__, 'onnx_executed':False,
              'fp32':{'latency':fp_latency,'quality':fp_quality,'serialized_weight_bytes':_weight_bytes(bundle.arrival)},
              'int8':{'latency':qp_latency,'quality':qp_quality,'serialized_weight_bytes':_weight_bytes(candidate)},
              'max_prediction_drift_native':float(np.abs(fp-qp).max()*scale),
              'policy_action_change_rate':action_change, 'gates':gate,
              'candidate_decision':'eligible_for_further_validation' if all(gate.values()) else 'keep_fp32',
              'promotion_scope':'diagnostic gates on current test set, NOT a production promotion',
              'api_model_changed':False, 'deprecation_notes':notes,
              'timing_limitations':'shared CPU, model-forward only, not HTTP or network latency; repeat on deployment hardware'}
    write_json(output,result)
    return result
