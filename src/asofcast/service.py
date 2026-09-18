"""Local model-backed replay dashboard and stateless live inference.

No authentication or public-production SLA is claimed. The default CLI bind is
127.0.0.1. Retrospective labels appear only in the replay audit endpoint.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from asofcast.bundle import Bundle, load_bundle, select_serving_forecaster
from asofcast.policy import policy_features
from asofcast.preprocessing import simulate_arrivals
from asofcast.timeline import Snapshot, Timeline
from asofcast.training import predict

STATIC = Path(__file__).with_name('static')
LOGGER = logging.getLogger(__name__)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    event_times: list[int] = Field(min_length=2, max_length=512)
    values: list[list[FiniteFloat]] = Field(min_length=2, max_length=512)
    arrival_times: list[list[FiniteFloat | None]] = Field(min_length=2, max_length=512)
    columns: list[str] = Field(min_length=1, max_length=64)
    wait_seconds: FiniteFloat = Field(ge=0)


def _infer(bundle: Bundle, normalized: Snapshot, raw: Snapshot, step: int) -> dict:
    started = time.perf_counter()
    waits = bundle.config['waits_seconds']
    x = normalized.features()[None]
    forecaster, _ = select_serving_forecaster(bundle)
    prediction = float(predict(forecaster, x)[0])
    baseline = float(predict(bundle.dlinear, x)[0])
    target = bundle.manifest['target_channel']
    scale, mean = bundle.scaler.scale[target], bundle.scaler.mean[target]
    final_step = step == len(waits) - 1
    gain, native_gain, minimum_gain = None, None, None
    action = 'COMMIT'
    if not final_step:
        state = policy_features(x, np.array([prediction], dtype=np.float32), 1 - waits[step]/waits[-1])
        gain = float(predict(bundle.policy, state)[0])
        native_gain = gain * scale
        minimum_gain = bundle.report['policy_selection']['threshold'] * (waits[step+1]-waits[step])/waits[-1]
        action = 'WAIT' if gain > minimum_gain else 'COMMIT'
    return {'step': step, 'wait_seconds': waits[step], 'origin_time': raw.origin_time,
            'decision_time': raw.decision_time, 'target_time': raw.target_time,
            'prediction': prediction * scale + mean, 'baseline_prediction': baseline * scale + mean,
            'expected_next_error_reduction': native_gain,
            'minimum_gain_to_wait': minimum_gain * scale if minimum_gain is not None else None,
            'action': action, 'next_check_seconds': waits[step+1] if action == 'WAIT' else None,
            'latest_values': [float(v) if known else None for v, known in zip(raw.values[-1],raw.known[-1],strict=True)],
            'latest_observed': raw.observed[-1].tolist(),
            'latest_source_times': [int(v) if known else None for v,known in zip(raw.source_times[-1],raw.known[-1],strict=True)],
            'latest_age_seconds': [float(v) if known else None for v,known in zip(raw.age_seconds[-1],raw.known[-1],strict=True)],
            'window_observed_fraction': float(raw.observed.mean()),
            'compute_ms': (time.perf_counter()-started)*1000,
            'compute_scope': 'feature conversion + arrival + baseline + policy, HTTP excluded',
            'deadline_scope': 'logical acquisition decision, not an end-to-end production SLA'}


def create_app(artifact_dir: Path) -> FastAPI:
    app = FastAPI(title='AsOfCast', version='0.1.0', description='Frozen-origin forecasting with audited data arrivals')
    app.state.bundle = None
    app.state.timelines = {}
    try:
        bundle = load_bundle(artifact_dir)
        torch.set_num_threads(bundle.config['cpu_threads'])
        app.state.bundle = bundle
        app.state.timelines['mixed'] = bundle.timeline
        raw = bundle.timeline
        outage = simulate_arrivals(raw.times, len(raw.columns), bundle.config['arrival_seed'], profile='outage')
        app.state.timelines['outage'] = Timeline(raw.times, raw.values, outage, raw.columns)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        LOGGER.warning('AsOfCast bundle not ready: %s', exc)

    def require_bundle() -> Bundle:
        if app.state.bundle is None:
            raise HTTPException(503, detail='MODEL_BUNDLE_NOT_READY')
        return app.state.bundle

    @app.get('/health')
    def health():
        return {'status':'alive'}

    @app.get('/ready')
    def ready(bundle: Bundle = Depends(require_bundle)):
        return {'status':'ready', 'run_id':bundle.report['run_id'], 'checksums_verified':True}

    @app.get('/')
    def index():
        return FileResponse(STATIC / 'index.html')

    @app.get('/api/metadata')
    def metadata(bundle: Bundle = Depends(require_bundle)):
        return {'run_id':bundle.report['run_id'], 'source_kind':bundle.report['source']['kind'],
                'serving_forecaster':select_serving_forecaster(bundle)[1],
                'source':bundle.report['source'], 'cases':len(bundle.test_origins),
                'columns':list(bundle.timeline.columns), 'config':bundle.config,
                'test_metrics':bundle.report['test_metrics'],
                'outage_test_metrics':bundle.report['outage_test_metrics'],
                'limitations':bundle.report['limitations'], 'parameters':bundle.report['parameters'],
                'paper_score_reproduced':False,
                'cloud_deployed':os.environ.get('ASOFCAST_CLOUD_DEPLOYED','').strip().lower() in {'1','true','yes'}}

    @app.get('/api/replay')
    def replay(case_id: int = Query(default=0, ge=0), scenario: Literal['mixed','outage'] = 'mixed',
               bundle: Bundle = Depends(require_bundle)):
        if case_id >= len(bundle.test_origins):
            raise HTTPException(404, detail='CASE_NOT_FOUND')
        raw = app.state.timelines[scenario]
        normalized = Timeline(raw.times, bundle.scaler.transform(raw.values), raw.arrivals, raw.columns)
        origin = int(bundle.test_origins[case_id])
        steps = []
        for step, wait in enumerate(bundle.config['waits_seconds']):
            raw_snap = raw.snapshot(origin, wait, bundle.config['lookback'], bundle.config['horizon'])
            norm_snap = normalized.snapshot(origin, wait, bundle.config['lookback'], bundle.config['horizon'])
            steps.append(_infer(bundle, norm_snap, raw_snap, step))
        chosen = next((s['step'] for s in steps if s['action']=='COMMIT'), len(steps)-1)
        target = bundle.manifest['target_channel']
        return {'run_id':bundle.report['run_id'], 'source_kind':bundle.report['source']['kind'],
                'serving_forecaster':select_serving_forecaster(bundle)[1],
                'scenario':scenario, 'case_id':case_id, 'origin_time':int(raw.times[origin]),
                'target_time':int(raw.times[origin]+bundle.config['horizon']*raw.grid_seconds),
                'target_actual':float(raw.values[origin+bundle.config['horizon'],target]),
                'ground_truth_scope':'retrospective_only_not_a_policy_input',
                'selected_step':chosen, 'steps':steps}

    @app.post('/api/predict')
    def online(payload: PredictRequest, bundle: Bundle = Depends(require_bundle)):
        if payload.columns != list(bundle.timeline.columns):
            raise HTTPException(422, detail='columns must exactly match model channel order')
        if payload.wait_seconds not in bundle.config['waits_seconds']:
            raise HTTPException(422, detail='wait_seconds must be a trained decision-grid value')
        try:
            arrivals = np.asarray([[np.inf if v is None else v for v in row] for row in payload.arrival_times],dtype=float)
            raw = Timeline(np.asarray(payload.event_times),np.asarray(payload.values),arrivals,tuple(payload.columns))
            if raw.grid_seconds != bundle.timeline.grid_seconds:
                raise ValueError('event grid differs from training grid')
            norm = Timeline(raw.times,bundle.scaler.transform(raw.values),raw.arrivals,raw.columns)
            origin = len(raw.times)-1
            step = bundle.config['waits_seconds'].index(payload.wait_seconds)
            raw_snap = raw.snapshot(origin,payload.wait_seconds,bundle.config['lookback'],bundle.config['horizon'])
            norm_snap = norm.snapshot(origin,payload.wait_seconds,bundle.config['lookback'],bundle.config['horizon'])
            result = _infer(bundle,norm_snap,raw_snap,step)
        except (ValueError, IndexError) as exc:
            raise HTTPException(422, detail=str(exc)) from exc
        return {'run_id':bundle.report['run_id'],
                'model_training_source':bundle.report['source']['kind'],
                'input_provenance':'caller-provided, not independently verified', **result}

    app.mount('/static',StaticFiles(directory=STATIC),name='static')
    return app


def default_app() -> FastAPI:
    return create_app(Path(os.environ.get('ASOFCAST_ARTIFACTS','artifacts/demo')))
