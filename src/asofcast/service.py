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

from asofcast.acquisition import predict_candidate_gains, select_joint_action
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


class AcquireRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    case_id: int = Field(ge=0)
    scenario: Literal['mixed', 'outage'] = 'mixed'
    wait_seconds: FiniteFloat = Field(ge=0)
    acquired: list[str] = Field(default_factory=list, max_length=64)
    sensor: str


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


def _model_disagreement(bundle: Bundle, x: np.ndarray) -> dict:
    models = [bundle.dlinear, bundle.value_only, bundle.arrival]
    if bundle.calibrated is not None:
        models.append(bundle.calibrated)
    standardized = np.asarray([float(predict(model, x)[0]) for model in models], dtype=float)
    target = bundle.manifest['target_channel']
    scale = float(bundle.scaler.scale[target])
    native = standardized * scale + float(bundle.scaler.mean[target])
    return {
        'value': float(np.std(native)),
        'label': 'model_disagreement_not_calibrated_uncertainty',
        'model_count': len(models),
    }


def _acquisition_state(bundle: Bundle, raw: Timeline, case_id: int, wait_seconds: float,
                       acquired_names: list[str]) -> dict:
    if bundle.acquisition is None or bundle.acquisition_cost_proxy is None:
        raise HTTPException(503, detail='ACQUISITION_MODEL_NOT_READY')
    if case_id >= len(bundle.test_origins):
        raise HTTPException(404, detail='CASE_NOT_FOUND')
    waits = bundle.config['waits_seconds']
    if wait_seconds not in waits:
        raise HTTPException(422, detail='wait_seconds must be a trained decision-grid value')
    if len(set(acquired_names)) != len(acquired_names):
        raise HTTPException(422, detail='acquired sensors must be unique')
    unknown = [name for name in acquired_names if name not in raw.columns]
    if unknown:
        raise HTTPException(422, detail=f'unknown acquired sensor: {unknown[0]}')
    acquired_indices = [raw.columns.index(name) for name in acquired_names]
    origin = int(bundle.test_origins[case_id])
    step = waits.index(wait_seconds)
    normalized_timeline = Timeline(raw.times, bundle.scaler.transform(raw.values), raw.arrivals, raw.columns)
    raw_snapshot = raw.snapshot(origin, wait_seconds, bundle.config['lookback'],
                                bundle.config['horizon'], acquired_channels=acquired_indices)
    normalized_snapshot = normalized_timeline.snapshot(
        origin, wait_seconds, bundle.config['lookback'], bundle.config['horizon'],
        acquired_channels=acquired_indices)
    x = normalized_snapshot.features()[None]
    forecaster, serving_name = select_serving_forecaster(bundle)
    standardized_prediction = float(predict(forecaster, x)[0])
    target = bundle.manifest['target_channel']
    scale = float(bundle.scaler.scale[target])
    mean = float(bundle.scaler.mean[target])
    prediction = standardized_prediction * scale + mean

    # Candidate eligibility depends only on the frozen origin slot's current availability.
    origin_observed = raw_snapshot.observed[-1].astype(bool)
    eligible = ~origin_observed
    if acquired_indices:
        eligible[np.asarray(acquired_indices, dtype=int)] = False
    remaining = 1.0 - wait_seconds / waits[-1]
    predicted = predict_candidate_gains(
        bundle.acquisition, normalized_snapshot.features(), standardized_prediction,
        bundle.acquisition_cost_proxy, eligible, remaining)

    if step < len(waits) - 1:
        wait_features = policy_features(x, np.array([standardized_prediction], dtype=np.float32), remaining)
        wait_gain = float(predict(bundle.policy, wait_features)[0])
        wait_cost = (bundle.report['policy_selection']['threshold']
                     * (waits[step + 1] - waits[step]) / waits[-1])
    else:
        wait_gain, wait_cost = 0.0, 0.0

    action = select_joint_action(
        wait_gain=wait_gain, wait_cost=wait_cost,
        acquisition_gains=predicted, acquisition_costs=bundle.acquisition_cost_proxy,
        eligible=eligible, acquisition_cost_weight=bundle.config['acquisition_cost_weight'])

    target_actual = float(raw.values[origin + bundle.config['horizon'], target])
    current_error = abs(prediction - target_actual)
    candidates = []
    for channel, name in enumerate(raw.columns):
        realized = None
        counterfactual_prediction = None
        if eligible[channel]:
            cf_raw = raw.snapshot(origin, wait_seconds, bundle.config['lookback'],
                                  bundle.config['horizon'],
                                  acquired_channels=[*acquired_indices, channel])
            cf_norm = normalized_timeline.snapshot(
                origin, wait_seconds, bundle.config['lookback'], bundle.config['horizon'],
                acquired_channels=[*acquired_indices, channel])
            cf_standardized = float(predict(forecaster, cf_norm.features()[None])[0])
            counterfactual_prediction = cf_standardized * scale + mean
            realized = current_error - abs(counterfactual_prediction - target_actual)
        predicted_native = float(predicted[channel] * scale) if np.isfinite(predicted[channel]) else None
        utility = (float(predicted[channel] - bundle.config['acquisition_cost_weight']
                         * bundle.acquisition_cost_proxy[channel])
                   if np.isfinite(predicted[channel]) and eligible[channel] else None)
        candidates.append({
            'sensor': name,
            'eligible': bool(eligible[channel]),
            'already_acquired': name in acquired_names,
            'passively_observed_origin': bool(origin_observed[channel] and name not in acquired_names),
            'predicted_gain': predicted_native,
            'predicted_gain_standardized': float(predicted[channel]) if np.isfinite(predicted[channel]) else None,
            'realized_gain_retrospective': float(realized) if realized is not None else None,
            'counterfactual_prediction_retrospective': float(counterfactual_prediction)
                if counterfactual_prediction is not None else None,
            'cost_proxy': float(bundle.acquisition_cost_proxy[channel]),
            'utility': utility,
        })
    candidates.sort(key=lambda row: (
        row['utility'] is not None,
        row['utility'] if row['utility'] is not None else float('-inf')
    ), reverse=True)

    recommended_sensor = raw.columns[action['channel']] if action['channel'] is not None else None
    latest_source_times = [
        int(value) if known else None
        for value, known in zip(raw_snapshot.source_times[-1], raw_snapshot.known[-1], strict=True)
    ]
    return {
        'run_id': bundle.report['run_id'],
        'source_kind': bundle.report['source']['kind'],
        'serving_forecaster': serving_name,
        'scenario': 'outage' if raw is not bundle.timeline else 'mixed',
        'case_id': case_id,
        'columns': list(raw.columns),
        'acquired': list(acquired_names),
        'origin_time': int(raw.times[origin]),
        'decision_time': raw_snapshot.decision_time,
        'target_time': raw_snapshot.target_time,
        'prediction': float(prediction),
        'target_actual_retrospective': target_actual,
        'counterfactual_scope': 'retrospective_audit_only_not_policy_input',
        'acquisition_scope': 'selected_sensor_origin_slot_only',
        'recommended_action': action['action'],
        'recommended_sensor': recommended_sensor,
        'recommended_net_utility': float(action['net_utility']),
        'wait_gain_predicted': float(wait_gain * scale),
        'wait_cost_standardized': float(wait_cost),
        'next_wait_seconds': waits[step + 1] if action['action'] == 'WAIT' and step < len(waits) - 1 else None,
        'available_origin_sensors': int(origin_observed.sum()),
        'total_sensors': len(raw.columns),
        'disagreement_proxy': _model_disagreement(bundle, x),
        'candidates': candidates,
        'latest_values': [
            float(value) if known else None
            for value, known in zip(raw_snapshot.values[-1], raw_snapshot.known[-1], strict=True)
        ],
        'latest_source_times': latest_source_times,
        'policy_inputs_scope': 'current_snapshot_only',
        'retrospective_fields_used_for_action': False,
    }


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
                'acquisition_policy':bundle.report.get('acquisition_policy'),
                'acquisition_pareto':bundle.report.get('acquisition_pareto', []),
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

    @app.get('/api/acquisition')
    def acquisition_state(case_id: int = Query(default=0, ge=0),
                          scenario: Literal['mixed','outage'] = 'mixed',
                          wait_seconds: float = Query(default=0, ge=0),
                          acquired: str = '',
                          bundle: Bundle = Depends(require_bundle)):
        names = [name for name in acquired.split(',') if name]
        return _acquisition_state(bundle, app.state.timelines[scenario], case_id,
                                  float(wait_seconds), names)

    @app.post('/api/acquire')
    def acquire(payload: AcquireRequest, bundle: Bundle = Depends(require_bundle)):
        raw = app.state.timelines[payload.scenario]
        if payload.sensor not in raw.columns:
            raise HTTPException(422, detail='unknown sensor')
        if payload.sensor in payload.acquired:
            raise HTTPException(422, detail='sensor already acquired')
        before = _acquisition_state(bundle, raw, payload.case_id,
                                    float(payload.wait_seconds), payload.acquired)
        row = next(item for item in before['candidates'] if item['sensor'] == payload.sensor)
        if not row['eligible']:
            raise HTTPException(422, detail='sensor is already available at this decision time')
        return _acquisition_state(bundle, raw, payload.case_id, float(payload.wait_seconds),
                                  [*payload.acquired, payload.sensor])

    @app.get('/api/revision-timeline')
    def revision_timeline(case_id: int = Query(default=0, ge=0),
                          scenario: Literal['mixed','outage'] = 'mixed',
                          bundle: Bundle = Depends(require_bundle)):
        raw = app.state.timelines[scenario]
        if case_id >= len(bundle.test_origins):
            raise HTTPException(404, detail='CASE_NOT_FOUND')
        events = []
        for wait in bundle.config['waits_seconds']:
            state = _acquisition_state(bundle, raw, case_id, float(wait), [])
            events.append({
                'kind': 'PASSIVE',
                'label': '즉시' if wait == 0 else f'{int(wait // 60)}분 대기',
                'wait_seconds': wait,
                'prediction': state['prediction'],
                'target_time': state['target_time'],
                'recommended_action': state['recommended_action'],
                'recommended_sensor': state['recommended_sensor'],
            })
        acquired = []
        for sequence in range(min(3, len(raw.columns))):
            state = _acquisition_state(bundle, raw, case_id, 0.0, acquired)
            if state['recommended_action'] != 'ACQUIRE' or not state['recommended_sensor']:
                break
            sensor = state['recommended_sensor']
            acquired.append(sensor)
            after = _acquisition_state(bundle, raw, case_id, 0.0, acquired)
            events.append({
                'kind': 'ACTIVE_ACQUISITION',
                'label': f'{sensor} 취득',
                'sensor': sensor,
                'sequence': sequence + 1,
                'prediction': after['prediction'],
                'target_time': after['target_time'],
                'acquired': list(acquired),
            })
        return {
            'case_id': case_id,
            'scenario': scenario,
            'events': events,
            'active_scope': 'simulated explicit origin-slot pulls; no hardware latency claimed',
        }

    @app.get('/api/pareto')
    def pareto(bundle: Bundle = Depends(require_bundle)):
        return {
            'run_id': bundle.report['run_id'],
            'points': bundle.report.get('acquisition_pareto', []),
            'cost_scope': 'relative train-derived proxy, not currency or measured device latency',
        }

    @app.get('/api/audit')
    def audit(case_id: int = Query(default=0, ge=0),
              scenario: Literal['mixed','outage'] = 'mixed',
              wait_seconds: float = Query(default=0, ge=0),
              bundle: Bundle = Depends(require_bundle)):
        state = _acquisition_state(bundle, app.state.timelines[scenario], case_id,
                                   float(wait_seconds), [])
        top = [row for row in state['candidates'] if row['eligible']][:3]
        return {
            'case_id': case_id,
            'scenario': scenario,
            'decision': state['recommended_action'],
            'sensor': state['recommended_sensor'],
            'top_candidates': top,
            'wait_gain_predicted': state['wait_gain_predicted'],
            'disagreement_proxy': state['disagreement_proxy'],
            'policy_inputs_scope': 'current_snapshot_only',
            'retrospective_fields_used_for_action': False,
            'counterfactual_scope': state['counterfactual_scope'],
        }

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
