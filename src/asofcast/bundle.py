"""Verify every exported file before loading weights; never unpickle arbitrary objects."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from asofcast.acquisition import AcquisitionValueModel
from asofcast.calibration import StalenessCalibratedForecaster
from asofcast.data import sha256_file
from asofcast.experiment import RunConfig
from asofcast.models import ArrivalForecaster, BaselineForecaster
from asofcast.policy import GainPolicy
from asofcast.preprocessing import TrainScaler
from asofcast.timeline import Timeline


@dataclass
class Bundle:
    config: dict
    report: dict
    manifest: dict
    scaler: TrainScaler
    timeline: Timeline
    test_origins: np.ndarray
    arrival: ArrivalForecaster
    dlinear: BaselineForecaster
    value_only: ArrivalForecaster
    calibrated: StalenessCalibratedForecaster | None
    policy: GainPolicy
    acquisition: AcquisitionValueModel | None
    acquisition_cost_proxy: np.ndarray | None


def load_bundle(directory: Path) -> Bundle:
    directory = Path(directory).resolve()
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    bundle_version = manifest.get('bundle_version')
    if manifest.get('status') != 'complete' or bundle_version not in (1, 2, 3):
        raise ValueError('incomplete or unsupported model bundle')
    required = {'config.json','scaler.json','report.json','replay.npz','arrival.pt',
                'dlinear.pt','value_only.pt','policy.pt'}
    if bundle_version >= 2:
        required.add('calibrated.pt')
    if bundle_version >= 3:
        required.add('acquisition.pt')
    files = manifest.get('files', {})
    if not required.issubset(files):
        raise ValueError('required bundle files are missing')
    for name, expected in files.items():
        path = directory / name
        if Path(name).name != name or path.is_symlink():
            raise ValueError('unsafe bundle file path')
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f'bundle checksum mismatch: {name}')
    cfg = RunConfig.model_validate(json.loads((directory / 'config.json').read_text(encoding='utf-8'))).model_dump()
    report = json.loads((directory / 'report.json').read_text(encoding='utf-8'))
    scaler = TrainScaler.from_dict(json.loads((directory / 'scaler.json').read_text(encoding='utf-8')))
    report_cfg = RunConfig.model_validate(report['config']).model_dump()
    if report['run_id'] != manifest['run_id'] or report_cfg != cfg:
        raise ValueError('bundle metadata mismatch')
    with np.load(directory / 'replay.npz', allow_pickle=False) as data:
        timeline = Timeline(data['times'], data['values'], data['arrivals'], tuple(data['columns'].tolist()))
        origins = data['test_origins'].copy()
    channels, target = len(timeline.columns), manifest['target_channel']
    if manifest['channels'] != list(timeline.columns) or len(scaler.mean) != channels:
        raise ValueError('bundle channel schema mismatch')
    if not isinstance(target, int) or not 0 <= target < channels or timeline.columns[target] != cfg['target']:
        raise ValueError('bundle target schema mismatch')
    if manifest['policy_features'] != channels * 6 + 2:
        raise ValueError('bundle policy feature schema mismatch')
    if (origins.ndim != 1 or not np.issubdtype(origins.dtype, np.integer) or len(origins) < 1
            or (origins < cfg['lookback'] - 1).any() or (origins + cfg['horizon'] >= len(timeline.times)).any()
            or (np.diff(origins) <= 0).any()):
        raise ValueError('bundle replay origins are invalid')
    models = {'arrival': ArrivalForecaster(cfg['lookback'], channels, target),
              'dlinear': BaselineForecaster(cfg['lookback'], channels, target),
              'value_only': ArrivalForecaster(cfg['lookback'], channels, target, metadata=False),
              'policy': GainPolicy(manifest['policy_features'])}
    calibrated = None
    acquisition = None
    acquisition_cost_proxy = None
    if bundle_version >= 2:
        if manifest.get('calibration_features') != channels * 4 + 1:
            raise ValueError('bundle calibration feature schema mismatch')
        calibrated = StalenessCalibratedForecaster(cfg['lookback'], channels, target)
        models['calibrated'] = calibrated
    if bundle_version >= 3:
        expected_acquisition_features = channels * 7 + 7
        if manifest.get('acquisition_features') != expected_acquisition_features:
            raise ValueError('acquisition feature schema mismatch')
        acquisition_cost_proxy = np.asarray(manifest.get('acquisition_cost_proxy'), dtype=np.float32)
        if (acquisition_cost_proxy.shape != (channels,) or not np.isfinite(acquisition_cost_proxy).all()
                or (acquisition_cost_proxy < 0).any()):
            raise ValueError('acquisition cost proxy schema mismatch')
        acquisition = AcquisitionValueModel(expected_acquisition_features)
        models['acquisition'] = acquisition
    for name, model in models.items():
        model.load_state_dict(torch.load(directory / f'{name}.pt', map_location='cpu', weights_only=True), strict=True)
        model.eval()
    return Bundle(cfg, report, manifest, scaler, timeline, origins,
                  models['arrival'], models['dlinear'], models['value_only'], calibrated,
                  models['policy'], acquisition, acquisition_cost_proxy)


def select_serving_forecaster(bundle: Bundle):
    if bundle.calibrated is not None:
        return bundle.calibrated, 'staleness_calibrated_dlinear'
    return bundle.arrival, 'arrival_aware_mlp'
