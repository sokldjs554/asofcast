"""Bounded DLinear official-implementation reproduction, separate from M2.

ETTh1, M, 336 -> 96, shared channel weights. Reference and local models train
independently under the same paired CPU harness. This is NOT every paper table
or the unchanged original GPU training program. The differences are reported.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import platform
import os
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from asofcast.data import COLUMNS, download_ett, git_blob_sha1, sha256_file
from asofcast.models import DLinear

REFERENCE_COMMIT = '0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6'
REFERENCE_BLOB = '1cf739ab3de99497d0225610153a05c52609c760'
REFERENCE_URL = f'https://raw.githubusercontent.com/cure-lab/LTSF-Linear/{REFERENCE_COMMIT}/models/DLinear.py'


def prepare_ett(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7 or len(values) < 14400 or not np.isfinite(values).all():
        raise ValueError('ETTh1 requires at least 14400 finite rows and seven measurement columns')
    mean, scale = values[:8640].mean(axis=0), values[:8640].std(axis=0, ddof=0)
    scale = np.where(scale == 0, 1.0, scale)
    data = ((values - mean) / scale).astype(np.float32)
    windows = {}
    for name, start, end in [('train', 0, 8640), ('val', 8640 - 336, 11520),
                              ('test', 11520 - 336, 14400)]:
        view = np.lib.stride_tricks.sliding_window_view(data[start:end], 432, axis=0)
        view = np.moveaxis(view, -1, 1)
        windows[name] = (view[:, :336], view[:, 336:])
    return {'windows': windows, 'mean': mean, 'scale': scale}


def load_official_model(path: Path) -> torch.nn.Module:
    raw = Path(path).read_bytes()
    if git_blob_sha1(raw) != REFERENCE_BLOB:
        raise ValueError('official reference source identity does not match the pinned Git blob')
    spec = importlib.util.spec_from_file_location('asofcast_pinned_dlinear_reference', path)
    if spec is None or spec.loader is None:
        raise ValueError('cannot load pinned official source')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model(SimpleNamespace(seq_len=336, pred_len=96, individual=False, enc_in=7))


def copy_reference_weights(reference, local: DLinear) -> None:
    local.seasonal.load_state_dict(reference.Linear_Seasonal.state_dict())
    local.trend.load_state_dict(reference.Linear_Trend.state_dict())


def _batches(windows, indices):
    x, y = windows
    for ids in indices:
        yield torch.from_numpy(x[ids].copy()), torch.from_numpy(y[ids].copy())


def _fit(model, windows: dict, seed: int) -> tuple[dict, list[dict], int]:
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=.005)
    best, state, stale, best_epoch = float('inf'), None, 0, 0
    history = []
    for epoch in range(1, 11):
        model.train()
        train_ids = torch.randperm(len(windows['train'][0]), generator=generator).numpy()
        train_ids = train_ids[:len(train_ids) // 32 * 32].reshape(-1, 32)
        losses = []
        for x, y in _batches(windows['train'], train_ids):
            optimizer.zero_grad()
            loss = torch.mean((model(x) - y) ** 2)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite training loss')
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        val_ids = torch.randperm(len(windows['val'][0]), generator=generator).numpy()
        val_ids = val_ids[:len(val_ids) // 32 * 32].reshape(-1, 32)
        with torch.no_grad():
            val = float(np.mean([float(torch.mean((model(x) - y) ** 2))
                                for x, y in _batches(windows['val'], val_ids)]))
        if not np.isfinite(val):
            raise ValueError('nonfinite validation loss')
        history.append({'epoch': epoch, 'learning_rate': optimizer.param_groups[0]['lr'],
                        'train_mse': float(np.mean(losses)), 'validation_mse': val})
        if val <= best:
            best, best_epoch, stale = val, epoch, 0
            state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= 3:
            break
        # Official type1 schedule is called AFTER each epoch.
        for group in optimizer.param_groups:
            group['lr'] = .005 * .5 ** (epoch - 1)
    if state is None:
        raise ValueError('no finite validation checkpoint')
    model.load_state_dict(state)
    model.eval()
    return state, history, best_epoch


def _prediction(model, windows) -> tuple[np.ndarray, np.ndarray]:
    x, y = windows
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(x), 32):
            outputs.append(model(torch.from_numpy(x[start:start + 32].copy())).cpu().numpy())
    return np.concatenate(outputs), y.copy()


def evaluate_reference(csv: Path, reference_path: Path, out: Path, seed: int = 2021) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / 'reference-report.json').write_text(
        json.dumps({'status': 'failed', 'gate': {'passed': False},
                    'reason': 'evaluation did not finish; consult execution log'}) + '\n',
        encoding='utf-8')
    provenance = download_ett(csv)  # Existing files also undergo the exact original-blob gate.
    frame = pd.read_csv(csv)
    if tuple(frame.columns) != ('date', *COLUMNS):
        raise ValueError('unexpected ETTh1 column order')
    prepared = prepare_ett(frame.loc[:, list(COLUMNS)].to_numpy())
    if not reference_path.exists():
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(REFERENCE_URL, timeout=30) as response:
            body = response.read(100_001)
        if len(body) > 100_000 or git_blob_sha1(body) != REFERENCE_BLOB:
            raise ValueError('downloaded reference identity mismatch')
        reference_path.write_bytes(body)
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(seed)
        reference = load_official_model(reference_path)
        local = DLinear(336, 7, 96)
        copy_reference_weights(reference, local)
        probe = torch.from_numpy(prepared['windows']['train'][0][:2].copy())
        initial_drift = float((reference(probe) - local(probe)).abs().max().detach())
        training, predictions = {}, {}
        for name, model in [('official', reference), ('asofcast', local)]:
            started = time.monotonic()
            state, history, best_epoch = _fit(model, prepared['windows'], seed)
            pred, truth = _prediction(model, prepared['windows']['test'])
            if not np.isfinite(pred).all():
                raise ValueError('nonfinite test predictions')
            difference = pred.astype(np.float64) - truth.astype(np.float64)
            metrics = {'mse': float(np.mean(difference ** 2)), 'mae': float(np.mean(np.abs(difference)))}
            torch.save(state, out / f'{name}.pt')
            predictions[name] = pred
            training[name] = {'metrics': metrics, 'history': history, 'best_epoch': best_epoch,
                              'seconds': time.monotonic() - started,
                              'checkpoint_sha256': sha256_file(out / f'{name}.pt')}
        drift = float(np.max(np.abs(predictions['official'] - predictions['asofcast'])))
        metric_delta = {metric: abs(training['official']['metrics'][metric] - training['asofcast']['metrics'][metric])
                        for metric in ('mse', 'mae')}
        gate = initial_drift <= 1e-6 and drift <= 1e-4 and max(metric_delta.values()) <= 1e-6
        np.savez_compressed(out / 'test-predictions.npz', truth=truth, **predictions)
        result = {
            'status': 'verified' if gate else 'failed', 'gate': {'passed': bool(gate)},
            'source_commit': os.getenv('GITHUB_SHA'),
            'scope': 'official model versus local implementation in a shared CPU protocol harness',
            'paper_score_reproduced': False, 'reference_commit': REFERENCE_COMMIT,
            'reference_blob': REFERENCE_BLOB, 'reference_sha256': sha256_file(reference_path),
            'dataset': provenance, 'protocol': {'dataset': 'ETTh1', 'features': 'M', 'lookback': 336,
                'horizon': 96, 'channels': 7, 'kernel_size': 25, 'individual': False,
                'train_end': 8640, 'validation_end': 11520, 'test_end': 14400,
                'train_windows': 8209, 'validation_windows': 2785, 'test_windows': 2785,
                'optimizer': 'Adam', 'learning_rate': .005, 'max_epochs': 10, 'patience': 3,
                'batch_size': 32, 'lr_schedule': 'official type1, adjusted after epoch',
                'seed': seed, 'scaler_fit': 'first 8640 rows only, population standard deviation',
                'initialization': 'official random weights copied once, then independently trained',
                'validation': 'shuffle with paired explicit generator; drop_last=True',
                'test': 'all windows, no shuffle/drop, no test-driven checkpoint selection'},
            'deviations_from_original_runner': [
                'CPU, two threads, current PyTorch; not the original GPU/PyTorch environment',
                'explicit paired torch.Generator; not original global DataLoader RNG consumption',
                'NumPy population normalization; no unused time features/decoder inputs',
                'test evaluated only after validation selection, not printed each training epoch'],
            'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                            'numpy': np.__version__, 'platform': platform.platform(), 'threads': 2},
            'initial_output_max_drift': initial_drift, 'test_output_max_drift': drift,
            'metric_abs_delta': metric_delta, 'thresholds': {'initial': 1e-6, 'test_output': 1e-4, 'metrics': 1e-6},
            'results': training,
            'predictions_sha256': sha256_file(out / 'test-predictions.npz'),
        }
        (out / 'reference-report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        if not gate:
            raise ValueError('official/local DLinear comparison exceeded predefined tolerance')
        return result
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_deterministic)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_reference(args.csv, args.reference, args.out)
    print(json.dumps({key: result[key] for key in ('status', 'scope', 'test_output_max_drift', 'metric_abs_delta')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
