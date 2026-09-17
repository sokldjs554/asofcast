"""Explicit data provenance: generated measurements never masquerade as ETT."""
from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from asofcast.preprocessing import simulate_arrivals
from asofcast.timeline import Timeline

ETT_URL = 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv'
# Upstream Git blob identity observed through GitHub on this implementation turn.
ETT_BLOB_SHA1 = 'a52c4925778c07c1ef1a2cf6fd01594919717d9e'
COLUMNS = ('HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def write_json(path: Path, payload: dict | list) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False,
                                     sort_keys=True) + '\n', encoding='utf-8')


def download_ett(destination: Path) -> dict:
    """Download with an integrity gate; network errors never trigger fake fallback."""
    destination = Path(destination)
    if destination.exists():
        body = destination.read_bytes()
    else:
        try:
            request = urllib.request.Request(ETT_URL, headers={'User-Agent': 'AsOfCast/0.1'})
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read(20_000_001)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError('ETT download failed. Supply the original CSV with --csv; '
                               'no synthetic replacement was created.') from exc
        if len(body) > 20_000_000:
            raise ValueError('unexpectedly large ETT response')
    if git_blob_sha1(body) != ETT_BLOB_SHA1:
        raise ValueError('ETT content differs from the pinned upstream blob; review provenance first')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + '.part')
        temporary.write_bytes(body)
        temporary.replace(destination)
    manifest = {'kind': 'ett', 'url': ETT_URL, 'git_blob_sha1': ETT_BLOB_SHA1,
                'sha256': sha256_file(destination), 'arrival_times': 'synthetic, not measured'}
    write_json(destination.with_suffix('.source.json'), manifest)
    return manifest


def write_synthetic_csv(path: Path, n: int = 7200, seed: int = 21) -> dict:
    """Stable autoregressive sensor fixture, not real electricity measurements."""
    if n < 64:
        raise ValueError('synthetic fixture requires at least 64 rows')
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    rng = np.random.default_rng(seed)
    time = np.arange(n, dtype=float)
    phase = 2 * np.pi * time / 24
    latent = np.zeros(n)
    for i in range(1, n):
        latent[i] = .85 * latent[i - 1] + rng.normal(0, 1)
    loads = []
    for channel in range(6):
        load = (25 + 5 * channel + (7 + channel) * np.sin(phase + channel * .45)
                + 2 * np.sin(phase / 7) + latent * (1 + channel * .2) + rng.normal(0, .8, n))
        loads.append(load)
    values = np.column_stack(loads + [np.zeros(n)])
    values[0, -1] = 35.0
    for i in range(1, n):
        equilibrium = 15 + .28 * values[i, 0] + .12 * values[i, 2] + .1 * values[i, 4]
        values[i, -1] = .73 * values[i - 1, -1] + .27 * equilibrium + rng.normal(0, .55)
    frame = pd.DataFrame(values, columns=COLUMNS)
    frame.insert(0, 'date', pd.date_range('2024-01-01', periods=n, freq='h'))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format='%.8f')
    source = {'kind': 'synthetic', 'generator': 'asofcast-sensor-fixture-v1',
              'generator_seed': seed, 'rows': n, 'sha256': sha256_file(path),
              'warning': 'ALL measurement values are generated. This is not ETT or factory evidence.'}
    write_json(path.with_suffix('.source.json'), source)
    return source


def load_source(path: Path, *, source_kind: str, arrival_seed: int,
                profile: str = 'mixed') -> tuple[Timeline, dict]:
    path = Path(path)
    if source_kind not in ('synthetic', 'ett', 'user-provided-csv'):
        raise ValueError('unsupported source kind')
    content_sha = sha256_file(path)
    if source_kind == 'ett' and git_blob_sha1(path.read_bytes()) != ETT_BLOB_SHA1:
        raise ValueError('ETT provenance check failed: bytes do not match the pinned original')
    source = {'kind': source_kind, 'sha256': content_sha, 'file_name': path.name,
              'arrival_times': 'synthetic, not measured',
              'timestamp_convention': 'dataset-local naive timestamps encoded as ordinal seconds; no timezone inferred'}
    if source_kind == 'synthetic':
        sidecar = path.with_suffix('.source.json')
        if not sidecar.exists():
            raise ValueError('synthetic source requires the generator sidecar')
        generated = json.loads(sidecar.read_text(encoding='utf-8'))
        if generated.get('kind') != 'synthetic' or generated.get('sha256') != content_sha:
            raise ValueError('synthetic sidecar integrity check failed')
        source.update(generated)
    if source_kind == 'ett':
        source.update({'url': ETT_URL, 'git_blob_sha1': ETT_BLOB_SHA1})
    frame = pd.read_csv(path)
    if 'date' not in frame.columns or len(frame.columns) < 2:
        raise ValueError('CSV needs a date column and numeric sensor columns')
    dates = pd.to_datetime(frame.pop('date'), errors='raise')
    if dates.isna().any():
        raise ValueError('missing event timestamps')
    times = dates.astype('int64').to_numpy() // 1_000_000_000
    values = frame.apply(pd.to_numeric, errors='raise').to_numpy(dtype=float)
    arrivals = simulate_arrivals(times, values.shape[1], seed=arrival_seed, profile=profile)
    timeline = Timeline(times, values, arrivals, tuple(frame.columns))
    source.update({'rows': len(frame), 'sensor_columns': list(frame.columns),
                   'measurement_values': int(values.size), 'grid_seconds': timeline.grid_seconds,
                   'arrival_seed': arrival_seed, 'arrival_profile': profile})
    return timeline, source
