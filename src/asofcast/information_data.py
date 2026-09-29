"""Pinned, research-only weather and chemical-sensor sources."""
from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from asofcast.data import sha256_file
from asofcast.research_data import hourly_complete


def longest_complete_segment(frame, grid_seconds, *, sentinel_min=-999):
    frame = frame.loc[~frame.index.duplicated(keep=False)].sort_index()
    values = frame.to_numpy(dtype=float)
    valid = np.isfinite(values).all(1)
    if sentinel_min is not None:
        valid &= (values > sentinel_min).all(1)
    frame = frame.loc[valid]
    if frame.empty:
        raise ValueError('no complete observations')
    breaks = np.flatnonzero(np.diff(frame.index.asi8) != grid_seconds * 1_000_000_000) + 1
    groups = np.split(np.arange(len(frame)), breaks)
    return frame.iloc[max(groups, key=len)].copy()


def causal_grid_sample(frame, grid_seconds, max_age):
    """Numeric source seconds to a regular grid, with a strict past-only age cap."""
    times = frame.index.to_numpy(dtype=float)
    if not len(times) or not np.isfinite(times).all() or (np.diff(times) < 0).any():
        raise ValueError('monotone finite sensor timestamps required')
    frame = frame.loc[~frame.index.duplicated(keep='first')]
    times = frame.index.to_numpy(dtype=float)
    grid = np.arange(np.ceil(times[0] / grid_seconds) * grid_seconds,
                     np.floor(times[-1] / grid_seconds) * grid_seconds + grid_seconds,
                     grid_seconds)
    indices = np.searchsorted(times, grid, side='right') - 1
    # A nanosecond tolerance handles decimal timestamps exactly at the age cap.
    keep = (indices >= 0) & (grid - times[np.maximum(indices, 0)] <= max_age + 1e-9)
    result = frame.iloc[indices[keep]].copy()
    result.index = grid[keep]
    return result


def prepare_information_source(name, directory, protocol):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    settings = protocol['new_data'][name]
    records = [r for r in protocol['downloads']
               if (r['name'].startswith('jena')) == (name == 'Jena2020')]
    for record in records:
        path = directory / (record['name'] + '.zip')
        if not path.exists():
            partial = path.with_suffix('.partial')
            with urllib.request.urlopen(record['url'], timeout=60) as response, partial.open('wb') as out:
                total = 0
                while block := response.read(1024 * 1024):
                    total += len(block)
                    if total > record['bytes']:
                        raise ValueError('source exceeds frozen size')
                    out.write(block)
            partial.rename(path)
        if path.stat().st_size != record['bytes'] or sha256_file(path) != record['sha256']:
            raise ValueError('source differs from frozen identity')
    if name == 'Jena2020':
        frames = []
        for record in records:
            with zipfile.ZipFile(directory / (record['name'] + '.zip')) as archive:
                frames.append(pd.read_csv(archive.open(record['members'][0]), encoding='latin1'))
        frame = pd.concat(frames, ignore_index=True)
        frame.index = pd.to_datetime(frame.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
        raw_count = len(frame)
        complete = longest_complete_segment(frame[settings['columns']], 600)
        retained_measurements = len(complete)
        frame = hourly_complete(complete)
    else:
        selected = []
        previous, last_grid, raw_count = None, -np.inf, 0
        usecols = [0] + [3 + c for c in settings['sensor_indices']]
        with zipfile.ZipFile(directory / 'gas.zip') as archive:
            with archive.open(settings['member']) as raw:
                for chunk in pd.read_csv(raw, sep=r'\s+', skiprows=1, header=None,
                                         usecols=usecols, chunksize=250000):
                    raw_count += len(chunk)
                    chunk.index = chunk.pop(0)
                    if previous is not None:
                        chunk = pd.concat((previous, chunk))
                    sampled = causal_grid_sample(chunk, settings['grid_seconds'], settings['max_sample_age_seconds'])
                    sampled = sampled.loc[sampled.index > last_grid]
                    if len(sampled):
                        last_grid = float(sampled.index[-1])
                        selected.append(sampled)
                    previous = chunk.loc[~chunk.index.duplicated(keep='first')].iloc[-1:]
        frame = pd.concat(selected, ignore_index=True)
        sampled_times = np.concatenate([part.index.to_numpy() for part in selected])
        frame.index = pd.Timestamp('2000-01-01') + pd.to_timedelta(sampled_times, unit='s')
        frame.columns = [f'S{c}' for c in settings['sensor_indices']]
        frame = longest_complete_segment(frame, settings['grid_seconds'], sentinel_min=None)
        retained_measurements = len(frame)
    values = frame.to_numpy(dtype=float)
    if (not np.isfinite(values).all() or frame.index.has_duplicates or
            not (np.diff(frame.index.asi8) == settings['grid_seconds'] * 1_000_000_000).all()):
        raise ValueError('prepared data must be finite and regular')
    frame.index.name = 'date'
    path = directory / f'{name}.csv'
    frame.to_csv(path, float_format='%.10g')
    return path, dict(name=name, downloads=records, preprocessing=settings,
                     raw_rows=raw_count, retained_measurements=retained_measurements,
                     prepared_rows=len(frame), start=str(frame.index[0]), end=str(frame.index[-1]),
                     prepared_sha256=sha256_file(path), missing_values_imputed=0,
                     test_previously_exposed_before_protocol=False)
