"""Pinned public measurements for the frozen research protocol."""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from asofcast.data import sha256_file

SOURCES = {
    'ETTh1': ('https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv',
              'f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066'),
    'ETTh2': ('https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv',
              'a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b'),
    'Appliances': ('https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip',
                   '2fccf354445d886e7917620b0195db1f3e3e34d5a067a93b844694a4c561255a'),
    'Tetouan': ('https://archive.ics.uci.edu/static/public/849/power+consumption+of+tetouan+city.zip',
                '3c4bf684161180937043a9fb65701a83d44b740b0d42f0492b6e7aec57ddbbfe'),
}


def canonical_column(name):
    return ' '.join(name.split())


def hourly_complete(frame):
    if (len(frame) < 12 or frame.index.has_duplicates or not frame.index.is_monotonic_increasing
            or not np.isfinite(frame.to_numpy(dtype=float)).all()):
        raise ValueError('regular finite measurements required')
    if not (np.diff(frame.index.asi8) == 600 * 1_000_000_000).all():
        raise ValueError('gap in 10-minute measurement grid')
    grouped = frame.resample('h', closed='right', label='right')
    counts = grouped.size()
    complete = counts == 6
    if not complete.iloc[1:-1].all():
        raise ValueError('interior hourly gap')
    return grouped.mean().loc[complete]


def prepare_source(name, data_dir, protocol):
    url, digest = SOURCES[name]
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    zipped = name in ('Appliances', 'Tetouan')
    raw_path = data_dir / (name + ('.zip' if zipped else '.csv'))
    if not raw_path.exists():
        with urllib.request.urlopen(url, timeout=40) as response:
            body = response.read(30_000_001)
        if len(body) > 30_000_000:
            raise ValueError('source exceeds size limit')
        raw_path.write_bytes(body)
    if sha256_file(raw_path) != digest:
        raise ValueError('source checksum differs from frozen identity')
    path = raw_path
    if zipped:
        with zipfile.ZipFile(raw_path) as archive:
            member = 'energydata_complete.csv' if name == 'Appliances' else 'Tetuan City power consumption.csv'
            frame = pd.read_csv(io.BytesIO(archive.read(member)))
        frame.columns = [canonical_column(name) for name in frame.columns]
        time_column = 'date' if name == 'Appliances' else 'DateTime'
        date_format = '%Y-%m-%d %H:%M:%S' if name == 'Appliances' else '%m/%d/%Y %H:%M'
        frame.index = pd.to_datetime(frame.pop(time_column), format=date_format)
        frame = hourly_complete(frame[protocol['preprocessing'][name]['columns']])
        frame.index.name = 'date'
        path = data_dir / f'{name}-hourly.csv'
        frame.to_csv(path, float_format='%.10g')
    return path, {'name': name, 'url': url, 'download_sha256': digest,
                  'prepared_sha256': sha256_file(path), 'prepared_file': path.name,
                  'test_previously_exposed_before_protocol': not zipped, 'arrival_times': 'simulated',
                  'license': 'CC BY 4.0' if zipped else 'see upstream ETT repository'}
