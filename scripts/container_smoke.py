"""Verify one running AsOfCast container is serving the expected model/runtime."""
from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path


def _get_json(url: str, *, attempts: int = 20, delay_seconds: float = 1.0) -> dict:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=5) as response:  # noqa: S310 - caller controls CI URL
                if response.status != 200:
                    raise RuntimeError(f'{url} returned HTTP {response.status}')
                payload = json.loads(response.read().decode('utf-8'))
                if not isinstance(payload, dict):
                    raise ValueError(f'{url} did not return a JSON object')
                return payload
        except (OSError, ValueError, json.JSONDecodeError, urllib.error.URLError) as exc:
            last = exc
            if attempt + 1 < attempts:
                time.sleep(delay_seconds)
    raise RuntimeError(f'failed to read {url}: {last}')


def validate_responses(*, health: dict, ready: dict, metadata: dict, replay: dict,
                       expected_run_id: str, expected_runtime_sha: str) -> dict:
    if health.get('status') != 'alive':
        raise ValueError('health endpoint is not alive')
    if ready.get('status') != 'ready' or ready.get('checksums_verified') is not True:
        raise ValueError('readiness contract failed')
    if ready.get('serving_backend') != 'onnxruntime':
        raise ValueError('container is not serving through ONNX Runtime')
    if ready.get('run_id') != expected_run_id:
        raise ValueError('container run id does not match expected bundle')
    if ready.get('runtime_sha256') != expected_runtime_sha:
        raise ValueError('container runtime hash does not match expected ONNX model')
    if metadata.get('run_id') != expected_run_id:
        raise ValueError('metadata run id does not match readiness')
    if metadata.get('serving_backend') != 'onnxruntime':
        raise ValueError('metadata serving backend does not match readiness')
    if metadata.get('runtime_sha256') != expected_runtime_sha:
        raise ValueError('metadata runtime hash does not match readiness')
    origin = replay.get('origin_time')
    target = replay.get('target_time')
    steps = replay.get('steps')
    if not isinstance(origin, (int, float)) or not isinstance(target, (int, float)) or target <= origin:
        raise ValueError('replay target must remain after the frozen origin')
    if not isinstance(steps, list) or not steps:
        raise ValueError('replay must contain model-backed steps')
    predictions = [step.get('prediction') for step in steps if isinstance(step, dict)]
    if len(predictions) != len(steps) or any(not isinstance(v, (int, float)) or not math.isfinite(v)
                                              for v in predictions):
        raise ValueError('replay predictions must be finite')
    return {
        'status': 'passed',
        'run_id': expected_run_id,
        'runtime_sha256': expected_runtime_sha,
        'serving_backend': 'onnxruntime',
        'prediction_count': len(predictions),
        'target_after_origin': True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:18080')
    parser.add_argument('--expected-run-id', required=True)
    parser.add_argument('--expected-runtime-sha', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    base = args.base_url.rstrip('/')
    health = _get_json(base + '/health')
    ready = _get_json(base + '/ready')
    metadata = _get_json(base + '/api/metadata')
    replay = _get_json(base + '/api/replay?case_id=0&scenario=mixed')
    result = validate_responses(
        health=health,
        ready=ready,
        metadata=metadata,
        replay=replay,
        expected_run_id=args.expected_run_id,
        expected_runtime_sha=args.expected_runtime_sha,
    )
    report = {**result, 'health': health, 'ready': ready, 'metadata': metadata,
              'replay_summary': {'origin_time': replay['origin_time'], 'target_time': replay['target_time'],
                                 'selected_step': replay.get('selected_step')}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
