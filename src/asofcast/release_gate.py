"""Fail-closed model release decisions bound to evaluation and artifact hashes."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from asofcast.bundle import load_bundle, select_serving_forecaster
from asofcast.serving_runtime import _load_manifest

SCHEMA = 'asofcast.release-decision.v1'


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _validate_evaluated_bundle(identity: object, record: dict) -> dict:
    if not isinstance(identity, dict):
        raise ValueError('evaluated bundle identity is required for promotion')
    run_id = identity.get('run_id')
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError('evaluated bundle run_id is invalid')
    for key in ('manifest_sha256', 'report_sha256'):
        if not _is_sha256(identity.get(key)):
            raise ValueError(f'evaluated bundle {key} is invalid')
    fields = ('run_id', 'manifest_sha256', 'report_sha256')
    if any(identity[key] != record.get(key) for key in fields):
        raise ValueError('evaluated bundle does not match the promotion bundle')
    return {key: identity[key] for key in fields}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path, label: str) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'{label} file is missing')
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'invalid {label} JSON: {exc}') from exc
    if not isinstance(value, dict):
        raise ValueError(f'{label} must be a JSON object')
    return value


def _evaluation_contract(path: Path) -> tuple[dict, bool, list[str]]:
    evaluation = _read_json(path, 'evaluation')
    candidate_id = evaluation.get('candidate_id')
    gate = evaluation.get('gate')
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise ValueError('evaluation candidate_id is required')
    if not isinstance(gate, dict) or not isinstance(gate.get('passed'), bool):
        raise ValueError('evaluation gate.passed must be boolean')
    passed = gate['passed']
    reasons = gate.get('reasons', [])
    if not isinstance(reasons, list) or any(not isinstance(item, str) or not item for item in reasons):
        raise ValueError('evaluation gate reasons must be nonempty strings')
    if passed and reasons:
        raise ValueError('passed evaluation cannot contain failure reasons')
    if not passed and not reasons:
        raise ValueError('failed evaluation must contain at least one reason')
    if 'general_goal_achieved' in evaluation:
        goal = evaluation['general_goal_achieved']
        if not isinstance(goal, bool) or goal != passed:
            raise ValueError('evaluation general goal is inconsistent with gate result')
    return evaluation, passed, reasons


def _bundle_record(bundle_dir: Path) -> tuple[dict, object]:
    root = Path(bundle_dir)
    bundle = load_bundle(root)
    _, forecaster_name = select_serving_forecaster(bundle)
    return {
        'run_id': bundle.report['run_id'],
        'source_kind': bundle.report['source']['kind'],
        'forecaster': forecaster_name,
        'manifest_sha256': _sha256(root / 'manifest.json'),
        'report_sha256': _sha256(root / 'report.json'),
    }, bundle


def build_release_decision(evaluation_path: Path, out_path: Path, *, bundle_dir: Path | None = None,
                           runtime_dir: Path | None = None) -> dict:
    """Build an immutable release decision from an already-computed evaluation gate.

    A passed gate must identify the same verified bundle evaluated upstream.
    A failed gate may be recorded without a bundle because it cannot be promoted.
    """
    evaluation_path = Path(evaluation_path)
    out_path = Path(out_path)
    if out_path.exists():
        raise ValueError('release decision output already exists')
    evaluation, passed, reasons = _evaluation_contract(evaluation_path)
    if passed and bundle_dir is None:
        raise ValueError('a verified bundle is required for promotion')
    if runtime_dir is not None and bundle_dir is None:
        raise ValueError('runtime validation requires a verified bundle')

    decision = {
        'schema': SCHEMA,
        'candidate_id': evaluation['candidate_id'],
        'status': 'promoted' if passed else 'rejected',
        'gate_passed': passed,
        'reasons': reasons,
        'evaluation_sha256': _sha256(evaluation_path),
        'evaluation_schema': evaluation.get('schema'),
        'evaluation_scope': evaluation.get('scope'),
        'scope': 'artifact promotion decision; not a claim of universal model superiority',
    }

    bundle = None
    if bundle_dir is not None:
        record, bundle = _bundle_record(bundle_dir)
        if passed:
            decision['evaluated_bundle'] = _validate_evaluated_bundle(
                evaluation.get('evaluated_bundle'), record)
        decision['bundle'] = record
    if runtime_dir is not None:
        assert bundle is not None
        manifest, model_path = _load_manifest(bundle, Path(runtime_dir))
        decision['runtime'] = {
            'backend': 'onnxruntime',
            'onnx_sha256': manifest['onnx_sha256'],
            'runtime_manifest_sha256': _sha256(Path(runtime_dir) / 'runtime-manifest.json'),
            'runtime_model_sha256': _sha256(model_path),
            'max_abs_drift_standardized': manifest['max_abs_drift_standardized'],
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp = out_path.with_name(out_path.name + '.tmp')
    temp.write_text(json.dumps(decision, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(out_path)
    return decision


def validate_release_decision(path: Path) -> dict:
    decision = _read_json(path, 'release decision')
    if decision.get('schema') != SCHEMA:
        raise ValueError('release decision schema mismatch')
    candidate_id = decision.get('candidate_id')
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError('release decision candidate_id is missing')
    passed = decision.get('gate_passed')
    status = decision.get('status')
    if not isinstance(passed, bool) or status not in {'promoted', 'rejected'}:
        raise ValueError('release decision status contract is invalid')
    if (status == 'promoted') != passed:
        raise ValueError('release decision status is inconsistent with gate result')
    reasons = decision.get('reasons')
    if not isinstance(reasons, list) or any(not isinstance(item, str) or not item for item in reasons):
        raise ValueError('release decision reasons are invalid')
    if passed and reasons:
        raise ValueError('promoted release decision cannot contain failure reasons')
    if not passed and not reasons:
        raise ValueError('rejected release decision must contain failure reasons')
    evaluation_hash = decision.get('evaluation_sha256')
    if not _is_sha256(evaluation_hash):
        raise ValueError('release decision evaluation hash is invalid')
    if passed:
        bundle = decision.get('bundle')
        if not isinstance(bundle, dict) or not bundle.get('run_id'):
            raise ValueError('promoted release decision requires a verified bundle record')
        for key in ('manifest_sha256', 'report_sha256'):
            value = bundle.get(key)
            if not _is_sha256(value):
                raise ValueError(f'promoted release decision {key} is invalid')
        _validate_evaluated_bundle(decision.get('evaluated_bundle'), bundle)
    runtime = decision.get('runtime')
    if runtime is not None:
        if not isinstance(runtime, dict) or runtime.get('backend') != 'onnxruntime':
            raise ValueError('release decision runtime contract is invalid')
        for key in ('onnx_sha256', 'runtime_manifest_sha256', 'runtime_model_sha256'):
            value = runtime.get(key)
            if not _is_sha256(value):
                raise ValueError(f'release decision runtime {key} is invalid')
    return decision
