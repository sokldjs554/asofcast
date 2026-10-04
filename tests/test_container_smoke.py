import pytest


def valid_payloads():
    return {
        'health': {'status': 'alive'},
        'ready': {
            'status': 'ready', 'run_id': 'run-1', 'checksums_verified': True,
            'serving_backend': 'onnxruntime', 'runtime_sha256': 'a' * 64,
        },
        'metadata': {
            'run_id': 'run-1', 'serving_backend': 'onnxruntime', 'runtime_sha256': 'a' * 64,
        },
        'replay': {
            'origin_time': 100, 'target_time': 106,
            'steps': [{'prediction': 1.25}, {'prediction': 1.5}],
        },
    }


def test_container_smoke_validation_binds_run_and_runtime_hash():
    from scripts.container_smoke import validate_responses
    p = valid_payloads()
    result = validate_responses(**p, expected_run_id='run-1', expected_runtime_sha='a' * 64)
    assert result['status'] == 'passed'
    assert result['run_id'] == 'run-1'
    assert result['runtime_sha256'] == 'a' * 64
    assert result['prediction_count'] == 2


@pytest.mark.parametrize('field,value', [
    ('backend', 'torch'),
    ('run_id', 'wrong'),
    ('sha', 'b' * 64),
])
def test_container_smoke_validation_rejects_wrong_identity(field, value):
    from scripts.container_smoke import validate_responses
    p = valid_payloads()
    if field == 'backend':
        p['ready']['serving_backend'] = value
    elif field == 'run_id':
        p['ready']['run_id'] = value
    else:
        p['ready']['runtime_sha256'] = value
    with pytest.raises(ValueError):
        validate_responses(**p, expected_run_id='run-1', expected_runtime_sha='a' * 64)
