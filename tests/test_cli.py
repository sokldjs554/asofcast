import os
import subprocess
import sys
from pathlib import Path


def test_cli_help_and_version():
    env = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')}
    result = subprocess.run([sys.executable, '-m', 'asofcast', '--help'], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert 'generate' in result.stdout and 'fetch-ett' in result.stdout and 'serve' in result.stdout


def test_cli_help_lists_optional_verification_commands():
    env = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')}
    result = subprocess.run([sys.executable, '-m', 'asofcast', '--help'], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    for command in ('onnx-eval', 'track-mlflow', 'spark-profile'):
        assert command in result.stdout
