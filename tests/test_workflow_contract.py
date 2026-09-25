from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[1] / '.github' / 'workflows' / 'ci.yml'


def test_ci_exercises_real_data_and_optional_stack():
    text = WORKFLOW.read_text(encoding='utf-8')
    required = ['real-data:', 'fetch-ett', '--source-kind ett', 'optimization:', '.[optimize]', 'mlops:', '.[mlops]', 'large-data:', '.[spark]', 'upload-artifact']
    missing = [item for item in required if item not in text]
    assert not missing, f'CI is missing required verification paths: {missing}'


CAPTURE_WORKFLOW = Path(__file__).resolve().parents[1] / '.github' / 'workflows' / 'capture-demo.yml'
CAPTURE_SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'capture_live_demo.py'


def test_m2_capture_workflow_records_active_acquisition_media():
    assert CAPTURE_WORKFLOW.exists(), 'M2 live capture workflow is missing'
    assert CAPTURE_SCRIPT.exists(), 'M2 live capture script is missing'
    workflow = CAPTURE_WORKFLOW.read_text(encoding='utf-8')
    script = CAPTURE_SCRIPT.read_text(encoding='utf-8')
    for item in ['playwright install', 'chromium', 'capture_live_demo.py', 'upload-artifact', 'ffmpeg']:
        assert item in workflow
    assert 'command -v ffmpeg' in workflow
    assert 'apt-get install -y ffmpeg' in workflow
    for item in ['https://asofcast.onrender.com', 'record_video_dir', 'acquireRecommended',
                 'sensor-acquire', 'counterfactual', 'revision', 'pareto',
                 'm2-decision-console.png', 'm2-after-acquisition.png',
                 'acquisition_click_mode']:
        assert item in script


def test_browser_check_entrypoint_targets_current_m2_suite():
    script = (WORKFLOW.parents[2] / 'scripts' / 'check_browser.py').read_text(encoding='utf-8')
    assert 'browser_tests' in script
    assert '--artifacts' in script
    assert "getElementById('sensors')" not in script
    assert '#stepRange' not in script


def test_ci_runs_session_regressions_and_preserves_reports():
    text = WORKFLOW.read_text(encoding='utf-8')
    assert 'browser-consistency:' in text
    assert 'python -m playwright install --with-deps chromium' in text
    assert 'scripts/check_browser.py' in text
    assert '--repeats 3' in text
    assert 'asofcast-browser-evidence' in text
from pathlib import Path


def test_native_onnx_and_reference_checks_are_mandatory_ci_steps():
    workflow = (Path(__file__).parents[1] / '.github/workflows/ci.yml').read_text()
    assert 'python scripts/check_onnx_runtime.py' in workflow
    assert 'python -m asofcast.reference_eval' in workflow
    assert 'dlinear-reference-evidence' in workflow
    assert 'optimization-onnx/' in workflow
