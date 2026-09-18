from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[1] / '.github' / 'workflows' / 'ci.yml'


def test_ci_exercises_real_data_and_optional_stack():
    text = WORKFLOW.read_text(encoding='utf-8')
    required = ['real-data:', 'fetch-ett', '--source-kind ett', 'optimization:', '.[optimize]', 'mlops:', '.[mlops]', 'large-data:', '.[spark]', 'upload-artifact']
    missing = [item for item in required if item not in text]
    assert not missing, f'CI is missing required verification paths: {missing}'


CAPTURE_WORKFLOW = Path(__file__).resolve().parents[1] / '.github' / 'workflows' / 'capture-demo.yml'
CAPTURE_SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'capture_live_demo.py'


def test_live_demo_capture_workflow_records_portfolio_media():
    assert CAPTURE_WORKFLOW.exists(), 'live demo capture workflow is missing'
    assert CAPTURE_SCRIPT.exists(), 'live demo capture script is missing'
    workflow = CAPTURE_WORKFLOW.read_text(encoding='utf-8')
    script = CAPTURE_SCRIPT.read_text(encoding='utf-8')
    required_workflow = ['playwright install chromium', 'capture_live_demo.py', 'upload-artifact', 'ffmpeg']
    missing_workflow = [item for item in required_workflow if item not in workflow]
    assert not missing_workflow, f'capture workflow is missing: {missing_workflow}'
    required_script = ['https://asofcast.onrender.com', 'screenshot', 'record_video_dir', 'portfolio-hero.png', 'demo-full.png']
    missing_script = [item for item in required_script if item not in script]
    assert not missing_script, f'capture script is missing: {missing_script}'
