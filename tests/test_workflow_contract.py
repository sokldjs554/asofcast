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
                 'counterfactual', 'revision', 'pareto', 'm2-decision-console.png']:
        assert item in script
