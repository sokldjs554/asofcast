from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[1] / '.github' / 'workflows' / 'ci.yml'


def test_ci_exercises_real_data_and_optional_stack():
    text = WORKFLOW.read_text(encoding='utf-8')
    required = ['real-data:', 'fetch-ett', '--source-kind ett', 'optimization:', '.[optimize]', 'mlops:', '.[mlops]', 'large-data:', '.[spark]', 'upload-artifact']
    missing = [item for item in required if item not in text]
    assert not missing, f'CI is missing required verification paths: {missing}'
