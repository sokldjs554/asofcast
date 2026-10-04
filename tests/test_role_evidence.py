from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dataflow_role_evidence_maps_every_posting_area_to_executable_evidence():
    path = ROOT / 'docs' / 'dataflow-role-evidence.md'
    assert path.exists(), 'Dataflow role evidence document is missing'
    text = path.read_text(encoding='utf-8')
    required = [
        '모델 설계·학습', '데이터 전처리', '성능 평가·개선', '추론 최적화', '실험 문서화',
        'Python', 'Git', 'PyTorch', 'MLflow', 'DVC', 'Docker', 'Render', 'PySpark', 'DLinear',
        'src/asofcast/serving_runtime.py', 'src/asofcast/release_gate.py', '.github/workflows/ci.yml',
        'TensorFlow', 'AWS/GCP/Azure', 'local[2]', '다인 협업',
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f'role evidence is missing posting/evidence terms: {missing}'


def test_contributing_documents_reproducible_collaboration_process_without_claiming_team_history():
    path = ROOT / 'CONTRIBUTING.md'
    assert path.exists(), 'CONTRIBUTING.md is missing'
    text = path.read_text(encoding='utf-8')
    required = ['branch', 'Pull Request', 'RED', 'GREEN', 'release-gate', '실험', '다인 협업 경험을 주장하지']
    missing = [item for item in required if item not in text]
    assert not missing, f'collaboration contract is incomplete: {missing}'


def test_serving_and_release_decisions_are_recorded_as_an_architecture_decision():
    path = ROOT / 'docs' / 'adr' / '0001-serving-runtime-and-release-gate.md'
    assert path.exists(), 'serving/release ADR is missing'
    text = path.read_text(encoding='utf-8')
    required = ['PyTorch', 'ONNX Runtime', 'fail-closed', 'release gate', '성능', '테스트 통과']
    missing = [item for item in required if item not in text]
    assert not missing, f'ADR is missing required design rationale: {missing}'


def test_readme_leads_with_engineering_evidence_before_research_limitations():
    text = (ROOT / 'README.md').read_text(encoding='utf-8')
    evidence = text.index('## 한눈에 보는 엔지니어링 근거')
    research = text.index('## 연구 상태와 한계')
    assert evidence < research
    head = text[:research]
    for item in ['PyTorch', 'ONNX Runtime', '51,894,720', 'MLflow', 'DVC', 'DLinear', 'FastAPI']:
        assert item in head
    assert '일반적인 성능 우위 목표는 미달성' not in text[:evidence]