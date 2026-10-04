# 데이터플로 AI 모델 개발자 공고 대응 근거

이 문서는 제공받은 2026-10-10 마감 데이터플로 **인공지능 AI 모델 개발자** 공고의 주요업무·자격요건·우대사항을 AsOfCast의 실제 코드와 검증 경로에 매핑합니다. 기술 이름만 추가한 항목은 근거로 세지 않습니다.

## 주요업무

| 공고 영역 | AsOfCast 근거 | 검증 방법 |
|---|---|---|
| 모델 설계·학습 | `src/asofcast/models.py`, `training.py`, `experiment.py`, 정보가치 정책 연구 | ETTh1/ETTh2 등 시간순 분리 학습, 다중 seed 평가 |
| 데이터 전처리 | `src/asofcast/data.py`, `timeline.py`, `information_data.py` | 측정 시각과 도착 시각을 분리하고 미래 정보 접근을 차단 |
| 성능 평가·개선 | `src/asofcast/metrics.py`, `research_diagnosis.py`, `release_gate.py` | 강한 단순 baseline, MAE, 비용 포함 손실, bootstrap 구간, 모델 승격/기각 |
| 추론 최적화 | `src/asofcast/serving_runtime.py`, `service.py` | PyTorch 기준 출력과 ONNX Runtime parity 및 SHA-256 검증, Docker smoke |
| 실험 문서화·공유 | `docs/`, GitHub Actions artifact, experiment issue/PR template | 성공뿐 아니라 실패 결과·재현 명령·제약도 보존 |

## 자격요건

주요업무 5개, 자격요건 7개, 우대사항 5개를 합한 17개 항목을 대조했습니다. 경력·학력은 지원 허용 조건이므로 기술 구현 성과로 세지 않습니다.

| 자격요건 항목 | 대응 및 범위 |
|---|---|
| 경력 무관·신입 지원 가능 | 공고의 지원 허용 조건. 지원자의 실제 경력은 이력서에서 확인 |
| 학력 무관 | 공고의 지원 허용 조건. 프로젝트로 학력 사실을 증명하지 않음 |
| Python | 학습·평가·API·전처리의 중심 언어 |
| 머신러닝 기본기 | 시간순 분할, 기준 모델, 정보 누출 방지, 최종 평가 |
| 데이터 분석·전처리 | 시각 분리, 결측·순서 검사, 실제 시계열 처리 |
| Git 기반 협업 역량 | branch·PR·CI·변경 설명을 통한 리뷰 가능성. 개인 프로젝트 범위 |
| 문제 해결·소통 | 실패 원인·수정 이유·재평가·제약을 문서로 공개 |

### Python / 머신러닝 기본기

학습, 평가, FastAPI, CLI, MLOps 및 데이터 파이프라인의 중심 언어는 Python입니다. 시간순 train/policy/validation/test 분리, baseline 비교, seed 반복, 정보 누출 방지, held-out 판정을 코드와 문서에 고정했습니다.

### 데이터 분석·전처리

ETT, Appliances, Tetouan, Jena, Gas 등 서로 다른 시계열을 처리했고, event time과 arrival time을 별도로 모델링했습니다. 결측·중복·시간 순서 오류를 검사하고 판단 시점보다 미래의 데이터를 입력으로 허용하지 않습니다.

### Git 기반 협업

branch → 실험 계획 → RED/GREEN → Pull Request → CI → release-gate 흐름을 `CONTRIBUTING.md`와 GitHub template로 고정했습니다. 이 저장소는 개인 포트폴리오이므로 **다인 협업 경험을 주장하지 않습니다.** 대신 리뷰 가능한 변경 단위와 재현 가능한 증거 계약을 보여줍니다.

### 문제 해결 중심 소통

성능이 좋지 않은 후보도 삭제하지 않고 원인, 수정 가설, 재평가, 기각 이유를 문서화했습니다. 2026-10-02 후속 후보도 일부 조건 개선만으로 승격하지 않고 `docs/research-20261002/release-decision.json`에 rejected로 기록합니다.

## 우대사항

### PyTorch 또는 TensorFlow

AsOfCast는 **PyTorch**를 학습·참조 추론에 사용합니다. 공고가 PyTorch 또는 TensorFlow를 요구하므로 TensorFlow를 기술 수를 늘리기 위해 중복 추가하지 않습니다. 모델 출력 동등성을 확인한 ONNX Runtime 서빙 경로를 별도로 둡니다.

### MLOps

- MLflow: 실험 bundle과 release decision을 별도 run으로 기록
- DVC: 모델/replay/release-decision artifact pointer 보존
- `src/asofcast/release_gate.py`: 테스트 성공과 모델 성능 승격을 분리
- `.github/workflows/ci.yml`: 반복 학습·최적화·MLOps 증거를 artifact로 보존

### 클라우드 환경 운영

FastAPI 공개 데모는 Render에서 동작하고 `/health`, `/ready`를 CI에서 원격 확인합니다. 새 Docker 경로는 ETTh1 bundle과 검증된 ONNX Runtime을 read-only mount하고 재시작 전후 smoke를 실행합니다. **AWS/GCP/Azure 운영을 주장하지 않습니다.** 또한 Docker CI 성공만으로 공개 Render가 ONNX backend라고 표현하지 않습니다.

### 대규모 데이터 처리

`src/asofcast/large_data.py`는 UCI ElectricityLoadDiagrams20112014를 PySpark로 파싱해 numeric cast, null scan, Parquet write/read와 row count를 검증합니다. 전체 대상은 **51,894,720 measurement cells**입니다. Spark master는 `local[2]`이며 실제 다중 노드 cluster 운영 경험으로 표현하지 않습니다.

### 논문 구현·재현

`src/asofcast/reference_eval.py`와 CI의 `reference-reproduction` job은 DLinear 공식 구현을 고정해 로컬 구현과 같은 데이터/초기 조건에서 독립 학습하고 반복 비교합니다. 원 논문의 전체 benchmark를 재현했다고 주장하지 않습니다.

## 면접에서 바로 확인할 실행 경로

```bash
# 전체 회귀
python -m pytest -q

# 실측 ETTh1 학습/검증
python -m asofcast run --csv data/raw/ETTh1.csv --source-kind ett --config configs/ett_m1.json --out artifacts/ett-m1
python -m asofcast verify --artifacts artifacts/ett-m1

# 검증된 ONNX 서빙 runtime
python -m asofcast serving-runtime --artifacts artifacts/ett-m1 --out artifacts/ett-runtime

# 연구 후보 승격/기각
python -m asofcast release-gate --evaluation docs/research-20261002/promotion-summary.json --out release-decision.json
```

## 해석 범위

AsOfCast는 한 프로젝트에서 모델 설계, 데이터 파이프라인, 평가, 최적화, MLOps, API/데모, 대규모 처리, 논문 대조를 연결합니다. 반대로 실제 산업 센서의 arrival log/취득비용, AWS/GCP/Azure 운영, 다중 노드 Spark cluster, 실제 다인 협업 이력을 이 프로젝트가 증명한다고 표현하지 않습니다.
