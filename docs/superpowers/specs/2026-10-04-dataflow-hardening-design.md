# 데이터플로 지원용 AsOfCast 보완 설계

## 목적

데이터플로 AI 모델 개발자 공고의 기술 요구를 README 문구가 아니라 실제 실행 경로와 검증 증거로 보여준다. 기존 AsOfCast의 연구 정직성, 고정된 시간 인과성 계약, 실패한 성능 우위 결과는 바꾸지 않는다.

## 공고와 연결되는 목표

- Python / 머신러닝 / 전처리: 기존 시계열 학습·평가 파이프라인을 유지하고, 새 기능도 동일한 검증 규칙을 따른다.
- PyTorch: 현재 학습·기준 추론 엔진을 유지한다. 공고가 `PyTorch 또는 TensorFlow`를 요구하므로 TensorFlow를 중복 도입하지 않는다.
- 추론 최적화 / 클라우드 운영: 검증용으로만 존재하던 ONNX Runtime을 실제 FastAPI 서빙 선택지로 연결하고 Docker 컨테이너에서 실측 모델로 smoke test한다.
- MLOps: MLflow/DVC 단순 기록에 더해 모델 승격/기각 결정을 해시와 함께 남기는 release gate를 추가한다.
- 대규모 데이터: 기존 51,894,720 measurement-cell PySpark/Parquet 경로를 유지하고 제출 문서에서 실행 증거를 바로 찾을 수 있게 한다.
- Git 협업 / 문제 해결 소통: 기존 PR/실험 이슈 템플릿에 CONTRIBUTING과 의사결정 문서를 추가하고, 변경·실험·한계·재현 명령을 같은 구조로 요구한다.
- 논문 구현·재현: 기존 DLinear 공식 구현 대조 CI를 유지하며 새 기능 때문에 회귀하지 않게 한다.

## 설계 선택

### 1. 실제 서빙용 ONNX Runtime 경로

`asofcast serving-runtime` 명령이 검증된 모델 bundle에서 현재 serving forecaster를 ONNX로 내보낸다. 대표 실측 입력에서 PyTorch와 ONNX 출력의 모양, 유한값, 최대 절대오차를 검사하고 통과한 경우에만 `runtime-manifest.json`을 만든다. manifest에는 원 bundle run id, forecaster 이름, ONNX SHA-256, parity 값, opset, 런타임 버전을 기록한다.

FastAPI는 `ASOFCAST_SERVING_BACKEND=torch|onnx|auto`를 지원한다. `onnx`는 manifest/파일/해시/parity 계약이 맞지 않으면 readiness를 실패시켜 명시적 요청을 조용히 PyTorch로 바꾸지 않는다. `auto`는 검증된 ONNX가 있으면 사용하고 그렇지 않으면 이유를 기록한 채 PyTorch를 사용한다. 정책/센서 가치 모델은 기존 PyTorch 경로를 유지하고, 주 forecaster만 ONNX Runtime으로 교체한다.

### 2. Docker 기반 클라우드 실행 검증

코드 전용 Docker 이미지를 만들고 모델 bundle/runtime은 read-only volume으로 주입한다. GitHub Actions에서 실측 ETTh1 bundle과 검증된 ONNX runtime을 다운로드해 컨테이너를 실행하고 `/health`, `/ready`, `/api/metadata`와 실제 예측 요청을 확인한다. 동일 bundle로 컨테이너를 재시작해 run id와 runtime hash가 유지되는지도 확인한다.

Render 공개 배포는 기존 서비스와 분리해 다룬다. 이 변경만으로 AWS/GCP/Azure 운영 경험을 주장하지 않는다.

### 3. MLOps release gate

`asofcast release-gate`는 이미 계산된 평가 gate를 입력으로 받아 후보 모델의 승격 여부를 결정한다. 입력 gate가 실패하면 `rejected` manifest를 만들고 성공으로 가장하지 않는다. 통과한 경우에만 `promoted` manifest를 만들며 bundle report/manifest와 serving runtime의 SHA-256을 함께 기록한다. 최신 2026-10-02 연구는 전체 목표가 실패했으므로 실제 증거 문서에서는 `rejected`가 정상 결과다.

MLflow 기록은 선택적으로 release manifest를 받아 promotion status, gate source, runtime backend/hash를 태그로 남긴다. DVC는 CI에서 기존 모델/replay와 함께 release manifest도 추적한다.

### 4. 협업·문서·면접 전달

`CONTRIBUTING.md`에 브랜치 → 실험 이슈 → RED/GREEN 테스트 → PR → 검증 증거 → 모델 승격의 흐름을 적는다. 팀 협업을 실제로 수행했다고 과장하지 않고, 저장소가 협업 가능한 규칙을 갖췄다는 증거로만 사용한다.

README 첫 화면은 문제, 핵심 결과, 공고 역량 근거, 데모 순서로 재배치한다. 실패한 일반 성능 우위 결과는 숨기지 않되 첫 문장에서 프로젝트 전체를 실패로 오해하지 않도록 `확인된 성과`와 `미달성 연구 목표`를 분리한다.

## 비목표

- TensorFlow를 PyTorch와 중복 도입하지 않는다.
- AWS/GCP/Azure를 사용하지 않고 사용했다고 주장하지 않는다.
- solo repository를 실제 다인 협업 경험으로 표현하지 않는다.
- 2026-10-02 연구의 전체 성능 목표 실패를 성공으로 바꾸지 않는다.
- ONNX benchmark를 HTTP end-to-end SLA로 표현하지 않는다.

## 완료 기준

1. 기존 200개 테스트가 모두 통과한다.
2. 새 ONNX serving runtime 테스트와 실제 ONNX Runtime parity 검증이 통과한다.
3. Docker image가 빌드되고 실측 ETTh1 + ONNX backend로 API smoke와 재시작 검증을 통과한다.
4. release gate가 pass fixture는 승격, 실제 실패 연구 요약은 기각하며 해시가 재현된다.
5. MLflow/DVC CI 계약이 release manifest를 포함한다.
6. DLinear 공식 구현 재현, Spark 51.9M 처리, 브라우저 회귀를 기존대로 유지한다.
7. README/CONTRIBUTING/직무 증거 문서가 실제 코드·CI와 일치한다.
