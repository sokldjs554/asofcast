# AsOfCast 개발·실험 협업 규칙

AsOfCast는 현재 개인 포트폴리오 저장소입니다. 따라서 이 문서는 **다인 협업 경험을 주장하지 않습니다.** 대신 다른 개발자가 합류해도 같은 실험 계약과 검증 절차를 재현할 수 있도록 Git 기반 작업 규칙을 고정합니다.

## 1. branch 단위로 문제를 분리합니다

- 기능: `feat/<topic>`
- 연구 가설: `research/<topic>-<date>`
- 결함 수정: `fix/<topic>`
- `main`에는 직접 실험 코드를 쌓지 않고 Pull Request 단위로 변경 이유와 증거를 남깁니다.

## 2. 실험 전에 가설과 판정 기준을 적습니다

`.github/ISSUE_TEMPLATE/experiment.md`에 다음을 먼저 기록합니다.

1. 확인할 가설
2. 데이터 출처와 판단 시점에 이용 가능한 정보
3. baseline과 성공/실패 기준
4. 실행 명령과 seed
5. 결과를 보고 나서 바꾸지 않을 비용·채택 기준

성능을 본 뒤 기준을 낮추거나 좋은 조건만 고르는 방식으로 성공을 선언하지 않습니다.

## 3. 구현은 RED → GREEN → 회귀 검증 순서로 진행합니다

- **RED:** 원하는 동작을 표현하는 테스트를 먼저 추가하고 실제 실패를 확인합니다.
- **GREEN:** 가장 작은 구현으로 테스트를 통과시킵니다.
- 관련 테스트 뒤에는 `python -m pytest -q` 전체 회귀를 실행합니다.
- Python은 Ruff, JavaScript는 `node --check`, workflow는 계약 테스트로 검사합니다.

## 4. Pull Request에는 결과뿐 아니라 실패 범위를 씁니다

`.github/pull_request_template.md`의 체크리스트를 사용합니다. 특히 다음을 구분합니다.

- 합성 데이터와 실측 데이터
- 예측 개선과 행동 정책 개선
- 코드/재현 검증 통과와 성능 채택 기준 통과
- 로컬 또는 CI 검증과 실제 클라우드 운영 범위

## 5. 모델 변경은 release-gate를 통과해야 합니다

연구 평가가 끝나면 다음 명령으로 모델 승격 여부를 별도로 기록합니다.

```bash
python -m asofcast release-gate \
  --evaluation docs/research-20261002/promotion-summary.json \
  --out release-decision.json
```

`gate.passed=false`인 후보는 테스트가 모두 통과해도 **rejected**입니다. 배포 후보인 경우 bundle/runtime hash까지 release decision에 묶습니다. MLflow에는 `track-release`로 이 의사결정을 별도 governance run으로 남깁니다.

## 6. CI가 증명하는 범위

`.github/workflows/ci.yml`은 Python 3.11/3.13 테스트, ETTh1 실데이터 학습, ONNX Runtime parity, DLinear 공식 구현 대조, MLflow/DVC, PySpark/Parquet, Docker 기반 ONNX 서빙 재시작 smoke, 브라우저 회귀를 분리해 실행합니다.

CI 성공은 **코드와 실행 계약이 재현됐다는 뜻**입니다. 모델의 일반적인 성능 우위는 별도의 held-out 성능 gate가 판단합니다.