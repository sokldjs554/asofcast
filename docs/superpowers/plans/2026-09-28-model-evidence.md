# Model Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 기존 모델과 개선 후보를 비용·시간 의존성·반복 초기값을 포함해 검증하고 과장 없는 결과를 보관한다.

**Architecture:** 기존 학습/번들은 그대로 사용한다. 별도 연구 모듈이 policy 구간에서 ridge 후보를 학습하고 validation 구간에서만 선택한다. 평가 실행기는 고정 프로토콜의 모든 조건을 실행해 원시 손실과 보수적인 채택 판정을 저장한다.

**Tech Stack:** Python, NumPy, PyTorch, pytest.

**Spec:** ../specs/2026-09-28-model-evidence-design.md

## Global Constraints

- seed=42,43,44; 학습 arrival=811; 평가 mixed/811, mixed/1811, outage/1811.
- ridge=1,10,100,1000; 비용 가중치=0.03; circular block=42, sensitivity=6; bootstrap=2000.
- 최종 평가 결과에 맞춰 설정을 수정하지 않는다. ETTh2 별도 학습/평가이며 zero-shot 주장이 아니다.
- 공개 모델 교체는 채택 기준 전체를 통과할 때만 검토한다.

## Review Focus

- 후보가 없는 사례는 COMMIT하며 NaN counterfactual을 선택하지 않는다.
- validation 선택에 test target이나 test counterfactual이 들어가지 않는다.
- 무작위 기준은 같은 취득 시점·비율을 유지하지만 비용 차이를 숨기지 않는다.
- seed 반복을 독립된 시계열 표본으로 부풀리지 않는다.
- 실패한 조건 하나가 전체 성공 판정에서 빠지지 않는다.

### Task 1: 공정 비교와 개선 후보

**Files:** create `src/asofcast/decision_evidence.py`, `tests/test_decision_evidence.py`.

**Interfaces:** `RidgeGainModel.fit/predict`, `choose_gain_actions`, `action_losses`, `random_matched_losses`, `select_validation`, `paired_block_interval`, `promotion_gate`.

- [ ] 손계산 가능한 비용/선택 예제로 미도착 후보 제외, 무작위 기대 손실, validation 선택, ridge 센서 분리, block 구간, 전체 채택 기준 실패 테스트를 먼저 작성한다.
- [ ] 신규 테스트가 기능 누락으로 실패함을 확인한다.
- [ ] 검증 가능한 최소 구현을 작성하고 신규·전체 테스트를 실행한다.
- [ ] 구현과 테스트를 커밋한다.

### Task 2: 고정 프로토콜 실제 실행

**Files:** create `configs/model_evidence_20260928.json`, `scripts/evaluate_model_evidence.py`.

**Interfaces:** 기존 `run_experiment/load_bundle/build_examples`와 Task 1 계산 함수를 조합한다. 결과 JSON/NPZ는 source/config/protocol hash와 모든 조건을 포함한다.

- [ ] 작은 실제 합성 번들로 평가 경로와 validation 선택의 test target 불변성 테스트를 먼저 작성한다.
- [ ] 기능 누락 실패를 확인하고 실행기를 구현한다.
- [ ] 단위·통합 테스트를 실행한다.
- [ ] 코드·프로토콜을 커밋한 뒤 ETT 전체 실험을 두 번 실행한다. 원시 수치·선택이 같아야 재현 성공이다.

### Task 3: 근거와 완료 상태 정정

**Files:** `README.md`, `docs/model-evidence-20260928.md`, `docs/assets/model-evidence/`.

- [ ] 실제 결과만 표로 보관하고 실패 조건과 추가 미검증 범위를 명시한다.
- [ ] 독립 리뷰와 전체 테스트를 확인한다.
- [ ] 연구 코드와 근거를 GitHub에 반영하고 해당 커밋 CI를 확인한다.
- [ ] 모델 우위 확보 여부를 기능/데모 완료와 구분해 보고한다.
