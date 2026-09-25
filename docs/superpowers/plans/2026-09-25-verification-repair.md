# AsOfCast 검증 결함 수정 계획

> For agentic workers: execute inline with test-driven development and verify each result.

**Goal:** 감사에서 재현한 추론 비교·실패 처리·화면 설명 결함을 고치고, 기존 모델·데이터 계약을 유지한다.
**Architecture:** 실제 서빙과 같은 inference_mode를 측정에 적용한다. ONNX 평가 함수가 실패 보고서까지 책임지고 예외를 CLI에 전달한다. 실험 비교와 실제 서빙 표시는 분리한다.
**Tech Stack:** Python, PyTorch, NumPy, ONNX Runtime, FastAPI, Chromium, GitHub Actions.
**Spec:** 사용자 승인된 2026-09-25 공고 감사 보고서의 9절 통과 기준.

## Global Constraints
- 기존 M2 정책, 데이터 분할, 학습 결과를 성능 개선으로 바꾸어 주장하지 않는다.
- 실제 런타임 실험과 fault-injection 테스트를 구분한다.
- 새 저장소·라이선스를 추가하지 않는다. 기존 PR #10에서 검증 후 반영한다.
- 원논문 수치와 제한된 공식 구현 대조는 다른 증거다.
- 미검증 원격 CI, 공개 배포, 개인 이력·지원 제출을 완료라고 하지 않는다.

## Review Focus
- NaN/Inf, shape mismatch, runtime exceptions -> failed JSON and nonzero exit.
- Missing dependencies / invalid arguments -> no stale verified report reused.
- Thread count restored and model mode recorded; both engines use NumPy input/output.
- Timing rounds alternate engine order; latency observations do not gate accuracy.
- UI caption cannot imply ONNX production serving.

## Task 1: ONNX evaluator + CLI
Files: src/asofcast/onnx_eval.py, src/asofcast/cli.py, tests/test_onnx_gate.py.
- [x] Add real evaluator fault-injection tests; run `pytest tests/test_onnx_gate.py -q`. Eight failures reproduced.
- [ ] Implement explicit inference mode, paired rounds, configurable threads/warmup, strict finite/shape/drift gate, atomic JSON reports and failure propagation.
- [ ] Run focused tests then `python -m pytest -q`; retain red and green logs.
- [ ] Add actual runtime failure canary and independent CI JSON assertion; run with installed ORT remotely.

## Task 2: UI and documentation truthfulness
Files: src/asofcast/static/index.html, browser_tests/test_m2_flow.py, README.md, docs/verification.md.
- [ ] Test exact serving caption and comparison caption in Chromium; observe old source fail.
- [ ] Change misleading runtime label; bump cache token. Retire old unfair speed comparison; publish only freshly measured conditions.
- [ ] Run Python and browser regressions on synthetic and checksum-verified ETTh1 bundles three times.

## Task 3: DLinear evidence
Files: separate research verification script, tests, workflow and protocol/results document.
- [ ] Inspect pinned official model, data split, experiment script and train loop before defining one bounded reproduction setting.
- [ ] Add tests for train-only normalization, long-horizon window boundaries and official/shared-weight model agreement.
- [ ] Execute official-versus-local ETTh1 long-horizon reference setting with fixed seed; preserve conditions, metrics and differences without claiming every paper table reproduced.

## Task 4: Release evidence
- [ ] Review final diff and source hashes; update the existing PR without overwriting concurrent changes.
- [ ] Verify actual CI jobs and downloaded reports, including failures and skips separately.
- [ ] Merge/deploy only if all required implementation and runtime gates are verified. Otherwise preserve branch and explain exact remaining scope.
