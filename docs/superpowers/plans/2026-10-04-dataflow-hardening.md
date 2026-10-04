# Dataflow Portfolio Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 데이터플로 AI 모델 개발자 공고의 약한 증거 영역을 실제 ONNX 서빙, Docker 운영, MLOps 승격 gate, Git/문서 계약으로 보완한다.

**Architecture:** 기존 PyTorch 학습·정책은 유지하고 주 forecaster만 검증된 ONNX Runtime으로 선택적으로 서빙한다. 모델 승격은 별도 release manifest로 fail-closed 처리하고 Docker/CI가 실측 bundle에서 전체 경로를 검증한다.

**Tech Stack:** Python 3.11–3.13, PyTorch 2.10, ONNX/ONNX Runtime, FastAPI/Uvicorn, MLflow, DVC, PySpark/Parquet, Docker, GitHub Actions, pytest/Playwright

**Spec:** `docs/superpowers/specs/2026-10-04-dataflow-hardening-design.md`

## Global Constraints

- 기존 시간 인과성, 데이터 분할, 비용, 모델 성능 판정 기준을 변경하지 않는다.
- `PyTorch 또는 TensorFlow` 요구이므로 TensorFlow는 추가하지 않는다.
- ONNX 명시 모드는 fail-closed, auto 모드만 검증 실패 시 PyTorch fallback을 허용한다.
- 팀 협업, 상용 SLA, AWS/GCP/Azure 운영을 수행하지 않은 범위까지 주장하지 않는다.
- 2026-10-02 전체 성능 목표 실패를 그대로 보존한다.

## Review Focus

- runtime manifest의 run id/forecaster/hash가 bundle과 다를 때 ONNX가 로드되지 않아야 한다.
- ONNX 출력 shape/NaN/Inf/parity drift가 readiness 이전에 차단되어야 한다.
- `auto`와 명시 `onnx` 모드의 fallback 정책이 섞이지 않아야 한다.
- release gate가 실패 평가를 성공 manifest로 만들 수 없어야 한다.
- Docker 재시작 뒤 동일 모델/run id/runtime hash가 유지되어야 한다.

---

### Task 1: Verified Serving Runtime

**Files:**
- Create: `src/asofcast/serving_runtime.py`
- Modify: `src/asofcast/cli.py`
- Modify: `src/asofcast/service.py`
- Create: `tests/test_serving_runtime.py`
- Modify: `tests/test_service.py`

**Interfaces:**
- Produces: `prepare_serving_runtime(bundle_dir: Path, out_dir: Path, ...) -> dict`, `load_serving_runtime(bundle, runtime_dir, mode) -> ServingRuntime`
- Consumes: existing `select_serving_forecaster`, bundle checksums and `onnxruntime` optional stack.

- [ ] RED: tests for manifest creation, hash/run-id mismatch, explicit ONNX fail-closed, auto fallback, and service `/ready` backend fields.
- [ ] Verify RED with targeted pytest.
- [ ] GREEN: implement ONNX export/parity manifest and runtime abstraction; route serving forecaster predictions through it.
- [ ] Run targeted tests, then full pytest.
- [ ] Commit.

### Task 2: Release Gate and MLflow/DVC Evidence

**Files:**
- Create: `src/asofcast/release_gate.py`
- Modify: `src/asofcast/cli.py`
- Modify: `src/asofcast/mlops.py`
- Create: `tests/test_release_gate.py`
- Modify: `tests/test_optional_stack.py`
- Create: `docs/research-20261002/promotion-summary.json`

**Interfaces:**
- Produces: `build_release_decision(...) -> dict`, CLI `release-gate`, optional `--release-manifest` for `track-mlflow`.
- Consumes: precomputed evaluation gate; bundle/runtime hashes.

- [ ] RED: pass/reject/tamper tests and MLflow tag contract tests.
- [ ] Verify RED.
- [ ] GREEN: implement fail-closed release manifest and MLflow integration.
- [ ] Run targeted and full pytest.
- [ ] Commit.

### Task 3: Docker Cloud Runtime Verification

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`
- Create: `scripts/container_smoke.py`
- Modify: `.github/workflows/ci.yml`
- Modify: `tests/test_workflow_contract.py`

**Interfaces:**
- Consumes: real ETTh1 bundle + verified serving runtime.
- Produces: container health/readiness/predict/restart evidence artifact.

- [ ] RED: workflow contract test requiring Docker build, ONNX runtime preparation, two container starts and evidence upload.
- [ ] Verify RED.
- [ ] GREEN: add Docker image, smoke script and CI job.
- [ ] Run targeted/full tests and local Docker verification if engine available.
- [ ] Commit.

### Task 4: Collaboration and Interview Evidence

**Files:**
- Create: `CONTRIBUTING.md`
- Create: `docs/dataflow-role-evidence.md`
- Create: `docs/adr/0001-serving-runtime-and-release-gate.md`
- Modify: `README.md`
- Modify: `src/asofcast/static/index.html`
- Modify: `src/asofcast/static/style.css`
- Modify: browser/service tests as required.

**Interfaces:**
- Consumes: verified Task 1–3 behavior and existing Spark/DLinear/MLOps evidence.
- Produces: first-screen job-requirement map with exact evidence links; no unsupported claims.

- [ ] RED: assertions for visible evidence links/copy and no false claims.
- [ ] Verify RED.
- [ ] GREEN: write collaboration protocol, ADR, role evidence, README hero and compact UI evidence band.
- [ ] Run unit + browser tests.
- [ ] Commit.

### Task 5: End-to-End Verification

**Files:**
- Modify only if verification finds a defect.

**Interfaces:**
- Consumes all prior tasks.
- Produces final evidence logs and branch ready for review.

- [ ] Run full pytest at least 3 times.
- [ ] Run Ruff on changed Python and `git diff --check`.
- [ ] Run real ONNX runtime preparation/parity on verified ETTh1 artifact.
- [ ] Build/run Docker with ONNX backend and repeat restart smoke if Docker exists locally; otherwise require CI job result before completion.
- [ ] Run existing browser suite 3 times with real ETTh1 bundle.
- [ ] Verify MLflow + DVC + Spark + DLinear/reference CI contracts remain present.
- [ ] Re-read spec and README claims line-by-line against evidence.
