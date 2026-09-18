# Active Sensor Acquisition M2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn AsOfCast into a value-of-information forecasting demo that learns which missing origin sensor to acquire, compares acquisition against waiting/commit, and exposes the decision through a richer live UI.

**Architecture:** Add a causal origin-slot acquisition primitive to Timeline, train a small per-sensor acquisition-value regressor from policy-partition counterfactual labels, export/load it in bundle v3, and use it in stateless FastAPI replay endpoints. The UI consumes those endpoints for a Decision Console, sensor map, counterfactual lab, revision timeline, and Pareto panel while preserving M1 endpoints.

**Tech Stack:** Python 3.11–3.13, NumPy, PyTorch, FastAPI, vanilla JS/CSS, GitHub Actions, Render, Playwright.

**Spec:** `docs/superpowers/specs/2026-09-18-active-sensor-acquisition-design.md`

## Global Constraints

- Active acquisition reveals only the selected channel's origin-slot measurement.
- Future targets/future passive arrival times never enter deployable policy features.
- Retrospective realized gains are labeled as audit-only.
- No MIT license or MIT badge.
- Existing M1 API endpoints remain compatible.
- No claim of optimal control or real hardware acquisition.
- Tests must follow red → green for each production change.

---

### Task 1: Causal acquisition snapshot primitive

**Files:**
- Modify: `src/asofcast/timeline.py`
- Test: `tests/test_timeline.py`

**Produces:** `Timeline.snapshot(..., acquired_channels=())` where acquired channels reveal only the origin row.

- [ ] Add tests proving an acquired channel reveals the origin value even when passively late.
- [ ] Add tests proving acquisition cannot reveal future events and rejects invalid channel indices.
- [ ] Run targeted tests and confirm failure.
- [ ] Implement minimal acquisition override in snapshot construction.
- [ ] Run timeline tests and full unit suite.

### Task 2: Acquisition features, labels, model and decision rule

**Files:**
- Create: `src/asofcast/acquisition.py`
- Modify: `src/asofcast/policy.py`
- Test: `tests/test_acquisition.py`

**Produces:**
- `acquisition_features(snapshot_features, prediction, channel_indices, cost_proxy, remaining_fraction)`
- `AcquisitionValueModel`
- `acquisition_gain_targets(...)`
- `rank_acquisition_candidates(...)`
- `select_joint_action(...)`

- [ ] Write tests for causal feature shape and candidate identity.
- [ ] Write tests for gain-label sign using explicit before/after predictions.
- [ ] Write tests that joint action prefers acquire, wait, or commit under controlled gains/costs.
- [ ] Verify red.
- [ ] Implement minimal model/helpers.
- [ ] Verify targeted and full tests.

### Task 3: Train/export acquisition model and evaluation evidence

**Files:**
- Modify: `src/asofcast/experiment.py`
- Modify: `src/asofcast/metrics.py`
- Modify: `configs/synthetic_m1.json`
- Modify: `configs/ett_m1.json`
- Test: `tests/test_experiment.py`

**Produces:**
- train-derived per-channel acquisition-cost proxy;
- trained `acquisition.pt`;
- report `acquisition_policy` and `acquisition_pareto`;
- manifest schema values for the acquisition model.

- [ ] Add failing experiment test asserting acquisition model/report fields exist and are finite.
- [ ] Verify red.
- [ ] Build policy-partition counterfactual acquisition dataset.
- [ ] Train AcquisitionValueModel with deterministic seed.
- [ ] Evaluate one-shot learned selection, one-shot oracle and Pareto weights on validation/test without test-set tuning.
- [ ] Export model and evidence.
- [ ] Verify targeted and full tests.

### Task 4: Bundle v3

**Files:**
- Modify: `src/asofcast/bundle.py`
- Test: `tests/test_bundle.py`

**Produces:** `Bundle.acquisition`, `Bundle.acquisition_cost_proxy` and v3 manifest validation.

- [ ] Add failing load/tamper tests for acquisition weight file/schema.
- [ ] Verify red.
- [ ] Load v3 model strictly while keeping v1/v2 compatibility.
- [ ] Verify bundle and full tests.

### Task 5: Active-acquisition API

**Files:**
- Modify: `src/asofcast/service.py`
- Test: `tests/test_service.py`

**Produces:**
- `GET /api/acquisition`
- `POST /api/acquire`
- `GET /api/revision-timeline`
- `GET /api/pareto`
- `GET /api/audit`

- [ ] Add failing tests for candidate ranking, stateless acquisition transition, invalid sensor rejection and retrospective labels.
- [ ] Add test proving acquired value comes from origin, never future.
- [ ] Verify red.
- [ ] Implement shared acquisition-state helper and endpoints.
- [ ] Verify targeted and full tests.

### Task 6: AI Decision Console frontend

**Files:**
- Replace emphasis in: `src/asofcast/static/index.html`
- Modify: `src/asofcast/static/app.js`
- Modify: `src/asofcast/static/style.css`
- Test: `tests/test_service.py`

**Produces:** First screen visibly centered on Active Sensor Acquisition rather than a generic time-series chart.

- [ ] Add failing HTML contract assertions for “AI Decision Console”, “Counterfactual Sensor Lab”, “Prediction Revision Timeline”, “Cost vs Accuracy”, and acquire controls.
- [ ] Verify red.
- [ ] Implement responsive sensor-card map, recommendation panel, counterfactual table, revision timeline and Pareto SVG.
- [ ] Keep existing evidence and source-provenance warnings.
- [ ] Verify JS syntax, service tests and mobile overflow via browser script.

### Task 7: Documentation, CI evidence, deploy and media

**Files:**
- Modify: `README.md`
- Modify: `docs/verification.md`
- Add/update: `scripts/capture_live_demo.py`, `.github/workflows/capture-demo.yml`
- Update tests: `tests/test_workflow_contract.py`

**Produces:** M2 README positioning, actual CI training evidence, live Render deployment, new real-browser screenshots/video.

- [ ] Update capture selectors/actions to exercise acquisition and counterfactual UI.
- [ ] Run branch CI including ETTh1 M2 experiment.
- [ ] Inspect logs/artifacts and record only measured M2 metrics.
- [ ] Open PR, verify all checks, merge to main.
- [ ] Verify main CI and Render live deploy.
- [ ] Capture public demo with Playwright and download media artifacts.

### Task 8: Dataflow portfolio refresh

**Files outside repo:**
- Generate `Dataflow_AsOfCast_M2_Portfolio.pdf`
- Generate editable `.pptx`
- Generate final `.mp4`, `.gif`, selected `.png`
- Package ZIP

**Produces:** submission-ready portfolio showing M2 active acquisition as the lead story.

- [ ] Replace old generic forecasting screenshots with Decision Console captures.
- [ ] Rewrite project title and problem framing around value-of-information acquisition.
- [ ] Include measured M2 evaluation and limitations.
- [ ] Render and visually inspect all PDF pages.
- [ ] Validate PPTX/MP4/ZIP integrity.
