# AsOfCast submission completion plan

> Execution: inline, with independent code review before merge.

**Goal:** Audit the existing project against Dataflow's AI model developer role, correct defects, and ship matching demo and application evidence.

**Architecture:** Preserve the production model. Import immutable research summaries from the completed independent evaluation; generate the demo table using the same numerical gate. Bind operational artifacts to exact model contents and reject invalid API inputs where review reproduces failures.

**Spec:** User's 2026-10-05 comprehensive audit request; `../specs/2026-10-04-dataflow-hardening-design.md` for existing service contracts.

**Tech Stack:** Existing Python, PyTorch, FastAPI, ONNX, static HTML/JS, GitHub Actions; no new product dependency.

## Constraints and review focus

- No claim of general learned-policy superiority: latest confirmation failed.
- Preserve prior reports and separate research source from production code.
- Research display must reject altered source bytes, contradictory gate, or stale generated data.
- Same data/config run ID must not authorize different learned weights.
- Invalid numeric request inputs must produce client errors, not server failures.
- Public model, demo source, recorded media and submission prose must have explicit dates/scopes.
- Existing personal facts stay unchanged; company interpretation must link primary sources.

## Task 1 Research display and company relevance

- [ ] Pin latest research summary, protocol, provenance and cost decomposition in `docs/research-20261005/`.
- [ ] Add failing tests for latest display and modified evidence, then implement `build_latest_evidence(summary_bytes)` in `scripts/build_demo_evidence.py`.
- [ ] Update static research copy and browser/capture expectations; retain older results through links.
- [ ] Update README and role evidence with original job URL and official Model Craft connection.
- [ ] Verify targeted tests, generated data check and JS syntax.

## Task 2 Correct reproduced operational defects

- [ ] Review independently reported failures and reproduce with tests before production edits.
- [ ] Correct exact runtime/model identity and request validation where necessary.
- [ ] Run targeted tests and full CI; review diff before merge.

## Task 3 Deliver coherent public project

- [ ] Merge validated correction, verify deployed source and real public behavior.
- [ ] Capture public demo, preserve media metadata, update application documents with verified results and company fit.
- [ ] Render and inspect all document pages, save updated versions and submission ZIP.

## Execution record

- Baseline: local `432aa79`, public main `d8dbd520`, matching production source. Research branches remain separate.
- Official job `rec_idx=54998048` and official Dataflow Model Craft page retrieved 2026-10-05.
- Independent review started; initial finding under investigation: runtime identity may omit model bytes.
- Full tests run on CI; local numerical tests avoid optional ONNX runtime initialization following earlier telemetry approval restriction.
