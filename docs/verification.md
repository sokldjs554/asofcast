# Verification evidence

This document records what was actually executed for AsOfCast. It separates measured evidence from assumptions.

## Remote CI baseline

Verified on GitHub Actions run [35256636440](https://github.com/sokldjs554/asofcast/actions/runs/35256636440)
for commit `3928b7674219defd84c0d6e42c996642ab5a5d4c`.

- Python 3.11: unit/integration tests and CLI checks passed.
- Python 3.13: unit/integration tests and CLI checks passed.
- ETTh1: checksum-pinned source fetched, full AsOfCast M1 training/evaluation completed and bundle verification passed.
- Optimization: dynamic INT8 and ONNX Runtime were measured from the ETTh1 bundle.
- MLOps: MLflow experiment tracking and DVC artifact pointers were executed.
- Large data: UCI ElectricityLoadDiagrams20112014 was downloaded and processed with Spark/Parquet.

Artifacts were uploaded by the workflow with 14-day retention.

## ETTh1 real-measurement experiment

The measurement values are from ETTh1. The original dataset does not provide network arrival timestamps, therefore arrival delay is simulated.

Test set:
- n: 640
- learned waiting policy MAE: 1.1281736006
- learned waiting policy RMSE: 1.5994764463
- mean wait: 53.4375 seconds
- matched random-mixture MAE: 1.1326014824
- full-wait mean error reduction: 0.0455343053
- fraction helped by full wait: 0.5265625

The learned-vs-random difference is small. This experiment does **not** establish general superiority, an optimal stopping policy, or the original DLinear paper score.

## Inference optimization

Measured on a shared GitHub Actions CPU runner, batch size 1, model-forward only.

Dynamic INT8:
- FP32 p95: 0.15621445 ms
- INT8 p95: 0.28238945 ms
- FP32 MAE: 1.1032959468
- INT8 MAE: 1.1079432994
- policy action change rate: 0
- decision: keep FP32

ONNX Runtime 1.30.0:
- Torch p95: 0.23892105 ms
- ONNX Runtime p95: 0.05798890 ms
- maximum standardized output drift: 2.384185791015625e-07
- dynamic batch parity checked at batch sizes 1 and 16

These timings are not an HTTP or production SLA measurement.

## MLOps execution

The CI job:
- created an MLflow SQLite tracking store and logged the verified ETTh1 bundle,
- initialized DVC in a clean workspace,
- versioned `arrival.pt` and `replay.npz`,
- preserved MLflow/DVC evidence as a workflow artifact.

This proves the local/CI tooling path, not a managed multi-user MLflow or remote DVC deployment.

## Spark / Parquet large-data execution

UCI ElectricityLoadDiagrams20112014:
- rows: 140,256
- client columns: 370
- measurement cells parsed: 51,894,720
- Parquet rows after round trip: 140,256
- Spark version: 3.5.9
- runner: local[2]
- measured duration: 117.33 seconds

Scope: full wide-table CSV parse, numeric cast, null scan, Parquet write/read. It does not claim a long-form explode of all measurement cells.

## Still not claimed

- Real industrial transport telemetry: not available; arrival delays are simulated.
- Original DLinear long-horizon paper-score reproduction: not performed in M1.
- Production SLA: not measured.
- Statistical proof that the learned waiting policy is generally better: not established.
- Cloud deployment/rollback evidence: tracked separately and only marked complete after a real deployment check.
