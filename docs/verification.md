# Verification evidence

This file separates measured evidence from claims. Results below are tied to the stated run, dataset and scope.

## AsOfCast M2 remote verification

M2 development evidence was executed on GitHub Actions run
[35315014365](https://github.com/sokldjs554/asofcast/actions/runs/35315014365)
at commit 77c7dee44547e829711edc267a61b9baeded8a35.

The following jobs completed successfully:

- Python 3.11 unit/integration tests and CLI checks
- Python 3.13 unit/integration tests and CLI checks
- checksum-pinned ETTh1 fetch → full M2 train/evaluate → bundle verify
- ONNX Runtime / dynamic INT8 optimization evidence
- MLflow and DVC execution

The branch run intentionally skipped main-only Spark and cloud-smoke jobs. Those are required again after merge to main.

## M2 active sensor acquisition — ETTh1

Measurement values: **ETTh1 real measurements**.

Arrival timestamps: **synthetic** because ETTh1 does not contain transport arrival telemetry.

Test set: **640 cases**.

Acquisition semantics: one-shot explicit reveal of the selected sensor's measurement at the frozen forecast origin. This is not real hardware control.

Default acquisition_cost_weight=0.03:

- immediate MAE: **1.1046016216**
- learned one-shot acquisition MAE: **1.1036567688**
- hindsight one-shot oracle MAE: **0.9221246839**
- acquisition rate: **0.0859375**
- mean relative acquisition cost proxy: **0.0843951628**
- mean realized gain: **0.0009449323**
- mean oracle gain: **0.1824770272**
- regret to oracle: **0.1815320998**
- oracle sensor top-1 hit rate: **0.0330033003**
- candidate gain prediction MAE, standardized: **0.0162407551**
- acquisition model parameters: **2,785**

Interpretation: at the default cost weight, the learned policy improved MAE only slightly over immediate prediction. Its sensor-selection hit rate against the hindsight oracle was low. The project therefore does **not** claim that active acquisition is solved or that the learned selector is generally superior. The oracle gap is retained as concrete evidence of remaining model error.

## M2 cost / accuracy sweep

Same test split, same trained acquisition model; only the decision cost weight changes.

| cost weight | MAE | acquisition rate | mean cost proxy | mean realized gain |
|---:|---:|---:|---:|---:|
| 0.00 | 1.0928599834 | 0.8484375 | 0.8328529596 | 0.0117416661 |
| 0.01 | 1.0983703136 | 0.5734375 | 0.5629733801 | 0.0062313015 |
| 0.03 | 1.1036567688 | 0.0859375 | 0.0843951628 | 0.0009449323 |
| 0.05 | 1.1042292118 | 0.01875 | 0.0184611399 | 0.0003724947 |
| 0.10 | 1.1046016216 | 0.0 | 0.0 | 0.0 |

The cost is a **relative train-derived proxy**, not currency and not measured sensor/network latency.

## Causal acquisition boundary

The implementation enforces:

- passive input: event_time <= origin_time and arrival_time <= decision_time
- active acquisition: only the selected channel at event_time == frozen_origin
- no event after the original origin is exposed by acquisition
- target time is fixed
- future target and future passive arrival times are not acquisition-policy inputs
- retrospective realized gain is returned only as audit evidence

Automated tests cover origin-only reveal, no future-event exposure, invalid acquisition channels, stateless API acquisition, and the retrospective/deployable field separation.

## Passive waiting evidence

ETTh1 passive path:

- learned waiting policy MAE: **1.1281736006**
- learned waiting policy RMSE: **1.5994764463**
- mean wait: **53.4375 seconds**
- matched random-mixture MAE: **1.1326014824**
- full-wait mean error reduction: **0.0455343053**
- fraction helped by full wait: **0.5265625**

The learned-vs-random difference is small; no general superiority claim is made.

## Inference optimization

Shared GitHub Actions CPU runner, batch size 1, model-forward only.

Dynamic INT8:
- FP32 p95: **0.15621445 ms**
- INT8 p95: **0.28238945 ms**
- FP32 MAE: **1.1032959468**
- INT8 MAE: **1.1079432994**
- policy action change rate: **0**
- decision: **keep FP32**

ONNX Runtime 1.30.0:
- Torch p95: **0.23892105 ms**
- ONNX Runtime p95: **0.05798890 ms**
- max standardized output drift: **2.384185791015625e-07**
- dynamic batch parity checked at 1 and 16

These timings are not an HTTP or production SLA measurement. They refer to the forecast serving model, not the M2 acquisition MLP.

## MLOps

The CI path logs the verified ETTh1 bundle to MLflow, initializes DVC in a clean workspace, versions model/replay artifacts with DVC pointers, and uploads evidence artifacts.

This proves the tooling path, not a managed multi-user MLflow deployment.

## Spark / Parquet large-data evidence

Previously verified on main and required again after M2 merge:

- UCI ElectricityLoadDiagrams20112014
- rows: 140,256
- client columns: 370
- measurement cells: **51,894,720**
- Parquet rows after round trip: 140,256
- Spark 3.5.9
- scope: wide CSV parse, numeric cast, null scan, Parquet write/read

## Cloud deployment

Public service: **https://asofcast.onrender.com**

Render:
- region: Singapore
- Python pinned to 3.13.7
- model bundle generated and verified during build
- main workflow checks /health and /ready

After M2 merges, final verification requires the M2 Render deploy to be live, main CI including Spark/cloud smoke to pass, and real Chromium capture of the public AI Decision Console.

## Still not claimed

- real industrial transport telemetry
- real device-control acquisition
- monetary or measured hardware acquisition cost
- optimal sequential active sensing
- general superiority of the learned acquisition selector
- original DLinear paper-score reproduction
- production SLA
- completed rollback drill
