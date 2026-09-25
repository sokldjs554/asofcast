> **2026-09-25 정정:** 아래는 2026-09-18 M2의 과거 실행 기록입니다. ONNX 비교의 PyTorch 호출에 autograd 추적 비용이 포함된 결함을 확인했으므로, 해당 속도 비율은 공정한 서빙 개선 성과로 사용하지 않습니다. 출력 parity의 과거 통과와 속도 비교의 결함은 별개입니다. 현재 수정과 새 검증 범위는 [verification-repair.md](verification-repair.md)를 따릅니다. 과거 배포 기록은 이번 수정본 배포 확인이 아닙니다.

# Verification evidence

This file separates measured evidence from claims. Results below are tied to the stated run, dataset and scope.

## Final AsOfCast M2 verification

Final main commit: `35400026774e093de68782ddbb66b7d4b978af77`.

GitHub Actions verification run: [35325196196](https://github.com/sokldjs554/asofcast/actions/runs/35325196196).

The following main-branch jobs completed successfully:

- Python 3.11 unit/integration tests and CLI checks
- Python 3.13 unit/integration tests and CLI checks
- checksum-pinned ETTh1 fetch -> full M2 train/evaluate -> bundle verify
- ONNX Runtime / dynamic INT8 optimization evidence
- MLflow and DVC execution
- UCI ElectricityLoadDiagrams Spark / Parquet large-data path
- public Render `/health` and `/ready` cloud smoke check

Public-demo capture run: [35325196233](https://github.com/sokldjs554/asofcast/actions/runs/35325196233).

- Chromium opened the public Render service, not a mock page.
- Decision Console, Counterfactual Lab, Revision Timeline, Pareto view and full-page screenshots were captured.
- The recording exercised an actual stateless origin-slot sensor pull.
- The default case correctly recommended `WAIT`; the media workflow separately recorded the user's manual highest-ranked eligible candidate exploration as `manual_top_candidate` rather than pretending the model recommended `ACQUIRE`.
- Final media artifact: `asofcast-m2-portfolio-media`, artifact id `10538623331`.

Render deployment:

- public service: **https://asofcast.onrender.com**
- deploy id: `dep-damffsjtqb8s73fs6vgg`
- deployed commit: `35400026774e093de68782ddbb66b7d4b978af77`
- final status: **live**
- region: Singapore
- Python pinned to 3.13.7

## M2 active sensor acquisition - ETTh1

Measurement values: **ETTh1 real measurements**.

Arrival timestamps: **synthetic**, because ETTh1 does not contain transport arrival telemetry.

Test set: **640 cases**.

Acquisition semantics: one-shot explicit reveal of the selected sensor's measurement at the frozen forecast origin. This is not real hardware control.

Default `acquisition_cost_weight=0.03`:

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

Same test split and trained acquisition model; only the decision cost weight changes.

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

- passive input: `event_time <= origin_time` and `arrival_time <= decision_time`
- active acquisition: only the selected channel at `event_time == frozen_origin`
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

## MLOps and large-data evidence

The final main CI path:

- logs the verified ETTh1 bundle to MLflow;
- initializes DVC in a clean workspace and versions model/replay artifacts with DVC pointers;
- processes UCI ElectricityLoadDiagrams20112014 with Spark 3.5.9;
- parses **51,894,720 measurement cells** from 140,256 rows x 370 client columns;
- verifies a 140,256-row Parquet round trip.

This proves the CI/tooling path, not a managed multi-user MLflow deployment.

## Deployment debugging evidence

Two failures were retained and fixed rather than hidden:

1. After the first M2 deploy, the public browser received new HTML with a stale cached M1 JavaScript asset. Render origin logs showed repeated `/api/replay` requests instead of M2 `/api/acquisition`. Static asset URLs were versioned and a regression test was added.
2. After browser capture succeeded, media conversion failed because system `ffmpeg` was not installed. The workflow now checks `command -v ffmpeg` and installs it when missing; a workflow contract test covers the fallback.

The final capture run succeeded through Chromium capture, MP4/GIF conversion, validation, and artifact upload.

## Still not claimed

- real industrial transport telemetry
- real device-control acquisition
- monetary or measured hardware acquisition cost
- optimal sequential active sensing
- general superiority of the learned acquisition selector
- original DLinear paper-score reproduction
- production SLA
- completed rollback drill
