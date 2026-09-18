# AsOfCast M2 — Active Sensor Acquisition Design

## Purpose

Extend AsOfCast from a wait-or-commit forecaster into a value-of-information decision system that decides **which sensor to actively acquire, whether passive waiting is more valuable, or whether to commit now**.

The public demo must make this decision process visible instead of looking like a conventional forecasting dashboard.

## Scope

M2 keeps M1's point-in-time guarantees, ETTh1/synthetic pipelines, DLinear/calibrated forecaster, MLflow/DVC/Spark/ONNX verification, FastAPI service, and Render deployment.

M2 adds:

1. causal active sensor acquisition at the original forecast origin;
2. a learned per-sensor acquisition-value model;
3. action selection across COMMIT / WAIT / ACQUIRE(sensor);
4. retrospective counterfactual evidence separated from deployable model scores;
5. revision timeline and cost-vs-error Pareto evidence;
6. an AI Decision Console UI centered on sensor acquisition.

## Causal contract

At decision time `d` for origin `o`:

- Passive observations still require `event_time <= o` and `arrival_time <= d`.
- An active acquisition may reveal **only the selected channel's measurement at event time `o`**.
- It may not reveal any event after `o`.
- Future target values and future passive arrival times are never policy inputs.
- Counterfactual realized gains may use the target only in offline training/evaluation and replay audit fields.
- Waiting never moves the forecast target time.

## Acquisition semantics

An acquisition action means an explicit pull of one sensor's measurement at the frozen origin. It is not a claim about real hardware control. The demo uses a **relative acquisition-cost proxy**, derived only from training-period arrival-delay statistics.

A channel is eligible when the origin-slot value is not already passively observed at the current decision time and has not already been acquired in the current stateless interaction.

## Model

### Forecast model
Use the existing serving forecaster (staleness-calibrated DLinear when available).

### AcquisitionValueModel
A small MLP predicts the expected one-step absolute-error reduction in standardized target units for each candidate sensor.

Inputs contain only current-state information:
- selected channel's latest value / observed / age / known features;
- selected channel recent availability;
- global current observed fraction;
- current forecast;
- remaining wait fraction;
- one-hot channel identity;
- train-derived cost proxy.

Training label for candidate channel `c`:
`abs(error_current) - abs(error_after_revealing_origin_value_for_c)`.

The target is retrospective; model features are causal.

## Decision rule

At each current state:

1. Existing GainPolicy estimates passive one-step wait gain.
2. AcquisitionValueModel estimates per-channel acquisition gain.
3. Net acquisition utility = predicted acquisition gain - `acquisition_cost_weight * cost_proxy`.
4. Net wait utility = predicted wait gain - existing delay threshold cost.
5. Choose:
   - ACQUIRE(sensor) when best acquisition utility is positive and >= wait utility;
   - WAIT when wait utility is positive and larger;
   - COMMIT otherwise.

This is a learned myopic policy, not an optimal sequential-control proof.

## Evaluation

M2 reports separate metrics:
- immediate forecast MAE;
- one-shot acquisition-policy MAE;
- acquisition rate;
- mean relative acquisition cost;
- mean realized acquisition gain;
- oracle positive gain;
- regret to one-shot oracle;
- selected-sensor top-1 oracle hit rate on eligible positive-gain cases;
- passive wait metrics from M1;
- Pareto points across acquisition cost weights.

No statistical superiority claim is made unless evidence supports it.

## Service API

- `GET /api/acquisition`
  - current decision state;
  - sensor candidates;
  - recommended action;
  - model disagreement proxy;
  - counterfactual retrospective fields for replay only.
- `POST /api/acquire`
  - statelessly add one acquired channel and recompute.
- `GET /api/revision-timeline`
  - forecast revisions across passive waits and recommended active acquisitions.
- `GET /api/pareto`
  - cost/error trade-off evidence from the recorded test report.
- `GET /api/audit`
  - compact explanation of the current decision.

The existing M1 replay/predict endpoints remain compatible.

## UI

Replace the visual emphasis of the old forecast dashboard with an **AI Decision Console**:

1. Decision header: forecast, disagreement proxy, available sensors, recommended action.
2. Sensor acquisition map: each sensor shows state, predicted gain, cost proxy, utility and acquire button.
3. Counterfactual Lab: predicted gain vs retrospective realized gain, clearly labeled.
4. Prediction Revision Timeline: how prediction changes after passive arrivals/acquisitions.
5. Cost vs Accuracy Pareto chart.
6. Existing verification evidence and passive-wait baseline remain below as secondary evidence.

## Deployment and media

After main merge:
- Render rebuild must be live;
- main CI must pass including cloud smoke;
- Playwright must capture the new Decision Console from the public Render URL;
- regenerate PNG captures, MP4/GIF, and the Dataflow portfolio pages so screenshots match the final demo.

## Non-claims

- No real industrial actuator/sensor pull is performed.
- Acquisition cost is a train-derived proxy, not money or real device latency.
- ETTh1 has real measurements but synthetic arrival timestamps.
- M2 is not an optimal active sensing theorem or paper reproduction.
- Retrospective counterfactual realized gains are audit evidence, not deployable inputs.
