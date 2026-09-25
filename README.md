# AsOfCast M2

> **센서가 늦게 도착할 때, 더 읽을지·기다릴지·지금 예측을 확정할지 비교하는 시계열 AI 시스템**

[예측 체험](https://asofcast.onrender.com) · [검증 기록](docs/verification.md) · [API](https://asofcast.onrender.com/docs)

일반적인 forecasting 데모는 “다음 값이 얼마인가?”에서 끝납니다. AsOfCast M2는 늦게 도착하는 센서 환경에서 한 단계 더 나아가 **현재 예측을 개선하려면 어떤 센서를 추가로 취득할 가치가 있는지**를 학습합니다.

현재 정보만으로 세 행동을 비교합니다.

- **ACQUIRE(sensor)** — 선택 센서의 frozen-origin measurement를 명시적으로 추가 취득
- **WAIT** — 다음 허용 판단 시점까지 passive arrival을 기다림
- **COMMIT** — 추가 정보 비용보다 현재 예측을 확정하는 편이 낫다고 판단

> Active acquisition은 실제 장비를 제어한다는 주장이 아닙니다. 공개 데모에서는 선택 센서의 **원래 예측 기준 시각(origin) 값만** 공개하는 통제된 시뮬레이션입니다.

## 30초 요약

| 질문 | 구현 |
|---|---|
| 왜 흔한 forecasting과 다른가? | 값 예측 + **다음으로 읽을 센서 선택** + WAIT/COMMIT을 하나의 정책으로 비교 |
| 정보 누출은? | passive는 event/arrival point-in-time 계약, active pull은 **선택 센서의 origin slot만** 허용 |
| 모델은? | PyTorch forecasting models + wait-gain MLP + **per-sensor AcquisitionValueModel** |
| Counterfactual은? | 각 센서를 받았을 때 예측이 얼마나 바뀌는지 계산. 실제 정답 기반 gain은 audit-only |
| 실측 데이터도 돌렸나? | 체크섬 고정 **ETTh1 측정값**으로 M2 전체 학습·평가 경로 실행 |
| 비용-정확도 trade-off는? | acquisition cost weight별 Pareto evidence를 같은 test split에서 계산 |
| 엔지니어링은? | ONNX Runtime, MLflow, DVC, Spark/Parquet, FastAPI, Render, GitHub Actions |

## 하나의 데모에서 문제부터 결과까지

첫 화면은 **상황 → 선택 → 결과** 순서입니다. 별도 설명서를 먼저 읽지 않아도 예측 대상, 고정된 목표 시각, 아직 없는 센서 수를 확인할 수 있도록 구성했습니다.

1. **상황 확인:** 어떤 센서의 몇 시간 뒤 값을 예측하는지, 기준 시각·현재 판단·목표 시각을 구분합니다. 공개 합성 사례에는 물리 단위를 임의로 붙이지 않습니다.
2. **선택:** 모델 추천을 실행하거나 센서를 직접 읽습니다. 대기 버튼은 실제 시간을 기다리는 대신 다음 허용 시점으로 재생합니다. 확정은 화면 내 선택이며 서버 저장·장비 제어가 아닙니다.
3. **결과 확인:** 실제 선택 전후 예측과 변화량을 비교합니다. 예측값 변화와 정확도 향상을 구분하고, 정답을 이용한 오차 비교는 접힌 사후 평가에서 따로 제공합니다.

모델 추천을 실행한 동작과 사용자의 직접 선택은 다르게 기록합니다. 대기 버튼으로 진행해도 이미 읽은 기준 시각 값과 예측 목표는 유지됩니다. 사례·상황·비교 시작 시점을 직접 바꾸면 새 체험을 시작합니다.

**모델과 검증 근거**를 펼치면 같은 페이지에서 센서별 예상/실제 이득, 상대 비용, 정책 점수, 모델 간 예측 차이, 비용–오차 곡선과 실험 범위를 확인할 수 있습니다. 대기만 했을 때의 가상 비교는 사용자가 실행한 기록과 섞지 않습니다. 사후 정답은 화면의 평가용이며 정책 행동 계산에는 사용하지 않습니다.

## Causal contract

### Passive data

판단 시각 d, frozen forecast origin o에서 모델 입력으로 사용할 수 있는 값은:

~~~text
event_time <= o
arrival_time <= d
~~~

두 조건을 동시에 만족해야 합니다.

### Active acquisition

센서 c를 active acquire하면:

~~~text
reveal = value[event_time = frozen_origin, channel = c]
~~~

만 허용합니다.

- origin 이후 event는 공개하지 않음
- 실제 future arrival time을 정책 feature로 사용하지 않음
- target은 기다리거나 취득해도 고정
- target truth는 offline label / retrospective audit에만 사용

## 모델 구조

~~~text
                   ┌─────────────────────────────┐
event / arrival ──►│ point-in-time snapshot      │
                   │ value · observed · age      │
                   └──────────────┬──────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │ arrival-aware forecaster  │
                    └─────────────┬─────────────┘
                                  │ current prediction
                 ┌────────────────┴────────────────┐
                 │                                 │
       ┌─────────▼──────────┐            ┌─────────▼────────────┐
       │ passive GainPolicy │            │ AcquisitionValueModel│
       │ value of waiting   │            │ value per sensor     │
       └─────────┬──────────┘            └─────────┬────────────┘
                 │                                 │
                 └────────────────┬────────────────┘
                                  ▼
                     joint utility comparison
                       ACQUIRE / WAIT / COMMIT
~~~

AcquisitionValueModel은 작은 MLP이며, 현재 snapshot·현재 forecast·candidate channel identity·train-derived cost proxy만으로 **한 번의 sensor pull이 줄일 것으로 예상되는 absolute error**를 추정합니다.

## ETTh1 M2 검증

개발 브랜치 원격 검증 run **35315014365**에서 체크섬이 고정된 ETTh1을 실제로 내려받아 M2 전체 학습·평가를 실행했습니다.

ETTh1 원본에는 transport arrival timestamp가 없으므로 **측정값은 실측, arrival delay는 synthetic condition**입니다.

### 기본 acquisition cost weight = 0.03

| 항목 | 결과 |
|---|---:|
| test cases | 640 |
| immediate MAE | 1.1046 |
| learned one-shot acquisition MAE | **1.1037** |
| hindsight one-shot oracle MAE | 0.9221 |
| acquisition rate | 8.59% |
| mean relative cost proxy | 0.0844 |
| mean realized gain | 0.0009 |
| oracle mean gain | 0.1825 |
| regret to oracle | 0.1815 |
| oracle sensor top-1 hit | 3.30% |

**해석:** 기본 비용 가중치에서 learned acquisition은 immediate보다 아주 조금만 개선됐고, oracle과의 격차는 큽니다. 따라서 “센서 선택 정책이 해결됐다”거나 “baseline보다 일반적으로 우월하다”고 주장하지 않습니다. 오히려 **현재 value model이 어디까지 잘못 선택하는지까지 정량화**한 것이 M2의 평가 포인트입니다.

### Cost vs Accuracy

| cost weight | MAE | acquisition rate | mean cost proxy |
|---:|---:|---:|---:|
| 0.00 | 1.0929 | 84.84% | 0.8329 |
| 0.01 | 1.0984 | 57.34% | 0.5630 |
| 0.03 | 1.1037 | 8.59% | 0.0844 |
| 0.05 | 1.1042 | 1.88% | 0.0185 |
| 0.10 | 1.1046 | 0% | 0 |

이 표는 “센서를 더 읽으면 무조건 좋다”가 아니라 **추가 정보의 양과 예측 오차 사이의 실제 trade-off**를 보여주기 위한 evidence입니다.

## 기존 wait policy

M2는 M1의 passive wait 경로도 유지합니다.

- learned wait policy MAE: 1.1282
- matched random-mixture MAE: 1.1326
- 평균 wait: 53.44초
- full wait가 error를 줄인 사례: 52.66%

차이가 작으므로 wait policy 역시 일반적인 우월성을 주장하지 않습니다.

## 추론 최적화

**현재 API의 실행 엔진은 PyTorch CPU입니다. ONNX Runtime은 별도 최적화 비교 경로입니다.**

2026-09-25 재검토에서 과거 ONNX 비교의 PyTorch 호출이 실제 서빙과 달리 autograd를 기록한 것을 확인했습니다. 이전 `0.239 ms → 0.058 ms`와 후속 `0.2192 ms → 0.0434 ms`는 동일 조건 비교가 아니므로 현재 성능 개선 수치로 사용하지 않습니다. 과거 기록은 [검증 이력](docs/verification.md)에 남깁니다.

수정한 비교는 두 엔진에 동일한 NumPy float32 입력을 주고 NumPy 출력을 받는 범위를 측정합니다. PyTorch는 `inference_mode`, CPU intra-op 2 threads, ONNX Runtime은 sequential 실행을 사용합니다. 준비 실행 후 엔진 실행 순서를 번갈아 측정하고, 원시 지연 샘플·버전·스레드·출력 편차를 함께 저장합니다. 이는 HTTP 지연이나 Render 운영 SLA가 아닙니다.

출력 모양 불일치·NaN/Inf·표준화 출력 편차 1e-4 초과·런타임 실패는 보고서 `failed` 및 명령 실패로 이어집니다. CI는 실제 ONNX Runtime에서 3회 측정하고, 별도로 표시한 오류 주입 검사 5종과 원시 샘플 재계산을 실행합니다. **새 수치는 검증 결과 원본 확인 후에만 확정합니다.**

기존 INT8 비교는 이미 inference mode였으며 위 결함과 구분합니다. 과거 동일 실행에서 FP32 p95 0.156 ms, dynamic INT8 0.282 ms였기에 INT8은 채택하지 않았습니다. 서로 다른 실행의 기준값은 섞지 않습니다.

## DLinear 공식 구현 대조 실험

M2 자체는 고정된 미래 목표 한 개를 예측하는 문제이며 원논문 점수 재현이 아닙니다. 별도 실험은 공식 구현 커밋을 고정해 ETTh1 다변량 `336 → 96`, 7개 채널, 학습 구간 정규화, 시간순 분할, Adam/학습률 0.005/최대 10 epoch/early stopping 설정으로 비교합니다.

공식 모델과 자체 모델은 초기 가중치를 한 번 맞춘 뒤 같은 CPU 학습 절차에서 **각각 독립 학습**합니다. 전체 테스트 예측과 MSE·MAE를 비교하고, 동일 seed로 2회 반복합니다. 이 실험도 원래 GPU 실행 환경이나 논문 전체 표의 재현을 뜻하지 않습니다. 실행 조건·차이·명령·근거 범위는 [수정 검증 안내](docs/verification-repair.md)에 있습니다.

## 대용량 처리와 MLOps

- **51,894,720 measurement cells** — UCI ElectricityLoadDiagrams20112014
- PySpark: wide CSV parse → numeric cast → null scan → Parquet write/read
- MLflow: verified ETTh1 experiment tracking
- DVC: model/replay artifact pointer
- GitHub Actions: Python 3.11/3.13, ETTh1, ONNX, MLflow/DVC, Spark, cloud smoke
- FastAPI + Render Singapore public deployment
- /health + /ready remote smoke test

## API

- GET /api/acquisition — 현재 sensor ranking + joint action
- POST /api/acquire — stateless origin-slot acquisition 후 재계산
- GET /api/revision-timeline — passive/active prediction revision
- GET /api/pareto — cost/error evidence
- GET /api/audit — 추천 행동의 현재-state 근거
- GET /api/replay — passive replay
- POST /api/predict — caller-provided history inference
- GET /docs — OpenAPI

## 실행

Python 3.11–3.13.

~~~bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"

python -m asofcast generate --out data/synthetic/demo.csv --rows 7200 --seed 21
python -m asofcast run   --csv data/synthetic/demo.csv   --source-kind synthetic   --config configs/synthetic_m1.json   --out artifacts/demo

python -m asofcast verify --artifacts artifacts/demo
python -m asofcast serve --artifacts artifacts/demo
~~~

설정 파일명은 M1과의 재현 호환성을 위해 유지하지만 현재 파일에는 M2 acquisition training 설정도 포함되어 있습니다.

## 주장하지 않는 범위

- 실제 산업 센서 장비에 pull 명령을 보낸 실증이 아닙니다.
- acquisition cost는 돈/실측 장비 latency가 아닌 train-delay 기반 relative proxy입니다.
- ETTh1의 measurement는 real이지만 arrival timestamp는 synthetic입니다.
- learned acquisition의 일반적인 superiority는 입증하지 않았습니다.
- hindsight oracle은 배포 가능한 정책이 아닙니다.
- DLinear 원 논문의 long-horizon score reproduction이 아닙니다.
- 공개 Render demo는 production SLA·인증·감사로그를 갖춘 상용 서비스가 아닙니다.

## 참고

- Zeng et al., *Are Transformers Effective for Time Series Forecasting?*, AAAI 2023
- DLinear reference: https://github.com/cure-lab/LTSF-Linear
- ETT dataset: https://github.com/zhouhaoyi/ETDataset

원본 데이터와 외부 의존성의 권리는 각 제공자에게 있습니다. 이 저장소는 별도의 오픈소스 사용허락을 부여하지 않습니다.
