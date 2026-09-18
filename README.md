# AsOfCast

> **늦게 도착하는 센서 데이터에서, 예측값뿐 아니라 “지금 확정할지 더 기다릴지”까지 판단하는 시계열 AI 시스템**

[Live Demo](https://asofcast.onrender.com) · [Verification Evidence](docs/verification.md) · [API](https://asofcast.onrender.com/docs)

일반적인 시계열 예측은 과거 관측이 모두 준비돼 있다고 가정하기 쉽습니다. AsOfCast는 **측정 시각(event time)** 과 **수집 시각(arrival time)** 을 분리하고, 실제 판단 시점까지 도착한 정보만으로 예측합니다. 추가 관측을 기다리더라도 예측 대상 시각은 바꾸지 않습니다.

## 30초 요약

| 질문 | AsOfCast가 한 일 |
|---|---|
| 어떤 문제를 풀었나? | 센서가 늦게 도착하거나 일부 누락되는 상황에서 예측값과 확정 시점을 함께 결정 |
| 어떻게 누출을 막았나? | `event_time <= origin_time` + `arrival_time <= decision_time`을 동시에 만족하는 값만 사용 |
| 어떤 모델을 썼나? | PyTorch DLinear 계열 기준 모델 + arrival/staleness-aware 모델 + learned wait-gain policy |
| 실측 데이터도 돌렸나? | 체크섬이 고정된 **ETTh1 실측값**으로 전체 학습·평가 경로 실행 |
| 추론 최적화는? | FP32/INT8/ONNX Runtime을 실제 측정. INT8은 느려서 기각, ONNX Runtime 경로는 parity와 latency 검증 |
| MLOps는? | GitHub Actions, MLflow, DVC를 실제 실행 |
| 대용량 처리는? | UCI ElectricityLoadDiagrams **51,894,720 measurement cells**를 Spark로 읽고 Parquet round-trip 검증 |
| 서비스 운영은? | FastAPI + Render 공개 배포, main CI에서 `/health`와 `/ready` smoke test |

## 이 프로젝트에서 다른 점

AsOfCast의 핵심은 단순한 “결측치가 있는 시계열 예측”이 아닙니다.

1. **Point-in-time availability** — 예측 시점에 아직 도착하지 않은 과거 값을 미래에서 끌어오지 않습니다.
2. **Frozen target** — 기다리는 동안에도 같은 미래 시각을 계속 예측합니다.
3. **Learned waiting** — 현재 상태에서 추가 관측을 기다릴 가치가 있는지 별도 정책이 추정합니다.
4. **실패한 최적화도 기록** — INT8이 실제 CPU에서 느려져 채택하지 않았고, 그 결과를 숨기지 않았습니다.
5. **실행 증거 중심** — 실측 학습, ONNX, MLflow/DVC, Spark, 클라우드 상태를 CI와 검증 문서로 분리해 남깁니다.

## 구조

```text
event_time / arrival_time
          │
          ▼
 point-in-time snapshot
  ├─ value
  ├─ observed / known
  └─ age / staleness
          │
          ├──────────────► DLinear baseline
          │
          ▼
 arrival-aware forecaster
          │
          ▼
 current prediction
          │
          + current availability state
          ▼
 learned wait-gain policy
          │
          ├─ COMMIT
          └─ WAIT → next allowed decision time
```

정책 입력에는 현재 시점에서 알 수 있는 정보만 들어갑니다. 실제 미래 도착 시각과 미래 정답은 평가·학습 라벨 이외에는 정책 입력으로 사용하지 않습니다.

## 검증 결과

### ETTh1 실측값 + 합성 arrival delay

ETTh1 원본에는 네트워크 도착시각이 없기 때문에 **측정값은 실측, arrival delay는 통제된 합성 조건**입니다.

| 항목 | 결과 |
|---|---:|
| 테스트 사례 | 640 |
| 학습형 대기 정책 MAE | 1.1282 |
| 학습형 대기 정책 RMSE | 1.5995 |
| 평균 대기 | 53.44초 |
| 같은 대기 분포의 무작위 기준 MAE | 1.1326 |
| full-wait가 오차를 줄인 사례 비율 | 52.66% |

학습형 정책과 무작위 기준의 차이는 작습니다. 따라서 **학습 정책의 일반적인 우월성을 주장하지 않습니다.**

### 추론 최적화

공유 GitHub Actions CPU runner, batch=1, model-forward only 조건입니다.

| 경로 | p95 | 결과 |
|---|---:|---|
| PyTorch FP32 | 0.156 ms | 유지 |
| PyTorch dynamic INT8 | 0.282 ms | 더 느려서 기각 |
| PyTorch 비교 실행 | 0.239 ms | ONNX 비교 기준 |
| ONNX Runtime | **0.058 ms** | 최대 표준화 출력 편차 2.38e-7 |

ONNX 수치는 HTTP·네트워크를 포함한 서비스 SLA가 아닙니다.

### 대규모 처리와 MLOps

- UCI ElectricityLoadDiagrams20112014: **140,256 rows × 370 clients = 51,894,720 measurement cells**
- Spark 3.5.9에서 wide CSV parse → numeric cast → null scan → Parquet write/read 실행
- MLflow SQLite tracking store에 ETTh1 실험 기록
- DVC로 모델 `arrival.pt`와 `replay.npz` 아티팩트 포인터 생성
- Python 3.11 / 3.13 CI 테스트
- Render 공개 배포 + main CI cloud smoke test

전체 실행 범위와 한계는 [docs/verification.md](docs/verification.md)에 기록했습니다.

## 데모를 볼 때 주의할 점

**공개 데모의 기본 센서 값과 arrival delay는 합성입니다.** 브라우저 화면은 미리 정한 숫자를 보여주는 정적 포트폴리오가 아니라 저장된 PyTorch 모델을 실제로 호출합니다. 테스트 사례와 지연 상황을 바꾸면 예측값·대기 결정·관측 상태가 다시 계산됩니다.

실측 ETTh1 결과는 데모의 합성 결과와 섞지 않고 CI 검증 기록으로 분리했습니다.

## 실행

Python 3.11–3.13을 사용합니다.

```bash
python -m venv .venv
# Windows: .venv\Scripts\python -m pip install -e ".[dev]"
# macOS/Linux:
.venv/bin/python -m pip install -e ".[dev]"

python -m asofcast generate --out data/synthetic/demo.csv --rows 7200 --seed 21
python -m asofcast run \
  --csv data/synthetic/demo.csv \
  --source-kind synthetic \
  --config configs/synthetic_m1.json \
  --out artifacts/demo

python -m asofcast verify --artifacts artifacts/demo
python -m asofcast serve --artifacts artifacts/demo
```

브라우저에서 `http://127.0.0.1:8000`을 엽니다.

### ETTh1 실험

```bash
python -m asofcast fetch-ett --out data/raw/ETTh1.csv
python -m asofcast run \
  --csv data/raw/ETTh1.csv \
  --source-kind ett \
  --config configs/ett_m1.json \
  --out artifacts/ett-m1
```

`configs/ett_m1.json`은 AsOfCast의 6시간 뒤 한 점 예측 설정입니다. **DLinear 원 논문의 장기 다변량 예측 점수를 재현한 실험으로 부르지 않습니다.**

## API

- `GET /health` — 프로세스 생존 상태
- `GET /ready` — 체크포인트 무결성 검증 및 모델 준비 상태
- `GET /api/metadata` — 모델/데이터 출처/실험 정보
- `GET /api/replay` — 저장된 테스트 사례를 실제 모델로 재계산
- `POST /api/predict` — caller가 제공한 과거 관측과 arrival time으로 온라인 예측
- `GET /docs` — OpenAPI 문서

## 현재 한계

- 실제 산업 네트워크의 arrival telemetry를 사용한 검증은 아닙니다.
- 원 DLinear 장기예측 논문 점수 재현은 M1 범위에 포함하지 않았습니다.
- learned waiting이 모든 조건에서 단순 정책보다 우수하다는 통계적 결론은 없습니다.
- Render 무료 인스턴스의 공개 데모는 상용 SLA·인증·감사로그를 갖춘 production 서비스가 아닙니다.
- rollback drill은 별도 운영 과제로 남겨 두었습니다.

## 저장소

- `src/asofcast` — point-in-time timeline, 전처리, 모델, 정책, 학습, 평가, 서비스
- `tests` — 데이터 누출, API, 체크포인트, 모델 추론, CI 계약 검증
- `configs` — 합성/ETT 실험 설정
- `.github/workflows/ci.yml` — Python, ETTh1, ONNX, MLflow/DVC, Spark, cloud smoke 검증
- `docs/verification.md` — 실행된 결과와 주장하지 않는 범위

## 참고

- Zeng et al., *Are Transformers Effective for Time Series Forecasting?*, AAAI 2023
- Official DLinear reference: https://github.com/cure-lab/LTSF-Linear
- ETT dataset: https://github.com/zhouhaoyi/ETDataset

원본 데이터와 외부 의존성의 권리는 각 제공자에게 있습니다. 이 저장소는 별도의 오픈소스 사용허락을 부여하지 않습니다.
