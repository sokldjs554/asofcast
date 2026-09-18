# AsOfCast

**같은 미래 시각을 예측하면서, 데이터를 얼마나 더 기다릴지도 학습합니다.**

측정 시각과 수집 시각을 분리하고, 판단 시점까지 도착한 관측만 모델에 전달합니다.
기준 모델, 도착 상태를 반영하는 모델, 대기 이득을 추정하는 정책을 독립적으로 학습·평가합니다.

> **기본 로컬 데모의 센서 값과 수집 지연은 합성입니다.**
> 별도로 GitHub Actions에서 체크섬이 고정된 ETTh1 실측 데이터를 내려받아 전체 학습·평가 경로를 실행했습니다.
> ETTh1 원본에는 실제 전송 도착시각이 없으므로 arrival time은 통제된 합성 지연입니다. 공장 운영 실증으로 해석하지 않습니다.

## 현재 실행 결과

로컬 기준 자동 테스트 66개를 통과했고, GitHub Actions에서 Python 3.11/3.13 테스트를 모두 통과했습니다.
실측 ETTh1 실험은 640개 테스트 사례에서 학습형 정책 MAE 1.1282, 평균 대기 53.44초를 기록했습니다.
동일 대기분포의 무작위 기준 MAE는 1.1326으로 차이는 작으므로 우월성을 과장하지 않습니다.

CPU 동적 INT8은 p95 0.156ms → 0.282ms로 느려져 FP32를 유지했습니다. 반면 ONNX Runtime은
동일 GitHub runner에서 p95 0.239ms → 0.058ms, 최대 표준화 출력 편차 2.38e-7로 검증됐습니다.
MLflow/DVC 실행과 UCI ElectricityLoadDiagrams 51,894,720개 측정 셀의 Spark/Parquet 처리도 원격 CI에서 완료했습니다.
세부 실행 증거와 한계는 [docs/verification.md](docs/verification.md)에 기록합니다.

화면은 실제 ASGI 모델 응답과 연결해 확인했습니다. 공개 데모는 **https://asofcast.onrender.com** 에 배포했습니다. Render Singapore 리전의 무료 인스턴스이며, 최초 배포에서 기본 Python 3.14가 프로젝트 지원 범위를 벗어나 실패한 뒤 `PYTHON_VERSION=3.13.7`로 고정해 재배포했습니다. 현재 서비스는 live 상태이며, main 브랜치 CI에서 `/health`와 `/ready`를 반복 확인하는 cloud smoke test를 추가했습니다.

## 실행

Python 3.11–3.13을 사용합니다. 검증 런타임은 Python 3.13.5 / PyTorch 2.10.0 CPU입니다.
압축본에는 `artifacts/demo`에 합성 데이터로 학습한 모델과 실행 기록이 포함되어 있습니다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m asofcast verify --artifacts artifacts/demo
.\.venv\Scripts\python.exe -m asofcast serve --artifacts artifacts/demo
```

브라우저에서 `http://127.0.0.1:8000`을 엽니다. macOS/Linux는 위 실행 파일 경로 대신
`.venv/bin/python`을 사용합니다. Windows 명령은 제공하지만 이 Linux 환경에서 Windows 실행을 검증한 것은 아닙니다.
깨끗한 Ubuntu GitHub Actions 환경에서 Python 3.11/3.13 의존성 설치와 테스트를 별도로 확인했습니다.

## 새 실험

합성 검증을 명시적으로 생성합니다. 다운로드 실패 시 합성 데이터로 몰래 대체하지 않습니다.

```bash
python -m asofcast generate --out data/synthetic/demo.csv --rows 7200 --seed 21
python -m asofcast run --csv data/synthetic/demo.csv --source-kind synthetic --config configs/synthetic_m1.json --out artifacts/new-run
python -m asofcast benchmark --artifacts artifacts/new-run --out artifacts/runtime_benchmark.json
```

출력 폴더가 이미 있으면 학습 결과를 덮어쓰지 않고 중단합니다.

실측 ETT 실험은 별도 명령입니다. 다운로드한 원본은 확인된 Git blob과 일치해야 합니다.

```bash
python -m asofcast fetch-ett --out data/raw/ETTh1.csv
python -m asofcast run --csv data/raw/ETTh1.csv --source-kind ett --config configs/ett_m1.json --out artifacts/ett-m1
```

`configs/ett_m1.json`은 AsOfCast의 **6시간 뒤 한 점 예측** 설정입니다.
원 논문의 장기 다변량 예측 설정과 달라 논문 성능 수치 재현으로 부르지 않습니다.

## 시간 계약

`event_time <= origin_time` 및 `arrival_time <= decision_time`을 동시에 만족하는 값만 사용합니다.
기다려도 `target_time = origin_time + horizon`은 고정됩니다. 결측 접두부를 뒤에서 채우지 않습니다.
정규화는 학습 마감까지 도착한 관측에만 적합하고, 학습/정책학습/검증/테스트의 입력·정답 구간을 분리합니다.
학습·정책학습의 정답도 해당 분할 마감까지 도착해야 합니다.

예측기에는 값, 최신 슬롯 도착 여부, 데이터 나이, 사용 가능한 값의 존재 여부를 입력합니다.
정책에는 **현재** 관측과 예측만 들어갑니다. 미래 예측·실제 미래 도착시각·정답은 입력하지 않습니다.
대기정책은 한 시점 뒤의 예상 오차 감소를 학습한 기준 정책입니다. 최적 순차정책을 증명한 것이 아닙니다.

## 비교와 검증

DLinear 방식의 분해·선형 모델, 같은 크기의 값 전용 모델, 도착 상태 포함 모델을 비교합니다.
즉시/고정 대기/마감까지 대기/학습 정책을 평가하고, 같은 대기 분포를 가진 무작위 비교의 정확한 기대값을 계산합니다.
정책 기준은 검증 구간에서 선택합니다. 테스트 구간과 지연 증가 시나리오에서 다시 조정하지 않습니다.

FP32와 CPU 동적 INT8은 실제 순전파 시간을 측정합니다. 예측 오차와 정책 선택 변경률도 함께 기록하고,
작은 모델에서 양자화가 더 느리면 FP32를 유지합니다. ONNX Runtime 경로는 원격 CI에서 출력 일치성과
실제 순전파 시간을 검증했습니다. 공개 클라우드 운영은 별도 검증 항목으로 관리합니다.

```bash
python -m pytest -q
python -m asofcast verify --artifacts artifacts/demo
```

## API

`GET /health`는 프로세스 생존, `GET /ready`는 체크포인트 무결성 검증 여부를 나타냅니다.
`GET /api/replay`는 테스트 사례를 모델로 다시 계산합니다. 사후 정답은 별도 감사용 필드로만 반환합니다.
`POST /api/predict`는 과거 측정값과 수집 시각만 받아 미래를 예측하며 미래 정답을 요구하지 않습니다.
`GET /docs`에서 정확한 요청 스키마를 확인할 수 있습니다.

로컬 실행 외에 Render 공개 데모도 제공합니다. 다만 인증·운영용 입력 제한·감사 로그·실제 통신 지연의 SLA 검증을
완료한 상용 운영 서비스라는 의미는 아닙니다.

## 저장소 구성

- `src/asofcast`: 시간 계약, 데이터 출처, 모델, 학습, 정책, 평가, 서비스.
- `tests`: 정보 누출, 학습, 체크포인트 변조, 온라인 추론, 실제 모델 호출 테스트.
- `configs`: 실제 사용한 실험 설정.
- `.github/workflows`: 실측 ETT, ONNX, MLflow/DVC, Spark 검증 경로.

기존 프로젝트의 소스는 복사하지 않았습니다. 이 저장소는 새 구현입니다.

## 참고 자료

DLinear 알고리즘: Zeng et al., *Are Transformers Effective for Time Series Forecasting?*, AAAI 2023.
https://arxiv.org/abs/2205.13504

공식 참조 구현: https://github.com/cure-lab/LTSF-Linear

ETT 데이터 및 설명: https://github.com/zhouhaoyi/ETDataset

원본 데이터와 외부 의존성의 권리는 각 제공자에게 있습니다. 이 프로젝트에 별도 오픈소스 사용허락을 부여하지 않습니다.
