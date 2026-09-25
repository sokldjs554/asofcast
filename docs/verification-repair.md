# 2026-09-25 검증 결함 수정

이 문서는 수정 내용과 실행 계약을 설명한다. CI 성공·배포 완료는 해당 소스의 실제 결과 확인과 구분한다.

## ONNX 검증

- 실제 서빙과 동일한 `torch.inference_mode()`에서 PyTorch forward를 측정한다.
- 두 엔진은 동일한 NumPy float32 입력과 NumPy 출력 경계, CPU intra-op 2 threads를 쓴다. ORT는 sequential, inter-op 1, spinning off이다. PyTorch inter-op 값은 별도 기록하며 같은 값이라고 숨기지 않는다.
- batch 1·16·전체 선택 사례에서 출력 모양, 유한성, 절대 편차 1e-4를 검사한 후에만 속도를 측정한다.
- 준비 실행 30회 후 각 엔진 300회 × 3라운드, 엔진 순서를 번갈아 측정한다. 이를 3회 독립 실행한다.
- 실행 실패 시 `failed` JSON과 nonzero 종료 코드를 남기고, 이전 `verified` 결과가 남지 않게 한다.
- `onnx_contract.py`는 보고서의 상태만 신뢰하지 않고 원시 샘플에서 pooled p50/p95를 재계산한다.
- `scripts/check_onnx_runtime.py`의 오류 주입 5종은 실제 ORT 출력 뒤 테스트 어댑터로 변조하는 **차단 검사**이다. 정확도나 속도 측정으로 세지 않는다.
- `tests/test_onnx_gate.py`는 외부 런타임을 대체한 제어 흐름 검사다. 이것만으로 실제 ONNX 실행을 주장하지 않는다.

```bash
python scripts/check_onnx_runtime.py --artifacts artifacts/ett-m1 --out optimization-onnx
```

현재 API는 PyTorch CPU이고 화면은 ONNX를 비교 실험으로 표시한다. 모델·정책·데이터 분할은 이 수정으로 변경하지 않는다. 이전 속도 수치는 `verification.md`에 정정 표시와 함께 보존한다.

## 제한 범위의 공식 DLinear 대조

공식 저장소 `cure-lab/LTSF-Linear`의 커밋 `0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6`, `models/DLinear.py`의 Git blob `1cf739ab3de99497d0225610153a05c52609c760`을 실행 전 확인한다. 공식 소스를 자체 작성 코드처럼 저장소에 포함하지 않는다. 그 코드의 권리는 원저작자에게 있다.

참조: 공식 `scripts/EXP-LongForecasting/Linear/etth1.sh`, `data_provider/data_loader.py`, `data_provider/data_factory.py`, `run_longExp.py`, `exp/exp_main.py`, `utils/tools.py`의 해당 커밋.

- ETTh1 original CSV의 기존 프로젝트 체크섬 검사, M features, 입력 336, 예측 96, 채널 7, kernel 25, shared weights.
- 학습 0:8640, 검증 8640:11520, 테스트 11520:14400. 검증·테스트는 336개 과거 입력 문맥을 허용하고 미래 target은 섞지 않는다.
- 정규화는 첫 8640개 행의 평균·모집단 표준편차만 사용한다.
- train 8209, validation 2785, test 2785 windows; Adam, learning rate 0.005, max epochs 10, patience 3, batch 32, official type1 schedule.
- 공식 random 초기 가중치를 자체 모델로 한 번 복사한 뒤, 동일한 명시적 shuffle generator(seed 2021)로 **독립 학습**한다. 테스트로 checkpoint를 고르지 않는다.
- 허용 오차는 초기 출력 1e-6, 최종 전체 출력 1e-4, MSE/MAE 차이 1e-6로 실행 전에 고정한다.

```bash
python -m asofcast.reference_eval --csv data/raw/ETTh1.csv \
  --reference data/reference/DLinear.py --out research-reference/run-1
```

**차이와 제한:** CPU 2 threads와 현재 PyTorch, 명시적 generator, NumPy 정규화, 사용하지 않는 시간 feature 생략, epoch별 test 출력 생략이다. 따라서 이는 공식 모델과 자체 구현의 **공유 평가 절차 내 결과 대조**이지 공식 훈련 프로그램을 변경 없이 돌린 결과나 논문 표 수치 재현이 아니다. `paper_score_reproduced=false`를 유지한다. M2의 작은 성능 차이를 이 실험의 결과로 바꿔 설명하지 않는다.

## 지원 문구의 정확한 범위

‘ONNX 서빙 완료’ 대신 ‘PyTorch API 서빙 및 ONNX 비교 검증’을 사용한다. ‘논문 성능 재현’ 대신 실제 실행이 통과한 경우에 한해 ‘공식 DLinear 구현과 동일 조건의 독립 학습 결과 대조’를 사용한다. Spark는 140,256행 × 370열의 측정 셀 51,894,720개이며, 5천만 행이나 다중 서버 운영이 아니다. 개인 PR 기록은 다인 팀 경력과 구분한다.
