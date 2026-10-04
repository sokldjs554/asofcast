# 제출 검토 근거 — 2026-10-05

10월 4일 main `9b8d65db`의 [CI 37209346589](https://github.com/sokldjs554/asofcast/actions/runs/37209346589) 아티팩트 원본을 보존했다. ZIP SHA-256을 GitHub의 아티팩트 digest와 대조했다. ONNX 세 차례 비교와 오류 주입, Docker 재생성, Spark/Parquet, MLflow/DVC 결과를 포함한다. 공개 데모는 PyTorch CPU이며 ONNX Docker 검증과 구분한다.

연구 누락 복구·캡처 출처 수정 후 Python 테스트 227개가 세 차례 통과했다. 각 실행에 Starlette/httpx 사용 중단 예정 경고 1개가 있었다. 로그와 범위는 `verification.json`에 기록했다. 장시간 전체 모델 학습을 새로 실행한 결과로 표현하지 않는다. 원본 연구 자료의 소스·실행 AST·설정 일치와 저장된 원시 예측의 집계 재현을 확인했다. 성능 승격 실패 판단을 유지했다.

## 압축을 풀지 않고 확인할 실행 결과

아래 JSON은 위 10월 4일 CI 원본 ZIP에서 바이트 변경 없이 꺼낸 파일이다. 최신 데모의 HTTP 지연 측정으로 해석하지 않는다. 추출 출처와 해시는 [provenance.json](reports/provenance.json)에 기록했다.

| 확인할 항목 | 원본 결과 | 범위 |
|---|---|---|
| 동일 입력 CPU 추론 | [1회](reports/onnx-1.json) · [2회](reports/onnx-2.json) · [3회](reports/onnx-3.json) | PyTorch inference_mode와 ONNX Runtime의 NumPy 입출력 호출 지연 |
| ONNX 검증·오류 주입 | [runtime-verification.json](reports/runtime-verification.json) | 실제 출력 일치 및 잘못된 출력·런타임의 명령 실패 |
| MLflow 실험 추적 | [실험 run](reports/mlflow-result.json) · [승격 판단 run](reports/mlflow-release-result.json) | 모델 실험과 governance run 분리 |
| 모델 승격 기준 | [release-decision.json](reports/release-decision.json) | 성능 후보 rejected 유지 |
| 대용량 전처리 | [uci-spark-report.json](reports/uci-spark-report.json) | 140,256행 × 370열, 51,894,720개 셀, Spark local[2] |
| Docker 재생성 | [restart-comparison.json](reports/restart-comparison.json) | 모델 실행 ID와 runtime 해시 유지 |

DVC의 추적 파일·상태 로그를 포함한 전체 원본: [MLOps ZIP](github-actions-artifact-11305219092.zip). [추론 ZIP](github-actions-artifact-11305164021.zip) · [Spark ZIP](github-actions-artifact-11305739894.zip) · [컨테이너 ZIP](github-actions-artifact-11305972077.zip).

데모의 후속 보완과 반복 검증은 [최신 제출 검토](../submission-verification-20261005-demo/README.md)에서 확인한다. 위 227개 검사 기록은 이전 통합 이력으로 보존한다.
