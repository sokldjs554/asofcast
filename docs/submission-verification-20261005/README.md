# 제출 검토 근거 — 2026-10-05

10월 4일 main `9b8d65db`의 [CI 37209346589](https://github.com/sokldjs554/asofcast/actions/runs/37209346589) 아티팩트 원본을 보존했다. ZIP SHA-256을 GitHub의 아티팩트 digest와 대조했다. ONNX 세 차례 비교와 오류 주입, Docker 재생성, Spark/Parquet, MLflow/DVC 결과를 포함한다. 공개 데모는 PyTorch CPU이며 ONNX Docker 검증과 구분한다.

연구 누락 복구·캡처 출처 수정 후 Python 테스트 227개가 세 차례 통과했다. 각 실행에 Starlette/httpx 사용 중단 예정 경고 1개가 있었다. 로그와 범위는 `verification.json`에 기록했다. 장시간 전체 모델 학습을 새로 실행한 결과로 표현하지 않는다. 원본 연구 자료의 소스·실행 AST·설정 일치와 저장된 원시 예측의 집계 재현을 확인했다. 성능 승격 실패 판단을 유지했다.
