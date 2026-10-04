# ADR 0001: PyTorch 기준 모델, 검증된 ONNX 서빙, 별도 release gate

- 상태: Accepted
- 날짜: 2026-10-04

## Context

AsOfCast는 연구·학습 기준 구현이 PyTorch입니다. 기존에는 ONNX Runtime을 벤치마크 경로에서만 비교했기 때문에 "추론 최적화"와 "실제 서비스가 그 런타임을 사용한다"는 주장이 분리돼 있었습니다. 또한 테스트 통과와 모델 성능 채택 여부를 같은 성공으로 오해할 수 있었습니다.

## Decision

1. **PyTorch를 학습 및 참조 구현으로 유지합니다.** 연구 결과와 기존 bundle 계약을 바꾸지 않습니다.
2. `serving-runtime`이 선택된 forecaster를 ONNX로 내보내고 PyTorch와 다중 batch parity, 입력 shape, bundle run id, SHA-256을 기록합니다.
3. FastAPI는 `torch`, `auto`, `onnx` 서빙 모드를 지원합니다. 명시적 `onnx`는 manifest·hash·parity·bundle identity가 맞지 않으면 **fail-closed**로 readiness를 실패시킵니다. `auto`에서만 PyTorch fallback을 허용하고 이유를 metadata에 노출합니다.
4. 모델 채택은 코드 테스트와 분리한 **release gate**가 결정합니다. 테스트 통과, Docker 기동, ONNX parity가 모두 성공해도 사전에 고정한 성능 기준을 통과하지 못하면 후보는 승격하지 않습니다.
5. Docker CI는 실측 ETTh1 bundle과 검증된 ONNX runtime을 read-only volume으로 주입하고 컨테이너 재시작 전후 `/ready`, `/api/metadata`, `/api/replay`를 검사합니다.

## Consequences

- 서비스 추론 최적화가 단순 벤치마크가 아니라 실행 가능한 경로가 됩니다.
- runtime 파일 변조나 잘못된 모델 결합을 조용히 허용하지 않습니다.
- 코드 품질과 모델 품질을 서로 다른 gate로 설명할 수 있습니다.
- 공개 Render가 어느 backend를 쓰는지는 `/api/metadata`의 `serving_backend`로 확인하며, Docker CI에서 ONNX가 검증됐다는 사실만으로 공개 Render가 ONNX라고 주장하지 않습니다.