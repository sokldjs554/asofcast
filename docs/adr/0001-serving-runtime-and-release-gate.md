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

### 2026-10-05 최종 검토: 평가 대상과 승격 파일 연결

성능 gate가 통과한 입력은 `evaluated_bundle`에 평가한 bundle의 `run_id`,
`manifest_sha256`, `report_sha256`을 함께 기록해야 합니다. `release-gate`는
실제 파일을 `load_bundle`로 검증한 뒤 이 세 값과 대조합니다. 식별자가 없거나
다르면 승격 파일을 만들지 않습니다. 생성한 판정에도 평가 대상 식별자를 보존하고,
판정을 읽거나 MLflow에 기록할 때 bundle 기록과 다시 대조합니다.

이 식별자는 평가를 수행한 단계에서 기록해야 합니다. 임의의 통과 결과에
나중에 다른 모델의 해시를 붙이는 것은 유효한 성능 평가가 아닙니다.
release gate는 상위 평가기의 수치 판정을 입력으로 받으며, 성능 실험 자체를
재실행하는 도구는 아닙니다. 실패한 평가는 모델 없이 `rejected`로 보존할 수 있습니다.

`track-mlflow` 역시 기록 전에 manifest·체크섬·모델 구조를 `load_bundle`로
검증합니다. 파일이 빠졌거나 손상되면 추적 서버에 run이나 artifact를 만들기 전에 실패합니다.

통과한 평가 입력의 추가 필드는 다음과 같습니다. 값은 평가에 실제 사용한 파일에서
계산한 값이어야 합니다.

```json
{
  "evaluated_bundle": {
    "run_id": "평가한 report.json의 run_id",
    "manifest_sha256": "평가한 manifest.json의 소문자 SHA-256 64자리",
    "report_sha256": "평가한 report.json의 소문자 SHA-256 64자리"
  }
}
```

## Consequences

- 서비스 추론 최적화가 단순 벤치마크가 아니라 실행 가능한 경로가 됩니다.
- runtime 파일 변조나 잘못된 모델 결합을 조용히 허용하지 않습니다.
- 코드 품질과 모델 품질을 서로 다른 gate로 설명할 수 있습니다.
- 공개 Render가 어느 backend를 쓰는지는 `/api/metadata`의 `serving_backend`로 확인하며, Docker CI에서 ONNX가 검증됐다는 사실만으로 공개 Render가 ONNX라고 주장하지 않습니다.
