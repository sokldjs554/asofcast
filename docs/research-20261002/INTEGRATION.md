# 2026년 10월 4일 연구 자료 통합

README에 존재하지 않던 결과 보고서, 고정 프로토콜, 실행기를 2026-10-02 보존 연구 묶음에서 복구했다. 당시 전체 결과는 실패 판정이며 공개 모델로 승격하지 않는다.

## 복구 범위

- `posterior_risk.py`, `posterior_policy.py`, 연구 실행 스크립트 4개, 프로토콜 2개, 테스트 5개를 원본을 가져온 뒤 Ruff 형식·import 정렬과 미사용 hashlib import 하나를 정리했다. import를 제외한 실행 AST와 프로토콜 내용이 원본과 동일한지 확인한 기록은 [integration-source-verification.json](integration-source-verification.json)에 있다. 원본 SHA-256은 [import-source-hashes.json](import-source-hashes.json)에 있다.
- [RESULTS_KO.md](RESULTS_KO.md)는 당시 보고서다. 상단 안내 외 본문은 보존했다. 200개 테스트, 재학습 3회, 당시 환경과 공개 상태는 **10월 2일의 기록**이다.
- [confirmation-aggregate.json](confirmation-aggregate.json)은 새 자료 전체 판정이다. [evaluation-evidence.zip](evaluation-evidence.zip)에 당시 `portable-final-suite/`의 개발·확인 평가 배열과 보고서, `evidence/`의 감사·재현 로그를 함께 보존했다. 보고서의 해당 상대 경로는 ZIP 내부 기준이다. 세 번의 전체 실행 원본과 데모 번들을 이 저장소에 중복 보관하지 않았다.
- 현재 FastAPI·UI·서빙·release gate는 최신 main 구현을 유지했다. 이 통합은 모델 성능 재실험 또는 새 후보 배포가 아니다.

## 현재 저장소에서 재실행

```sh
python -m pip install '.[dev,research]'
python -m asofcast fetch-ett --out data/raw/ETTh1.csv
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python scripts/run_posterior_confirmation.py --out artifacts/new-confirmation
```

실행기는 ETTh1 원본 해시를 확인하고 Taylor/MSFT를 pmdarima 2.1.1 번들에서 준비한다. 출력은 새 경로여야 하며 배포 동작은 없다. 현재 main과 라이브러리 버전에서 재학습한 결과가 당시 환경의 모든 텐서와 bitwise 동일하다고 미리 주장하지 않는다.

README 상대 링크 검사를 추가해 같은 누락이 CI에서 실패하도록 했다. 최신 공개 데모 캡처는 `docs/assets/live-m2/capture-report.json`의 `dataflow-20261004` 기록으로 구분한다.
