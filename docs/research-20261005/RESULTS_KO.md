# 2026년 10월 5일 행동 정책 평가와 제출 판단

**판단: 후보를 채택하지 않고 기존 서비스 모델을 유지합니다.** 강한 단순 기준 대비 비용 포함 손실 감소는 AppliancesEnergy에서 0%, SeoulBike에서 -0.14~-0.43%였습니다. 여섯 조건 모두 사전 채택 기준에 미달했습니다.

## 출처 재검토에 따른 정정

**AppliancesEnergy는 새 독립 자료가 아닙니다.** 9월 29일 `Appliances` 실험과 같은 UCI 374 `energydata_complete.csv` 원본입니다. 이전 실험은 1시간 집계·7개 열, 이번 실험은 원래 10분 간격·9개 열을 사용했지만, 표현을 바꿨다고 데이터의 독립성이 생기지는 않습니다. 이번 제출 감사에서 이를 확인했습니다. 이번 두 자료를 모두 새로운 독립 자료라고 표현한 이전 보고서와 `protocol.json`의 선언은 이 정정으로 대체합니다. 과거 선언과 수치는 감사 이력으로 보존하며 소급 수정하지 않습니다.

SeoulBike는 이번에 처음 사용한 UCI 560 자료입니다. 이미 본 자료를 새로운 확인 자료로 세지 않습니다. 수치 결과와 미채택 판단은 바뀌지 않으며, 두 독립 자료에서 일반적인 우위를 검증했다고 주장하지 않습니다.

## 결과

양수는 비용 포함 손실 감소입니다. 구간은 백분율이 아니라 표준화 손실 차이입니다. 비교 기준은 검증 구간에서 선택한 강한 단순 방법입니다.

| 자료 / 조건 | 손실 감소 | 95% 구간 | 채택 |
|---|---:|---:|---|
| AppliancesEnergy / mixed_2811 | 0.000% | [0, 0] | 미달 |
| AppliancesEnergy / mixed_3811 | 0.000% | [0, 0] | 미달 |
| AppliancesEnergy / outage_2811 | 0.000% | [0, 0] | 미달 |
| SeoulBike / mixed_2811 | -0.372% | [-0.003937, -0.000264] | 미달 |
| SeoulBike / mixed_3811 | -0.141% | [-0.004252, 0.002727] | 미달 |
| SeoulBike / outage_2811 | -0.430% | [-0.004613, -0.000128] | 미달 |

AppliancesEnergy에서는 검증 단계에서 단순 전략을 유지했습니다. SeoulBike에서는 학습 정책이 채택됐지만 최종 평가에서 추가 센서 취득 비용을 상쇄하지 못했습니다. 예측 오차와 대기 손실이 일부 줄어도 전체 목적함수가 좋아지는 것은 아닙니다. [비용 분해](cost-decomposition.json)에 항목별 수치를 보존했습니다.

채택에는 두 비교 기준 각각 대비 비용 포함 손실 1% 이상 감소, 95% 구간 하한 양수, 세 시드 모두 개선, 시드별 MAE 악화 최대 1%를 모두 요구했습니다. 동일 목표의 세 시드를 먼저 평균한 후 원점 단위 순환 블록 bootstrap을 사용했습니다(주 분석 42, 보조 분석 6, 2,000회). 도착 지연·센서 비용은 시뮬레이션이며 실제 운영 비용이 아닙니다.

## 재현과 코드

12회 학습은 2개 자료 × 3개 시드 × 2회 재학습입니다. 이는 재현성 확인이며 통계적으로 독립인 표본을 추가한 것이 아닙니다. 결과 배열 558개, 저장 정책 재생 배열 216개가 동일했고 미래 값 변경 검사 72개가 통과했습니다.

- [평가 원본 JSON](confirmation-summary.json): SHA-256 `1a2ceeaa921c01cb875f1fc177ec083f57ae7447f46f72dc17a0e7c3e2078812`
- [고정 프로토콜 원본](protocol.json): 데이터 신규성에 관한 문구는 위 정정과 함께 읽습니다.
- [연구 소스 1dfcaf5](https://github.com/sokldjs554/asofcast/tree/1dfcaf5da0d87961f519e9c772069e5a11c825b1) · [연구 PR 27](https://github.com/sokldjs554/asofcast/pull/27)
- [연구 소스 CI](https://github.com/sokldjs554/asofcast/actions/runs/37286107554): Python 3.11/3.13에서 각각 279개 테스트. 현재 서비스 테스트 개수와 구분합니다.
- [UCI Appliances 원본](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction), DOI 10.24432/C5VC8G
- [UCI SeoulBike 원본](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand), DOI 10.24432/C5F62R

모델 학습과 후보 선택 코드는 연구 브랜치에 보존했습니다. 제출용 main에는 이 집계·설명과 표시 검증만 반영하며 연구 모델을 서비스 모델로 교체하지 않습니다. 집계 표시 생성기는 원본 SHA-256과 수치 gate를 다시 확인합니다.

## 실험 이력

[9월 29일 예측기와 정책 분리 평가](../research-diagnosis-20260929.md) · [10월 2일 Taylor·MSFT 평가](../research-20261002/RESULTS_KO.md). Appliances의 예측 MAE 10.83~11.15% 개선은 이전 별도 예측기 실험 결과이며 이번 정책의 성과로 합산하지 않습니다.
