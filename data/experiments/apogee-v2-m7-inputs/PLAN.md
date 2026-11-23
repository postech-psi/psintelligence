# 실험 계획 A — GRU 입력 상태 교체 (M7 UKF → 기압+가속도 평활 상태)

## 배경

M7 UKF 리플레이에서 시뮬 UKF 속도 오차는 평균 5.0 m/s다. 같은 기압·가속도로 만든 최소제곱 평활 상태
(`apogee_physics.kinematic_smoother`, 1.5 s 창)는 0.44 m/s다. 지금 GRU는 UKF 속도를 입력 특징과 physics head
(`H = (h + v²/2g)·r`) 양쪽에 쓰므로, 이 오차를 모델이 학습으로 걸러내야 한다. 모델 크기를 7배로 늘려도
성능 차이가 시드 차이보다 작았으므로(`apogee-v2-m7-capacity`), 병목은 용량보다 입력 품질일 가능성이 높다.

## 가설

입력 특징과 physics head의 h, v를 평활 상태로 바꾸면 관성 구간 아포지 MAE가 줄어든다.

## 변형 (모두 GRU 32×1, 창 100, stride 10, 120 epoch, patience 12, 시드 42·43)

| 이름 | 입력 특징 | physics head의 h, v |
|:--|:--|:--|
| A0 `ukf` (대조군) | 현행 6특징: 기압, UKF v, 가속도, tilt, gyro, q(UKF) | UKF |
| A1 `smoothed` | 기압, **평활 v**, 가속도, tilt, gyro, q(평활) | 평활 |
| A2 `ukf+smoothed` | 현행 6특징 + 평활 h, 평활 v (8특징) | 평활 |

- A0은 같은 스크립트로 다시 학습한다. 기존 `apogee-v2-m7-capacity/h32-l1` 수치(test 관성 MAE 약 1.2 m)와
  비슷하게 나오는지로 스크립트 정합성을 확인한다.
- A1 대 A0은 "UKF 상태를 평활 상태로 교체"한 효과이고, A2 대 A1은 "UKF 채널을 함께 줄 때"의 추가 효과다.

## 고정 조건

- 데이터: `data/simulated/all_trajectories.csv.gz`(SHA `529186…`), 재생성 없음.
- 상태: `benchmark_apogee.py --state-source m7_ukf`와 같은 리플레이(`flight-computer/tools/ukf_replay`).
- 분할: `data/experiments/apogee-v2/split_manifest.json`(train 120 / val 30 / cal 25 / test 25).
- endpoint: `flight_phase == coasting` ∩ 100스텝 창 유효 ∩ 물리 기준선 정의. **모든 변형과 실험 B가 같은 집합**을 쓴다.
- 스케일러: 변형별 특징으로 train 비행 endpoint에서만 fit한다.
- 선택은 validation 손실로만 하고, test는 마지막에 한 번 본다.

## 평가

test 관성 endpoint에서 MAE, bias, P95|e|, 과대 5 m 초과 수를 본다. 구간은 다음과 같다.
- 전체
- 관성 진입 첫 1 s
- 아포지 직전 1 s
- 비행별 마지막 예측
- 남은 시간 대역(0–0.5, 0.5–1, 1–2, 2–3, 3–4, 4+ s)

학습 없는 물리 기준선(평활 상태)도 같은 표에 넣는다.

## 판정 기준 (실행 전 고정)

- **채택**: A1 또는 A2의 test 관성 MAE가 A0보다 **두 시드 모두** 0.1 m 이상 낮고, 첫 1 s 구간이 나빠지지 않을 때.
- **보류**: 개선이 0.1 m 미만이거나 시드 사이에 부호가 갈릴 때. 0.1 m는 기존 실험의 시드 간 차이(약 0.1 m)다.
- 채택하더라도 펌웨어 반영은 별도다. 평활기는 C++ 이식이 필요하다(창 75샘플 선형 최소제곱).

## 산출물

`data/experiments/apogee-v2-m7-inputs/`에 다음을 둔다.
- `summary.csv`, `predictions.csv`, `run_config.json`
- 변형·시드별 `model.pth`, `scaler.pkl`, `run.json`

실행 스크립트: `scripts/experiment_m7_inputs_residual.py --experiment inputs`.
