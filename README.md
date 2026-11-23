# PSIntelligence 101: PSI 에비오닉스를 위한 AI 파이프라인
---

## 프로젝트 개요

본 프로젝트는 로켓 비행 중 실시간 아포지 (최고 고도) 예측을 목표로 하는 엔지니어링 AI 파이프라인 구축 교육 과정입니다. 단순한 머신러닝 모델 학습을 넘어, 실제 에비오닉스 (항공전자장비) 에 탑재 가능한 안전하고 신뢰할 수 있는 AI 시스템을 설계하는 데 중점을 둡니다.

### 핵심 테마

| 테마 | 설명 | 적용 단계 |
|------|------|----------|
| 물리 기반 제약 | 탄도식 출력 head와 soft prior로 예측을 제한 (전체 물리 법칙 보장 아님) | Step 3 (PINN-style) |
| 시계열 맥락 학습 | 과거 비행 이력 (자이로 요동, 틸트 등) 을 기반으로 미래 예측 | Step 4 (GRU) |
| 불확실성 정량화 | AI 가 "자신의 무지"를 인식하고, 위험 상황에서 안전모드 진입 | Step 5 (UQ) |
| 온보드 배포 | 제한된 메모리/전력의 비행 컴퓨터에서 실시간 추론 가능 | Step 6 (Quantization+ONNX) |
| 안전성 우선 | 정확도보다 예측구간과 Fail-safe 로직을 우선시하는 엔지니어링 접근 | Step 5-6 |

---

## 프로젝트 로드맵 (Step 1-6)

```
Step 1 → Step 2 → Step 3 → Step 4 → Step 5 → Step 5.5 → Step 6
  │         │         │         │         │         │        │
  │         │         │         │         │         │        └─ 경량화 + ONNX
  │         │         │         │         │         └─ 이상 탐지 (커널 + CP)
  │         │         │         │         └─ 불확실성 정량화 (MC Dropout)
  │         │         │         └─ 시계열 모델링 (GRU)
  │         │         └─ 물리 법칙 통합 (PINN)
  │         └─ 상태 추정 (EKF/UKF)
  └─ 선형 필터 (Kalman)
```

---

> Step 1.5 (데이터 생성) 는 선택 단계입니다. 동봉된 `data/simulated/` 데이터로 Step 2~6 을 바로 실행할 수 있습니다.
> Step 5.5 (이상 탐지) 는 `scripts/model_filter.py` · `scripts/generate_ood.py` 로 입력을 만들고 실행합니다.


## 각 Step 상세 설명

### [Step 1] Linear Kalman Filter: 상태 추정의 기초

목표: 센서 노이즈가 제거된 로켓 상태 (고도, 속도) 추정

핵심 내용:
- 칼만 필터의 5 개 방정식 (예측/업데이트) 구현
- 공분산 행렬 (P, Q, R) 의 물리적 의미 이해
- 1 차원 수직 비행 모델에 적용

산출물:
- `101-1 기초확률론 & 선형칼만필터.pdf`: pdf 자료

---

### [Step 1.5] 시뮬레이션 데이터 생성 — 선택 단계

목표: AI 학습용 시뮬레이션 궤적 데이터셋 생성 (200편)

핵심 내용:
- 몬테카를로 샘플링 (질량, Cd, 발사각, 풍속, 무게중심 편심, 핀 캔트각)
- 센서 노이즈 모델링 (고도·가속도·자이로), 해상도 양자화
- 물리 기반 라벨 생성 (`h_theoretical`, `energy_ratio`)

> `energy_ratio = apogee / (h + v²/2g)` 는 **0~1 로 유계가 아니다.** 연소 중에는 추력이 일을 해 1을 넘는다(실측 33.2%, 최대 1.631). 자세한 정의와 올바른 사용법은 `data/SCHEMA.md §2.1` 참조.

필요 환경: 기본 `requirements.txt` 로 충분합니다. `data/simulated/` 가 동봉되어 있어
RocketPy 없이 Step 2~6 을 실행할 수 있습니다(데이터를 직접 재생성할 때만 RocketPy 필요).

> 동봉 데이터 스키마와 알려진 결함은 `data/SCHEMA.md` 를 참조하세요.

---


### [Step 2] EKF & UKF: 상태 추정의 두 구현

목표: 측정 가속도를 적분하는 기구학 추적기에서 EKF/UKF 구현을 비교

핵심 내용:
- EKF: 상태 전이 야코비안과 Joseph form 공분산 갱신
- UKF: Unscented Transform 으로 평균/공분산 전파
- **교육 포인트:** 상태 전이와 측정이 상태에 대해 선형이므로 두 필터는 수치적으로 동일하다.
  UKF 의 UT 가 아핀 변환의 평균·공분산을 정확히 전파하는 이유를 설명한다.
- 시뮬레이션 200편 검증 (Bella Lui 수치는 외부 예시로만 인용)

> 교재의 비선형 로켓 동역학(추력·항력·중력)은 §1.1 에 서술되어 있지만, 이 노트북의
> 구현은 그 모델을 쓰지 않는다. 실제 비선형 f 구현은 Step 5.5 의 명목 물리모델 필터가 담당한다.

---

### [Step 3] Physics-informed regression: 물리 기반 제약을 쓰는 아포지 예측

목표: 탄도 기준식을 출력 구조에 활용하고, 정성적 물리 prior를 학습에 추가한다. 원 논문의 PDE 해 복원이나 계수 식별을 직접 구현한 것은 아니다.

> ⚠️ Step 3는 기존 ASL 라벨과 시간 포함 7특징을 쓰는 **legacy 참고 구현**이다.
> v2 GRU의 AGL·관성가속도·공통 endpoint 결과와 직접 비교해 아키텍처 우열을 주장하지 않는다.

핵심 내용:
- 물리 기반 출력: `G=v_z²/(2g)`, `H_pred=(h+G)·r_eff`. r은 저장 라벨 `H/(h+G)`에 대응하는 보정 배율이다.
- Hard Constraint: bounded raw output과 하한 projection으로 유효 입력 영역에서 `H_pred≥h`를 구조적으로 강제한다. 이는 전체 운동방정식·에너지 보존·정확도·안전성의 보장이 아니다.
- Soft Constraint: 범위 벌점, 관성 구간 `r>1` 벌점, 다른 특징을 고정한 시간 좌표 편미분의 음수 벌점을 적용한다. 정확한 만족이나 실제 궤적 단조성은 보장하지 않는다.
- Stability Masking: 저속 구간에서 보조 penalty의 적용을 제한한다. 현재 r 정의에서 `v_z→0`이 분모 특이점은 아니다.
- `energy_ratio` 는 0~1 로 유계가 아니다. 연소 중 r>1 은 추력 일 때문이며, `clip` 하지 않는다 (SCHEMA §2.1).

> 실측(단일 seed, clamp 분리): 저속 raw RMSE Baseline **0.001916**, PINN coast off **0.140730**,
> coast on **0.020638**. clamp 후 두 PINN은 **0.000359**(r_floor)와 같고, Baseline은 **0.000569**(clamp 62.9%)다.
> 즉 저속 헤드라인은 clamp가 지배하며, raw 기준 PINN은 Baseline보다 나쁘다. 실제 비행 112의 +0.00 m도
> clamp 전 −3.52 m, projected model −0.00 m, t=5 s −11.24 m이므로 학습 기여로 해석하지 않는다.

논문 기반:
- Raissi et al. (2019). *Physics-Informed Neural Networks*. Journal of Computational Physics. — 개념 배경이며, 이 실습은 PDE solver가 아니다.

---

### [Step 4] GRU: 시계열 맥락을 이용하는 아포지 추정기

목표: 제한된 센서 관측에서 현재 snapshot만으로 식별하기 어려운 상태·기체 특성·환경 정보를 과거 window로 추정한다.

핵심 내용:
- Sliding Window: (Batch, Seq_Len, Features) 3D 텐서 전처리 (window=100, stride=10)
- Hidden State: 과거 100 스텝 (2 초) 비행 이력 압축 저장
- Physics Wrapper: `h_apogee = (h + v²/2g)·r` 를 **학습 손실에도 포함** (r 손실 + 아포지 손실)
- 좌표계: 시뮬의 해발 고도 407 m를 빼고 **발사대 상대고도(AGL)** 로 입력·라벨·physics head를 통일.
  동압도 같은 AGL 기준 ISA 근사로 계산한다. `r` head는 `1+0.5·tanh`로 `0.5~1.5`를 표현한다.
- 가속도: `acceleration_inertial = acceleration_noisy - 9.81`(sim), 실기 `accel_g*9.80665`.
  기존 `acceleration_noisy` 원본은 보존한다.
- 출력: 모델은 **raw r/H**를 반환한다. `H_projected=max(H_raw,h)`는 진단용 옵션이며 학습·best 선택에 쓰지 않는다.
- 스케일러는 train 비행에만 fit. 학습 종료 후 validation best state를 복원해 평가·UQ·ONNX가 같은 checkpoint를 쓴다.
- 실험 계약은 `apogee_pipeline.py`, raw/projection 모델은 `apogee_models.py`, benchmark는
  `scripts/benchmark_apogee.py --dry-run`을 사용한다. split/baseline manifest는
  `data/experiments/apogee-v2/`에 있다.

> GRU가 특정 핀 틀어짐을 “기억”하는지, 과거 이력이 실제로 기여하는지는 window 길이·특징 제거·
> snapshot 기준선 ablation 없이는 확인되지 않았다. 현재 결과는 시계열 입력의 잠재적 유용성을
> 보여주는 것이지 인과적 기억 메커니즘의 증거가 아니다.

> 실측(test 50비행/1,204창): GRU 아포지 MAE **0.99 m**, constant-r 기준선 **12.49 m** → **92.1% 개선**.
> AGL 기준 target 범위는 **0.584~1.073**이다.
>
> ⚠️ 위 수치는 **기존 저장 artifact**(legacy specific-force feature, test 50비행)의 결과다.
> 새 공통 split은 calibration 25 / test 25이고, raw/inertial 코드는 아직 재학습 전이다.
> 따라서 0.99 m와 아래 3.97 m를 새 코드·새 split의 성능으로 인용하지 않는다.
>
> ⚠️ 이 수치는 시뮬레이션의 **정답 h/v(clean)** 를 physics head에 넣은 조건이다. 실기에서는
> KF 추정값을 쓴다. 실제 비행 112는 AGL `0~185.46 m`로 학습 범위에 들어오며, 기존 artifact의 최종
> raw 오차는 **3.97 m**, below-current 비율은 12.3%다. 최종 반환값 0.00 m는 projection 결과이므로
> 성능으로 세지 않는다. 또한 t≈3 s raw 오차는 약 **−30.5 m**로, 최종 한 점만으로 정확도를 일반화하지 않는다.

논문 기반:
- Chung et al. (2014). *Empirical Evaluation of Gated Recurrent Neural Networks*. arXiv:1412.3555.
- Gers et al. (2000). *Learning to Forget: Continual Prediction with LSTM*. Neural Computation.

#### M7 UKF 추정치로 학습

Avionics의 `flight-computer/README.md` §6.5에 따라 호스트용 `ukf_replay`를
빌드한 뒤 아래처럼 시뮬레이션 관측을 재생한다 (Windows 경로 예시).

```bash
python scripts/benchmark_apogee.py --state-source m7_ukf --ukf-replay ../../Avionics/flight-computer/.pio/tools/ukf_replay.exe --split-manifest data/experiments/apogee-v2/split_manifest.json --windows 100 --models gru --dry-run
```

`--dry-run`을 빼면 32유닛·1층 GRU를 학습한다. 기본 출력은
`data/experiments/apogee-v2-m7-ukf/`이며, 결과에 리플레이 실행 파일의 SHA-256을
기록한다. 기존 clean/estimated 결과 디렉터리와는 분리된다.

20 ms 격자로 맞추기 **전에** 센서 관측을 리플레이한다. 피처의 고도는 기압값,
속도·동압·물리 헤드 상태는 UKF 결과를 사용한다. 정답 아포지는 라벨로만
사용하며, 입력이 무효한 구간은 시간을 압축하지 않고 해당 창을 제외한다.
추정 상태로 재계산한 `r`이 0.5–1.5를 벗어나더라도 라벨을 자르지 않는다.
이 경로는 시뮬레이션 추정치로 학습할 수 있게 한 것이며, 센서 장착 방향·실제
센서 도착 지연·실기 아포지 정확도는 검증하지 않는다.

코스팅 예측만 비교하려면 위 명령에서 `--dry-run`을 빼고
`--endpoint-phase coasting --models gru mlp`를 지정한다. 학습·검증·시험의
예측 시점만 `flight_phase=coasting`으로 제한한다. 각 100스텝 창 안의
연소 이력은 유지하고, 정규화도 전체 구간 모델과 같은 train 비행에 맞춘다.
결과는 `data/experiments/apogee-v2-m7-ukf-coasting/`에 분리 저장된다.
`flight_phase`는 시뮬레이션 실험용 필터이며 모델 입력이나 온보드 실행
조건은 아니다.

동일 분할·스케일러와 seed 42에서 코스팅 전용 GRU의 시험 창 MAE는
1.18 m, 전체 구간 학습 GRU는 1.35 m, 코스팅 전용 snapshot MLP는
5.77 m였다. 첫 코스팅 예측 MAE는 시험에서 2.82 → 2.71 m였지만
검증에서 1.79 → 2.74 m로 악화됐다. 이 1회 비교로 코스팅 진입 직후의
개선이 재현됐다고 볼 수 없으므로 기존 후보를 교체하지 않는다.

#### 기압 관측 모델 민감도 실험

`112.txt`의 정지 기압고도 잔차(표준편차 0.0612 m, 인접 상관 약 0.83)는
정지 입력의 기준으로만 사용한다. 움직이기 시작한 뒤의 표준편차 0.12 / 0.35 /
0.75 m와 기본 지연 10 / 20 / 30 ms는 민감도 실험 가정이다. 지연에 10 ms
폴링 위상을 더해 실제 UKF에 도착 시각으로 공급한다. 1.5 m / 40 ms /
1.5% 누락 조건은 학습에서 제외한다. IMU 관측 시각과 UKF의 `R`은 바꾸지 않는다.
텍스트 로그의 반복값만으로 BMP 갱신이나 운용 중 지연을 역산하지 않는다.

```bash
python scripts/benchmark_baro_replay.py \
  --ukf-replay ../../Avionics/flight-computer/.pio/tools/ukf_replay.exe \
  --baseline-42 data/experiments/apogee-v2-m7-ukf-coasting/w100_gru_h32_l1_seed42_raw \
  --baseline-43 data/experiments/apogee-v2-m7-capacity/h32-l1/w100_gru_h32_l1_seed43_raw \
  --output-dir data/experiments/apogee-v2-m7-baro-replay
```

32유닛·1층·100스텝 모델을 같은 비행 분할에서 학습한다. 세 조건의 훈련
코스팅 끝점으로 새 스케일러를 맞추고, 기존 모델과 새 모델을 각 조건의
동일한 창에서 비교한다. `coverage.csv`는 아예 예측할 수 없는 창을 따로 센다.
`112.txt`는 정점 전에 끊기므로 실기 아포지 정확도의 검증 자료로 사용하지 않는다.

시드 42·43 평균, 시험 25편의 아포지 직전 1초 MAE는 기존 대비 저잡음
0.46 → 0.27 m, 중간 0.51 → 0.42 m, 고잡음 0.69 → 0.76 m였다.
스트레스에서는 1.83 → 1.96 m로 악화했고, 직전 1초에 예측이 가능했던
비행도 25편 중 15편뿐이었다. 두 모델의 성능은 각 조건에서 동일한
유효 창끼리 비교했다. 혼합 학습 모델은 실기 로그로 검증하기 전까지
기존 모델을 교체하지 않는다.

#### GRU 입력 창 길이 비교

```bash
python scripts/benchmark_window_replay.py \
  --ukf-replay ../../Avionics/flight-computer/.pio/tools/ukf_replay.exe \
  --output-dir data/experiments/apogee-v2-m7-windows
```

센서 조건·비행 분할·정규화와 200 ms 예측 시점을 고정한다. 32유닛 1층 GRU를
50·100·150스텝으로 각각 두 시드에 학습한다. 훈련과 검증에는 150스텝까지
유효한 같은 시점을 사용하며, 짧은 모델에는 그 창의 끝부분만 준다. 평가는
세 모델 공통 시점에서 오류를 비교하고, 각 길이가 낼 수 있었던 모든 예측은
`coverage.csv`에 별도로 센다. 같은 시점의 현재 UKF 고도와 양의 수직속도만
반영한 탄도 예측도 `matched_comparison.csv`에서 확인할 수 있다.

별도 민감도 비교에서 중간 조건(0.35 m, 기본 지연 20 ms)을 기준으로
한 항목씩 바꿨다. 아포지 직전 1초의 UKF 수직속도 MAE는 기압 잡음만
0.12 → 0.35 → 0.75 m로 높일 때 0.68 → 1.96 → 4.21 m/s였다.
누락률만 1.5%로 바꾸면 100스텝 기준 유효 창은 125 → 46개로 줄었다.
누락 조건의 오차는 남은 창에서만 계산하므로 기준 조건과 직접 비교하지 않는다.

시험 결과는 100스텝을 기본 후보로 유지하는 쪽이다. 중간 조건의
아포지 0.5–2초 전 MAE는 50/100/150스텝에서 0.72/0.56/0.59 m였고,
탄도·현재고도 기준선은 각각 2.42/8.99 m였다. 스트레스 조건에서 아포지
직전 1초에 예측한 비행은 각각 22/15/9편(전체 25편)이었다. 50스텝은
누락 복구 후보지만 스트레스에서 100스텝과 공유하는 48개 창의 MAE는
2.30 m로 100스텝(1.96 m)보다 높았다. 실기 검증이나 펌웨어 적용은 하지 않았다.

#### GRU 수렴 진단

```bash
python scripts/benchmark_window_replay.py \
  --ukf-replay ../../Avionics/flight-computer/.pio/tools/ukf_replay.exe \
  --output-dir data/experiments/apogee-v2-m7-ukf-convergence \
  --train-window 100 --epochs 240 --trace-convergence
```

데이터·분할·센서 조건과 학습률 스케줄은 기존 창 비교와 동일하다.
진단 모드에서만 조기 종료를 건너뛰고, 각 시드의 `epoch_history.csv`에
훈련 중 배치 손실의 샘플 가중 평균, 검증 손실, 학습률, 검증 아포지
직전 1초 MAE, 최저 검증 손실 이후 경과 epoch를 기록한다. 체크포인트는
기존과 같이 검증 손실로 선택하며, 아포지 직전 MAE는 별도 진단 지표다.

#### 훈련 전용 기압 누락 증강

```bash
python scripts/benchmark_window_replay.py \
  --ukf-replay ../../Avionics/flight-computer/.pio/tools/ukf_replay.exe \
  --output-dir data/experiments/apogee-v2-m7-ukf-burst-aug \
  --train-window 100 --epochs 240 --trace-convergence --augment-baro
python data/experiments/apogee-v2-m7-ukf-burst-aug/compare_augmentation.py
```

기존 훈련 비행에만 기압 잡음·기본 도착 지연·연속 누락 길이가 각각
0.95 m / 40 ms / 2샘플, 1.25 m / 50 ms / 3샘플인 관측을 추가한다.
누락은 비행이 움직이기 시작한 뒤 합성하며 IMU는 그대로 둔다. 정규화는
증강한 훈련 창에서 다시 맞추고, UKF 복구 마스크와 기존 검증·시험 비행 및
센서 프로파일은 바꾸지 않는다. 비교 스크립트는 동일 창에서 기준 모델과
비교하고, 별도의 4샘플 연속 누락 조건과 `112.txt`의 기존 두 리플레이
가정도 확인한다. 추가 프로파일 수치는 실측 센서 사양이 아니다.

#### `112.txt` 10시드 비교

위 증강 실험의 시드 42·43을 재사용한다. 같은 옵션으로 `--output-dir`의
끝에 `seed-study-batch-a`와 `seed-study-batch-b`를 각각 붙이고, 시드를
`--seeds 504723 873529 698543 608065`와 `--seeds 650042 313474 733519 487659`로
지정해 추가 학습한다. 그 후 프로젝트 루트에서 실행한다.

```bash
python data/experiments/apogee-v2-m7-ukf-burst-aug/plot_112_seed_study.py
```

그래프는 실제 온보드 UKF 고도와 각 로그 행을 새 BMP 샘플로 간주한 리플레이의
GRU 예측을 나란히 표시한다. CSV에는 기압값이 바뀐 행만 공급한 가정도 포함한다.
시드 간 최솟값–최댓값은 모델 초기화에 따른 산포이며 실비행 오차나 신뢰구간이 아니다.

---

### [Step 5] Uncertainty Quantification: AI 의 자신감 측정

목표: 예측 불확실성을 정량화하고, 그 한계를 확인한다. OOD 탐지는 Step 5.5의 커널+CP가 담당.

> ⚠️ Step 5/6 코드는 공통 inertial feature와 raw 출력 경로로 갱신했지만, 저장된 UQ/ONNX
> 출력은 기존 legacy artifact 실행 결과다. 새 scaler·checkpoint·calibration을 재생성하기 전에는
> 두 결과를 같은 모델의 성능으로 합치지 않는다.

> 실측: native MC Dropout 95% 포함률 **86.2%** (독립 cal 25비행/test 25비행).
> 창 CP: 경험적 창 포함률 **94.6%**, 폭 **8.8 m**. 비행 최대잔차 CP: 비행 포함률 **96.0%**, 폭 **23.1 m**.
> OOD AUROC(비행별 σ 95분위, 정상 25 vs OOD 50)는 **0.383** — mass 0.000 · loft 0.529 ·
> drag 0.592 · wind 0.502. dropout 단독 OOD 탐지는 계열 선택적.
> conformal 후보 5.40 s(v=16.1 m/s)·4.71 s(v=21.5 m/s)는 모두 상승 중이므로 사출 성공이 아니다.

핵심 내용:
- Aleatoric Uncertainty: 센서 노이즈 등 데이터 고유 불확실성 (이 노트북은 **구현하지 않음**)
- Epistemic Uncertainty: 모델 무지. OOD에서 항상 커진다는 보장은 없으며 별도 검증이 필요하다.
- MC Dropout: dropout으로 학습한 모델에서 추론 시 dropout을 유지해 T회 순전파. T회 비용이 들고, 지연은 병렬 자원·배치에 의존한다.
- `mean ± 1.96σ_MC`: 보정 전 **dropout 변동 띠**다. 95% 예측 포함률을 보장하지 않는다.
- conformal 구간: 별도 calibration 절차. 창 단위와 비행 최대잔차 단위를 구분한다.
- 사출 예시의 `σ<2 & lower≤h`는 교육용 후보 규칙이며, 아포지 도달·안전 사출을 보장하지 않는다.

논문 기반:
- Gal & Ghahramani (2016). *Dropout as a Bayesian Approximation*. ICML.
- Kendall & Gal (2017). *What Uncertainties Do We Need in Bayesian Deep Learning?* NIPS.

---

### [Step 5.5] 커널 기반 이상 탐지 + Conformal Prediction

목표: 물리 잔차 기반 검정과 distribution-free(교환가능성 가정) CP를 계층적으로 결합

핵심 내용:
- 명목 물리모델 잔차 — 명목 질량·Cd·발사각으로 동역학을 적분해 비행체 이상을 드러냄
  (Step 2 의 기구학 추적기는 측정 가속도를 입력으로 써서 파라미터 이탈을 못 잡음 — 실측 1.00x)
- L1 χ² 검정 — 잔차가 독립 표준정규이고 공분산이 올바를 때 χ². 현재는 근사이며,
  백색성 위반(자기상관 +0.40)으로 이론 임계 대신 calibration 임계를 사용한다 (Mehra 1970 계열).
- L2 OC-SVM(RBF) + CP — 비가우시안·모델오차 잔차를 다루는 one-class 경계 + CP
- 비행 단위 CP — 사전에 정의한 비행별 점수의 교환가능성 아래 marginal 보장. 창 상관 자체가
  CP를 무효화하지는 않지만, 창 단위 점수와 비행 단위 사건은 다른 보장 대상이다.
- Safe-Logic 2.0 연동 — L0/L1 결정적 우선, L2 는 N-of-M + 히스테리시스 (`safe_logic.py`)

실측 결과 (정상 200편 + OOD 50편, FPR 목표 ≤ 10%, 비행 단위):

| 방법 | FPR(비행) | TPR(비행) |
|:--|:--:|:--:|
| L1 χ² (단발 초과, 비행단위 보정) | 14.0% (cal 10%) | 64.0% |
| OC-SVM + CP | 8.0% | 56.0% |
| KDE + CP | 6.0% | 52.0% |
| kPCA + CP | 4.0% | 52.0% |

> L1 은 calibration 50편에서 FPR 10% 로 맞췄지만 test 50편에서 14% 로 튄다.
> 50비행 분위수 추정의 한계이며, N-of-M 규칙 전체를 CP 로 calibration 하는 것이 남은 과제다.
> 창 단위 CP + N-of-M 은 비행 FPR 이 36% 이므로, 비행 단위 보장이 필요하면 OC-SVM+CP(8%)를 쓴다.
> 이 8%는 한 test 집합의 경험적 비율이며, 이론적 marginal 보장의 증명이 아니다.

계열별로는 질량 93 - 100% · 항력 67 - 100% · 저각 11 - 56% · 강풍 0 - 18% 로 탐지가 계열 선택적

입력 생성:
```bash
python scripts/generate_ood.py      # 정상 범위를 벗어난 OOD 50편
python scripts/model_filter.py      # 명목 물리모델 잔차 → residuals.npz
```

논문 기반:
- Schölkopf et al. (2001). Estimating the Support of a High-Dimensional Distribution. Neural Computation. — OC-SVM
- Tax & Duin (2004). Support Vector Data Description. Machine Learning. — SVDD
- Vovk et al. (2005). Algorithmic Learning in a Random World. — Conformal Prediction
- Angelopoulos & Bates (2021). A Gentle Introduction to Conformal Prediction. arXiv 2107.07511.
- Mehra, R. (1970). On the Identification of Variances and Adaptive Kalman Filtering. IEEE TAC 15(2), 175–184.

> 한계: 시뮬 기반 CP 보장은 calibration/test 교환가능성과 사전에 정의한 score/사건에만 적용된다.
> 위상별 FPR, 조건부 FPR, 다른 N-of-M 사건에는 별도 정당화가 필요하다.

---

### [Step 6] Quantization & Edge Deployment: 비행 컴퓨터를 위한 AI 경량화

> 이 단계는 **GRU 배포** 실습이다. Mamba 는 이론 배경으로만 소개하며, 실제 모델은 Step 4의 GRU다.

목표: 호스트에서 GRU 경량화·ONNX export를 실습한다. MCU 탑재는 별도 변환·툴체인 검증이 필요하다.

핵심 내용:
- Quantization: PyTorch dynamic INT8 (파일 크기 약 1/4, 이 PC CPU 지연은 별도)
- ONNX Export: FP32 `apogee` + `energy_ratio` 두 출력. INT8 ONNX 변환은 이 실습 범위 밖
- 수치 동등성: 같은 입력에서 PyTorch↔ONNX 최대 차이 출력
- 속도 벤치마크: 이 PC CPU 기준. 온보드 MCU 지연은 별도 측정 필요

산출물:
- `rocket_gru.onnx`: 호스트 검증용 FP32 ONNX (아포지 + r 두 출력)

---

## 환경 설정 (micromamba & uv)

본 프로젝트는 재현성 확보와 의존성 충돌 방지를 위해 `micromamba` (가상환경) 와 `uv` (패키지 관리) 를 사용합니다.

### 2. micromamba 설치

```bash
# 기존에 conda를 쓰고 있었다면 그대로 conda를 써도 무방함.
# conda가 설치되어 있지 않을 경우, 가볍고 관리가 수월한 micromamba 활용을 권장
# Linux/macOS
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba

# Windows (PowerShell)
winget install -e --id Mamba.Micromamba
```

### 2. 가상환경 생성

```bash
# 프로젝트 루트에서 실행
micromamba create -n psintel python=3.10 -c conda-forge

# 환경 활성화
micromamba activate psintel
```

### 3. uv 설치 및 의존성 설치

의존성 파일은 `requirements.txt` 하나입니다. RocketPy 는 기본 경로에 필요하지 않으므로
그 안에서 선택 항목으로 분리해 두었습니다.

```bash
# uv 설치 (pip 대체제) < 훨씬 빠르고 가벼움
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
owershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 기본 설치
uv pip install -r requirements.txt
```

| 포함 범위 | 패키지 |
|------|------|
| 필수 (Step 2~6) | numpy, pandas, scipy, matplotlib, scikit-learn, joblib, torch, onnx, onnxruntime |
| [선택] (Step 1.5 데이터 재생성) | rocketpy, seaborn — `requirements.txt` 안에서 주석을 풀어 설치 |
| [선택] (노트북 실행) | jupyterlab — 대부분의 환경에 이미 포함 |

> RocketPy 는 기본 경로에 필요하지 않습니다. RocketPy 는 `netCDF4`(h5py·C 라이브러리 의존) 등 무거운 패키지를 끌어오며, Step 1.5 의 데이터 재생성에만 쓰입니다.
> 자체 시뮬레이터(`scripts/generate_dataset.py`)를 쓰면 RocketPy 없이도 데이터를 생성할 수 있습니다.
> 환경 완전 재현이 필요하면 정확한 pin 153개가 git 이력에 남아 있습니다: `git show 0937b37:requirements-full.txt`


---

## 빠른 시작

```bash
# 1. 저장소 클론
git clone https://github.com/postech-psi/psintelligence.git
cd psintelligence

# 2. 가상환경 생성 / 활성화
micromamba create -n psintel python=3.10 -c conda-forge
micromamba activate psintel

# 3. 핵심 의존성만 설치 (RocketPy 불필요)
uv pip install -r requirements.txt

# 4. Step 2 부터 순차 실행  (Step 1 은 pdf, Step 1.5 는 선택)
jupyter notebook
```

Step 1.5 (데이터 재생성) 는 선택 단계입니다. `data/simulated/` 에 200편 시뮬레이션 데이터와
학습된 모델이 이미 포함되어 있어, Step 2~6 을 그대로 실행 가능
데이터를 직접 만들어보고 싶다면 두 가지 경로가 존재함

```bash
# (A) 자체 시뮬레이터 — RocketPy 불필요, 표준 라이브러리만 (200편 약 2.3초)
python scripts/generate_dataset.py
python scripts/verify_dataset.py --data-dir data/generated

# (B) 원본 경로 — RocketPy 로 step1.5 노트북 실행
uv pip install "rocketpy>=1.11" "seaborn>=0.13"   # 선택 의존성
```

데이터를 수정하거나 재생성한 뒤에는 무결성 검증을 실행 가능

```bash
python scripts/verify_dataset.py                 # 정본 검사 (21항목)
python scripts/verify_dataset.py --data-dir <경로>  # 생성 데이터 검사
```

> 스키마·알려진 결함·재현 모델: [`data/SCHEMA.md`](data/SCHEMA.md)

### 단일 비행 추론 데모

학습된 GRU로 실제 로그 한 편의 아포지를 예측하려면:

```bash
python main.py --csv data/112.txt
```

출력: AGL 기준 **raw r/H**, raw 오차, 로그 최대값, constant-r(r=1) 기준선,
`below_current_altitude` 비율. 실제 로그 고도는 발사 시점을 0으로 둔 AGL로 변환한다.
projection까지 보려면 `--show-projected`를 명시한다.

현재 배포 artifact는 legacy specific-force scaler로 학습된 것이므로 metadata에 encoding 표시가
없으면 CLI가 `legacy_specific_force`로 판정해 scaler mean을 한 번 이동한다. 새 Step 4 artifact는
`acceleration_encoding='inertial'`을 저장하므로 `--accel-encoding auto`가 그대로 사용한다.
수동 진단은 `legacy`/`inertial`을 명시할 수 있다.

### 공통 실험 계약

```bash
python test_apogee_pipeline.py
python test_apogee_benchmark.py
python -m py_compile apogee_pipeline.py apogee_models.py main.py
python scripts/benchmark_apogee.py --dry-run --windows 25 50 100
python scripts/benchmark_apogee.py --dry-run --state-source estimated
```

- `data/experiments/apogee-v2/split_manifest.json`: train 120 / validation 30 /
  calibration 25 / test 25.
- `data/experiments/apogee-v2/baseline_manifest.json`: `HEAD@c152ea7`의
  best/last/scaler hash, feature 순서, legacy specific-force artifact encoding, AGL datum,
  state source를 기록한다. 현재 작업 트리 artifact가 다르면 `validate_artifact_hashes`가
  불일치를 보고한다.
- benchmark는 `data/experiments/apogee-v2/<run>`에만 쓰며 `data/simulated/` 정본을
  덮어쓰지 않는다.
- calibration 25비행은 모델 선택·학습에 사용하지 않는다.
- 현재 torch 학습/추론은 실행 제한으로 검증하지 않았다. `test_apogee_models.py`와
  실제 benchmark 수치는 torch 허용 환경에서 별도 실행해야 한다.


---

## 교육적 목표

본 프로젝트는 단순한 코드 구현을 넘어 다음과 같은 엔지니어링 사고방식을 함양합니다:

1. 점진적 복잡도: 선형 칼만 (Step 1) → 기구학 EKF/UKF (Step 2) → AI (Step 3-6)
2. 물리 기반 AI: 데이터만 믿지 않고 물리 법칙을 제약 조건으로 활용
3. 안전성 우선: 정확도보다 불확실성 인식과 Fail-safe 로직을 우선시
4. 배포 고려: 연구실 PC 가 아닌 온보드 컴퓨터 제약을 고려한 모델 설계
5. 논문 기반: 관련 논문·공식 문서를 바탕으로 구성하되, 원 방법과 다른 단순화·구현 범위·미검증 한계를 구분

---

## 주요 참고 문헌

| Step | 논문 | 저자 | 학회 |
|------|------|------|------|
| 1-2 | *A New Approach to Linear Filtering and Prediction Problems* | Kalman (1960) | - |
| 3 | *Physics-Informed Neural Networks* | Raissi et al. (2019) | J. Comput. Phys. |
| 4 | *Empirical Evaluation of Gated RNNs* | Chung et al. (2014) | arXiv:1412.3555 |
| 4 | *Learning to Forget: LSTM* | Gers et al. (2000) | Neural Computation |
| 5 | *Dropout as Bayesian Approximation* | Gal & Ghahramani (2016) | ICML |
| 5 | *What Uncertainties Do We Need?* | Kendall & Gal (2017) | NIPS |
| 6 | *Mamba: Linear-Time Sequence Modeling* | Gu & Dao (2023) | arXiv:2312.00752 |
| 6 | *Vision Mamba* | Zhu et al. (2024) | arXiv:2401.09417 |

---

## 주의사항

1. Step 순서 준수: 각 Step 은 이전 Step 의 산출물을 사용하므로 순차적 실행 권장
2. 데이터: `data/simulated/` 동봉 데이터로 Step 2~6 실행 가능. Step 1.5 재생성은 선택
3. GPU 권장: Step 4-6 은 GPU 가 있으면 학습 속도가 5-10 배 빠름
4. 메모리 관리: Step 6 ONNX Export 시 모델 크기가 급증할 수 있으므로 주의

---

## 문의

- 개발자: 포항공과대학교 기계공학과 21학번 이승원
- 이메일: dongdong0615@postech.ac.kr / sungwon.lee.2002@gmail.com

---


> PSI 후배들이 AI를 접하고 공부하는데 도움이 되기를 바랍니다.
