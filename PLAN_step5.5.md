# Step 5.5 계획안: 커널 기반 이상 탐지 — 필터 혁신(Innovations) 위에 통계적 보장 얹기

버전: v0.1 (초안)
작성일: 2026-08-09
위치: `step5-UQ.ipynb` (Step 5) 와 `step6-Mamba&Deploy.ipynb` (Step 6) 사이
파일명 후보: `step5.5-kernel-anomaly-detection.ipynb`

---

## ⚠️ 구현 완료 — 이 문서는 초안이며 아래 내용으로 대체되었습니다

구현: `step5.5-kernel-anomaly-detection.ipynb` (2026-10-07, 전 셀 실행 검증 완료)
확정 사양: 위키 `plans/psintelligence-step5.5-kernel-anomaly.md` (updated 2026-08-18) 의
결정 D / E / F 가 이 초안보다 최신입니다.

### 초안이 틀렸던 부분 (실측으로 반증)

이 초안 §1.1 은 *"EKF/UKF 가 이미 '이상이 있으면 혁신이 커진다'는 정보를 만든다"* 를 전제로 합니다.
이 전제는 Step 2 의 필터 설계에서 성립하지 않습니다.

Step 2 의 EKF/UKF 는 측정된 가속도를 입력으로 쓰는 기구학 추적기입니다. 어떤 궤적이든
"그게 정답"으로 따라가므로 혁신에는 센서 잡음만 남습니다.

| 필터 | 정상 vs 파라미터 이탈 OOD 잔차 비 | 비행 단위 TPR |
|:--|:--:|:--:|
| Step 2 기구학 (초안의 전제) | 1.00x | ~0% |
| 명목 물리모델 (구현에서 채택) | 2.05x | 54~60% |

→ 구현은 `scripts/model_filter.py` 의 명목 물리모델 잔차를 L1/L2 공통 입력으로 씁니다.
   Step 2 의 혁신(`innovations.npz`)은 센서 결함 채널로 남겨둡니다.

### 초안 대비 구현에서 바뀐 것

| 항목 | 초안 | 구현 |
|:--|:--|:--|
| L1/L2 입력 | Step 2 혁신 | 명목 물리모델 잔차 (센서 결함은 Step 2 혁신이 담당) |
| 혁신 파이프라인 | (b) 내부 재실행 권장 | 위키 결정 E-가 채택 — Step 2 가 혁신을 `.npz` 로 저장 |
| CP 적용 단위 | 명시 없음 | 비행 단위 — 창 단위는 상관 때문에 FPR 이 α 를 초과 (창 12.2% vs 비행 6.0%) |
| 커널 주축 | 4종 비교 | OC-SVM 주축 + KDE 는 기준선 (KDE 가 TPR 60% 로 더 높지만 ONNX 미지원·추론 19배) |
| OOD | 존재 가정 | `scripts/generate_ood.py` 로 4계열 50편 생성 (정본 200편은 전부 정상) |
| 위상 | 부스트/코스트/아포지/강하 4개 | 2개(부스트/관성) — 데이터가 아포지에서 잘림 |

### 측정된 탐지력 (정상 200편 + OOD 50편, FPR 목표 ≤ 10%, 비행 단위)

| 방법 | FPR(비행) | TPR(비행) | drag | loft | mass | wind |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| L1 χ² (보정임계 + N-of-M) | 6.0% | 68.0% | 100% | 56% | 93% | 0% |
| OC-SVM + CP | 6.0% | 54.0% | 67% | 11% | 100% | 9% |
| KDE + CP | 6.0% | 60.0% | 80% | 22% | 100% | 9% |
| kPCA + CP | 4.0% | 54.0% | 67% | 11% | 100% | 9% |

탐지가 계열 선택적입니다 — 질량은 완전 탐지, 항력은 양호, 저각·강풍은 취약.

### L1 임계 재calibration (2026-10-07)

χ² 이론 임계는 $z_k$ 가 백색일 때만 성립합니다. 명목 모델 잔차의 lag-1 자기상관을
실측하니 +0.3994 로 유의하게 위반 → 이론 임계의 실제 창 FPR 이 34.4% 였습니다.

- 시도한 오진: $\hat\rho = \mathbb{E}[z_{std}^2] = 0.55$ 로 $S$ 를 스케일 보정 →
  FPR 이 34.4% → 56.3% 로 악화. 창 내 표준편차는 저주파 드리프트를 과소평가하므로
  분산 추정치가 아닙니다.
- 채택한 해법: 임계를 calibration 창에서 경험적으로 재설정 → $q = 84.21$,
  창 FPR 11.7%. 비행 단위로는 FPR 6.0% / TPR 68.0% 로 전 방법 중 최고.
- 남은 과제: 잔차 백색화(AR 등)로 이론 임계를 되살리기.

### 비행 단위 계층 검증 (B11)

| 채널 | 정상 abort | OOD abort |
|:--|:--:|:--:|
| L2만 (OC-SVM+CP) | 0.0% | 32.0% |
| L1만 (χ²) | 6.0% | 68.0% |
| L1+L2 | 6.0% | 68.0% |

정상 abort 율이 α=10% 이하이고 OOD abort 율이 68% — 결정 D 의 계층 구조가 작동합니다.

### 조건이 지배한다 — 반증 실험 (a) 결과 (2026-10-07)

⚠️ 아래 "OC-SVM 이 χ² 를 이기지 못한다"는 결론은 집계 규칙에 의존하며, 모양형 OOD 에서는
정반대다. 실험 (a) 를 수행한 결과로 이 절을 정정한다.

모양형 OOD 생성 (`scripts/generate_ood_shape.py`): 파라미터는 전부 정상 범위,
고도 잡음의 표본 σ 를 1.5 m 로 강제하고 모양만 바꿈 — `ar1`(상관) / `t3`(중꼬리) /
`sine`(협대역). 50편.

실험 성립 검증 (이게 성립해야 실험이 유효하다)

| 항목 | 정상 | 모양형 OOD | 비 |
|:--|:--:|:--:|:--:|
| 창 Σz² 중앙 | 21.656 | 21.294 | 0.98x ✓ 분산 동일 |
| 창 z_std 중앙 | 0.728 | 0.663 | 0.91x |
| lag-1 자기상관 | −0.031 | +0.676 | 모양 크게 다름 |

동일 조건 비교 (비행 단위, TPR@FPR=10%, 95분위 집계)

| 검출기 | 모양형 AUC | 모양형 TPR | 범위형 AUC | 범위형 TPR |
|:--|:--:|:--:|:--:|:--:|
| χ² (크기 통계) | 0.568 | 18% | 0.808 | 58% |
| OC-SVM (크기 피처) | 0.829 | 70% | 0.746 | 66% |
| OC-SVM (모양 피처) | 0.995 | 98% | 0.557 | 18% |
| OC-SVM (전체) | 0.998 | 100% | 0.733 | 54% |

결론 (정정 포함)

1. 지배 요인은 검출기가 아니라 "피처 ∩ 이상 유형"의 정합이다.

   | 이상 유형 | χ² (크기) | 커널 one-class |
   |:--|:--:|:--:|
   | 크기 변화 (파라미터 이탈) | 58~68% | 54~66% |
   | 모양 변화 (분산 동일) | 18% | 98~100% |

   크기 피처 → 크기 이상에만 작동. 모양 피처 → 모양 이상에만 작동.

2. 앞선 "χ² 가 커널 3종을 모두 이긴다"는 결론은 집계 규칙에 의존한다.
   N-of-M 규칙에서는 χ² 68% > OC-SVM 56% 였으나, 95분위 집계 + 동일 FPR 에서는
   OC-SVM(크기 피처) 66% > χ² 58% 로 뒤집힌다 (AUC 는 χ² 우세 0.808 vs 0.746).
   → 집계 규칙도 검출기만큼 중요하다. 두 수치를 나란히 보고해야 한다.

3. 모양형에서 OC-SVM 은 압도적이다 — AUC 0.995, TPR 98% vs χ² 18%.
   계획이 커널을 채택한 근거("분포 무가정 one-class")는 모양형 이상에 대해 유효하다.

4. 올바른 사용법은 배타적 선택이 아니라 채널 병치다 — χ²(크기) + 커널(모양).
   두 채널의 실패 모드가 서로 겹치지 않는다 (18% vs 98% 가 교차).
   이는 결정 D 의 계층 구조(L1 결정적 / L2 ML)와도 정합한다.

남은 반증 실험

- (b) 크기 통계량을 빼고 모양 피처만 — 본 실험에서 이미 수행(98%)
- (c) 6-DOF 자세 채널 같은 교차 구조 피처

### 부록: 범위형 OOD 에서의 초기 비교 (집계 규칙 주의)

⚠️ 위 표는 집계 규칙이 섞여 있다 — L1 에는 N-of-M 비행 규칙, 커널에는 95분위 집계를
적용했다. 같은 N-of-M 규칙으로 통일하면 아래와 같다 (창 임계는 각각 calibration 창에서 CP).

| 방법 | 창 임계 | FPR(비행) | TPR(비행) | drag | loft | mass | wind |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| L1 χ² | 84.21 | 6.0% | 68.0% | 100% | 56% | 93% | 0% |
| OC-SVM | 0.125 | 22.0% | 56.0% | 87% | 22% | 87% | 0% |
| KDE | 11.35 | 8.0% | 58.0% | 87% | 33% | 87% | 0% |
| kPCA | 0.938 | 22.0% | 60.0% | 87% | 33% | 87% | 9% |

χ² 가 지배한다. 게다가 OC-SVM·kPCA 는 FPR 목표(≤10%)를 지키지도 못한다(22%).

이유 3가지

1. χ² 는 정합(matched) 검출기다. H0 에서 잔차 분포가 알려져 있고(단위분산·가산),
   이 문제의 OOD 신호는 주로 크기 변화다. Σz² 는 가우시안 스케일 변화의 (근사)
   충분통계량이라, 분포 무가정 학습기가 원리적으로 이길 수 없다.
2. 피처가 사실상 1차원 — 8개 중 6개가 같은 스칼라 신호의 크기 통계량이라 Σz² 가
   놓치는 방향이 없다. (3피처로 줄여도 OC-SVM FPR 30% — 나아지지 않는다.)
3. 커널 스코어의 초과가 시간적으로 뭉친다 — 잔차의 자기상관(+0.40)을 그대로 물려받아
   한 비행에서 초과가 연속 발생 → N-of-M 규칙이 정상 비행에서 발화 → FPR 22%.
   이는 성능 저하가 아니라 보장 실패다.

이것이 말하지 않는 것 (반증 조건)

- 단일 문제 · 단일 명목모델 · 크기형 OOD 4계열 · calibration 50비행 · 8피처 · γ 1개 값에서의 결과다.
- OC-SVM 은 H0 가 미지이고 이상이 모양/방향 변화인 다변량 문제를 위한 도구다.
  여기서는 H0 가 알려져 있고 이상이 크기 변화 — 가장 불리한 조건이다.
- 계획의 커널 채택 근거는 "고장 데이터 없는 one-class + 소표본"이었다. 그 근거는
  *음성 샘플의 부재*에 관한 것이고, 명목 모델이라는 물리 사전정보를 계산에 넣지 않았다.

반증 실험 (다음 단계 후보)

- (a) 같은 분산에서 모양만 바뀌는 OOD 를 만들어 OC-SVM vs χ² 재비교
- (b) 크기 통계량을 빼고 모양 피처(왜도·첨도·자기상관·채널 간 비율)만 사용
- (c) 6-DOF 자세 채널처럼 교차 구조가 있는 피처 공간

함의: 이 문제에서 주역은 분류기가 아니라 conformal calibration 계층이다.
CP 는 χ² 에게도 유한표본 FPR 보장을 준다(비행 단위 6.0% ≤ α). 커널은 교체 가능한
부품이고, 기여는 보증 메커니즘에 있다.

### 남은 과제

1. 잔차 백색화 (AR/MA) — L1 의 이론적 exact 보장 복원.
2. N-of-M 규칙 전체를 하나의 conformal 객체로 calibration — 계획의 연구 novelty 1위 (비행 단위 FPR).
3. MC Dropout(Step 5) 기준선 비교 — torch 필요, 미완.
4. 저각·강풍 계열 탐지 개선 — 피처 설계 또는 6-DOF 모델 필요.

---

## 1. 왜 Step 5.5 인가 (Motivation)

### 1.1 Step 5 의 한계 — 정직한 진단

Step 5 는 MC Dropout 기반으로 OOD 를 탐지한다:

```
GRU 예측 → MC Dropout σ (1000회 샘플링) → σ 비율 임계값 → Safe-Logic
```

문제점 3가지:
1. 통계 보장 없음 — σ 임계값을 어떻게 정하든 "오탐률 ≤ α"가 보장되지 않는다. 임계값은 경험적 선택.
2. 예측 모델에 OOD 탐지가 내장 — 예측이 틀리면(모델 오류) 이상 탐지도 같이 틀어진다. 관심사 분리(separation of concerns) 부재.
3. 물리 정보 미활용 — Step 2 의 EKF/UKF 가 이미 "이상이 있으면 혁신(innovation)이 커진다"는 정보를 만들고 있는데 버려진다.

### 1.2 Step 5.5 의 핵심 아이디어

"물리 필터가 남긴 혁신 위에 커널 기반 이상 탐지를 얹고, CP(Conformal Prediction)로 통계 보장을 단다."

```
Step 2 (이미 존재)          Step 5.5 (신규)
EKF/UKF 혁신 시퀀스  →   [χ² 검정: 물리가 잡는 오류]  → exact 보장 (CP 불필요)
                            ↓ 비가우시안 잔차
                         [커널 one-class: OC-SVM/SVDD/KDE/kPCA]  → 분포 무가정
                            ↓ CP calibration
                         [임계값: P(오탐) ≤ α 유한표본 보장]  → Safe-Logic 2.0
```

커널 적용의 베스트 포지션 근거:
- 고장 데이터가 없는 one-class 문제 + 실측 몇 회뿐인 소표본 = 커널의 홈그라운드
- EKF/UKF 혁신은 H0 에서 영평균 백색 가우시안 → 물리적으로 검증 가능한 입력
- MC Dropout 대체가 아니라 보완 — 계층적 인증 구조 (A5 인사이트, 2026-08-04 합의)

---

## 2. 사전 자산 (Pre-requisites)

| 자산 | 상태 | 필요 작업 |
|:--|:--|:--|
| Step 1.5 시뮬 데이터 (200 flights) | ✅ 존재 | 정상 비행 / OOD 비행 분리 |
| Step 2 EKF/UKF 구현 | ✅ 존재 | 혁신 시퀀스 저장 로직 추가 필요 (현재 상태만 저장) |
| Step 5 MC Dropout 파이프라인 | ✅ 존재 | 비교 기준선으로 사용 |
| sklearn (OneClassSVM, KernelDensity, KernelPCA) | ✅ requirements 에 포함 예상 | 확인 필요 |

⚠️ 전제 작업: Step 2 노트북이 혁신(innovation) $y - h(\hat{x}^-)$ 를 저장하지 않는다. Step 5.5 를 위해:
- (a) Step 2 를 수정해 혁신 시퀀스를 `.npy` 로 저장하거나
- (b) Step 5.5 내부에서 필터를 재실행해 혁신을 추출 (독립성 유지, 권장)

---

## 3. 노트북 구조 (Cell 구성 초안)

### Part A. 이론 (MD 셀)

| Cell | 내용 | 핵심 레퍼런스 |
|:--|:--|:--|
| A1 | 혁신의 통계적 성질: H0 에서 영평균 백색 가우시안, 공분산 = HPHᵀ+R | Kalman filter 표준 |
| A2 | χ² 검정으로 잡을 수 있는 것 vs 없는 것 (비가우시안, 센서 포화, 모델 오차) | Mehra (1971) innovations fault detection |
| A3 | ML2 복습: OC-SVM / SVDD / KDE / kPCA — one-class 이상 탐지 4종 | Schölkopf 2001, Tax & Duin 2004, Bishop PRML |
| A4 | CP 기초: split conformal, nonconformity score, 유한표본 보장 P(오탐) ≤ α | Vovk 2005, Angelopoulos & Bates 2021 |
| A5 | 계층적 인증: "물리로 잡을 수 있으면 exact, 못 잡으면 CP" | 8/4 논의 구조 |

### Part B. 구현 (코드 셀)

| Cell | 내용 | sklearn API |
|:--|:--|:--|
| B1 | 혁신 시퀀스 추출 함수 (필터 재실행 또는 저장 파일 로드) | — |
| B2 | 피처 구성: 혁신 윈도우 통계 (평균/분산/최대) + 원시 센서 피처 | — |
| B3 | 정상 비행 train/calibration/test 분할 (비행 단위, 비행 내 샘플 아님!) | train_test_split |
| B4 | OC-SVM 학습 (RBF, ν 설정) | `sklearn.svm.OneClassSVM` |
| B5 | SVDD — OC-SVM(RBF) 과의 관계 설명 + (선택) 직접 구현 | OC-SVM ≈ SVDD (커널 공간) |
| B6 | KDE 밀도 임계 | `sklearn.neighbors.KernelDensity` |
| B7 | kPCA 재구성 오차 | `sklearn.decomposition.KernelPCA` |
| B8 | CP calibration 함수: calibration set 스코어의 (1−α)(1+1/n) quantile | 직접 구현 (~15줄) |
| B9 | OOD 비행에서 FPR/TPR 평가 + MC Dropout (Step 5) 과 비교 | sklearn metrics |
| B10 | 비행 단계별 분석: 부스트/코스트/아포지/강하 — 단계별 calibration | 직접 구현 |
| B11 | Safe-Logic 2.0: 커널+CP 스코어로 abort 판단 시연 | — |

---

## 4. 실험 설계 (검증 프로토콜)

### 4.1 데이터 분할 원칙 (중요)

```
정상 비행 150   →  train 100 + calibration 25 + test 25   (비행 단위 분할!)
OOD 비행 50     →  test 전용
```

함정 주의: 로켓 비행은 한 비행 내 샘플이 강한 상관관계를 가진다. 샘플 단위가 아니라 비행 단위로 분할해야 교환성 가정과 데이터 누수(leakage)를 피할 수 있다. (이것이 CP 적용 시 가장 흔한 실수)

### 4.2 평가 지표

| 지표 | 정의 | 목표 |
|:--|:--|:--|
| FPR (오탐률) | 정상 샘플 중 이상 판정 비율 | CP 보장: ≤ α |
| TPR (재현율) | OOD 샘플 중 탐지 비율 | 높을수록 좋음 (보장 아님) |
| σ 비교 | MC Dropout σ 비율 vs 커널 스코어 | 커널 방식이 더 깔끔한 분리 보이는지 |
| 단계별 FPR | 각 비행 단계에서의 FPR | 단계별 보장 달성 여부 |

### 4.3 핵심 질문 (실험이 답해야 할 것)

1. 커널+CP 가 MC Dropout σ 임계값 대비 동일 FPR 수준에서 더 높은 TPR을 주는가?
2. CP 보장 P(FPR ≤ α) 가 시뮬 데이터에서 실제로 성립하는가? (marginal)
3. 단계별 분포 이동(부스트 vs 강하)에서 marginal 보장이 어떻게 무너지고, 단계별 calibration 이 어떻게 복구하는가?
4. innovations 피처 vs 원시 센서 피처 — 어느 쪽이 더 잘 분리하는가?

---

## 5. 비교 대상 (실험 매트릭스)

| 방법 | 스코어 | 보장 | 구현 난이도 |
|:--|:--|:--|:--|
| MC Dropout σ (Step 5 기존) | 예측 σ | ❌ 없음 | 낮음 |
| OC-SVM 마진 | decision function | CP 로 보장 | 낮음 (sklearn 1줄) |
| SVDD 거리 | 중심 거리 | CP 로 보장 | 중간 (직접 구현) |
| KDE | −log 밀도 | CP 로 보장 | 낮음 |
| kPCA | 재구성 오차 | CP 로 보장 | 낮음 |
| χ² + 커널 + CP (하이브리드) | 물리/ML 결합 | 이중 보장 | 중간 — 본 Step 의 목적 |

---

## 6. 교환성 위반 대응 (정직한 한계)

CP 의 보장은 교환성(exchangeability) 위에서 성립하는데, 비행은 시계열 + 단계별 분포 이동:

1. 비행 단위 분할 — 교환성의 기본 단위를 비행으로 (샘플이 아니라)
2. 단계별(phase-conditional) calibration — 부스트/코스트/아포지/강하 각각 따로 calibration → 단계별 FPR 보장 (8/4 A1 인사이트의 구현)
3. (선택) ACI (Adaptive Conformal Inference) — Gibbs & Candès, 온라인 스트림에서 임계값 적응 — Step 5.5 에서는 소개만 하고 실제 구현은 future work

문서화할 한계:
- 시뮬 데이터는 RocketPy 기반 → 실측과의 sim-to-real shift 존재 (보장은 calibration 분포 안에서만)
- OOD 정의가 시뮬 파라미터 범위 이탈로 제한됨 (mass/Cd/wind/launch_angle) — 실세계 고장 모드와 다를 수 있음

---

## 7. 산출물 / 완료 기준 (Definition of Done)

- [ ] `step5.5-kernel-anomaly-detection.ipynb` — 이론 MD + 실행 코드 완성
- [ ] 혁신 추출 파이프라인 (Step 2 재실행 또는 저장 로드)
- [ ] CP calibration 함수 구현 및 단위 검증 (합성 데이터로 quantile 보장 확인)
- [ ] 실험 매트릭스 6종 실행 결과 표 (FPR/TPR/단계별)
- [ ] MC Dropout vs 커널+CP 비교 플롯
- [ ] "혁신 = 비가우시안 잔차" 시각화 (이상 비행에서 혁신이 어떻게 커지는지)
- [ ] README 테마 표에 Step 5.5 추가 ("통계적 보장" 테마)

---

## 8. 참고 문헌

- Schölkopf, Platt, Shawe-Taylor, Smola, Williamson (2001). *Estimating the Support of a High-Dimensional Distribution*. Neural Computation. — OC-SVM
- Tax & Duin (2004). *Support Vector Data Description*. Machine Learning. — SVDD
- Vovk, Gammerman, Shafer (2005). *Algorithmic Learning in a Random World*. — CP
- Angelopoulos & Bates (2021). *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. arXiv 2107.07511
- Gibbs & Candès (2021). *Adaptive Conformal Inference Under Distribution Shift*. NeurIPS. — ACI
- Mehra (1971). *On the Identification of Variances and Adaptive Kalman Filtering*. IEEE TAC. — innovations fault detection
- Feldman et al. (2025). *Conformal Safety Monitoring for Flight Testing*. — 항공 안전 모니터링 CP 적용 사례

---

## 9. 열린 질문 (미결정 — 구현 시 결정)

1. ν (OC-SVM) vs CP 임계값 — 둘 다 FPR 을 조절하는데, ν 를 사전 고정할지 CP 로 대체할지? (권장: ν 는 느슨하게, CP 가 최종 임계 결정)
2. 피처 차원 — 혁신 윈도우 통계만 쓸지 (저차원, 해석 가능) vs 원시 센서 포함 (고차원, 성능) — 4.3-Q4 로 실험 결정
3. SVDD 직접 구현 — OC-SVM(RBF) 과 SVDD 는 밀접하나 완전 동일하지 않음. 교육적으로 SVDD 의 구(sphere) 해석을 직접 구현해 보여줄지 — 교육 가치 vs 구현 부담 트레이드오프
4. χ² 검정과 커널의 결합 방식 — 혁신이 가우시안이면 χ² 로 먼저 필터링하고, 비가우시안 잔차만 커널 입력으로? 아니면 커널이 전체 혁신을 받고 χ² 는 별도 채널로? (실험 4.3-Q4 와 연결)
