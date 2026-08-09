# Step 5.5 계획안: 커널 기반 이상 탐지 — 필터 혁신(Innovations) 위에 통계적 보장 얹기

**버전:** v0.1 (초안)
**작성일:** 2026-08-09
**위치:** `step5-UQ.ipynb` (Step 5) 와 `step6-Mamba&Deploy.ipynb` (Step 6) 사이
**파일명 후보:** `step5.5-kernel-anomaly-detection.ipynb`

---

## 1. 왜 Step 5.5 인가 (Motivation)

### 1.1 Step 5 의 한계 — 정직한 진단

Step 5 는 MC Dropout 기반으로 OOD 를 탐지한다:

```
GRU 예측 → MC Dropout σ (1000회 샘플링) → σ 비율 임계값 → Safe-Logic
```

문제점 3가지:
1. **통계 보장 없음** — σ 임계값을 어떻게 정하든 "오탐률 ≤ α"가 보장되지 않는다. 임계값은 경험적 선택.
2. **예측 모델에 OOD 탐지가 내장** — 예측이 틀리면(모델 오류) 이상 탐지도 같이 틀어진다. 관심사 분리(separation of concerns) 부재.
3. **물리 정보 미활용** — Step 2 의 EKF/UKF 가 이미 "이상이 있으면 혁신(innovation)이 커진다"는 정보를 만들고 있는데 버려진다.

### 1.2 Step 5.5 의 핵심 아이디어

**"물리 필터가 남긴 혁신 위에 커널 기반 이상 탐지를 얹고, CP(Conformal Prediction)로 통계 보장을 단다."**

```
Step 2 (이미 존재)          Step 5.5 (신규)
EKF/UKF 혁신 시퀀스  →   [χ² 검정: 물리가 잡는 오류]  → exact 보장 (CP 불필요)
                            ↓ 비가우시안 잔차
                         [커널 one-class: OC-SVM/SVDD/KDE/kPCA]  → 분포 무가정
                            ↓ CP calibration
                         [임계값: P(오탐) ≤ α 유한표본 보장]  → Safe-Logic 2.0
```

**커널 적용의 베스트 포지션 근거:**
- 고장 데이터가 없는 one-class 문제 + 실측 몇 회뿐인 소표본 = 커널의 홈그라운드
- EKF/UKF 혁신은 H0 에서 영평균 백색 가우시안 → 물리적으로 검증 가능한 입력
- MC Dropout 대체가 아니라 **보완** — 계층적 인증 구조 (A5 인사이트, 2026-08-04 합의)

---

## 2. 사전 자산 (Pre-requisites)

| 자산 | 상태 | 필요 작업 |
|:--|:--|:--|
| Step 1.5 시뮬 데이터 (200 flights) | ✅ 존재 | 정상 비행 / OOD 비행 분리 |
| Step 2 EKF/UKF 구현 | ✅ 존재 | **혁신 시퀀스 저장 로직 추가 필요** (현재 상태만 저장) |
| Step 5 MC Dropout 파이프라인 | ✅ 존재 | 비교 기준선으로 사용 |
| sklearn (OneClassSVM, KernelDensity, KernelPCA) | ✅ requirements 에 포함 예상 | 확인 필요 |

**⚠️ 전제 작업:** Step 2 노트북이 혁신(innovation) $y - h(\hat{x}^-)$ 를 저장하지 않는다. Step 5.5 를 위해:
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
| B8 | **CP calibration 함수**: calibration set 스코어의 (1−α)(1+1/n) quantile | 직접 구현 (~15줄) |
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

**함정 주의:** 로켓 비행은 한 비행 내 샘플이 강한 상관관계를 가진다. **샘플 단위가 아니라 비행 단위로 분할**해야 교환성 가정과 데이터 누수(leakage)를 피할 수 있다. (이것이 CP 적용 시 가장 흔한 실수)

### 4.2 평가 지표

| 지표 | 정의 | 목표 |
|:--|:--|:--|
| FPR (오탐률) | 정상 샘플 중 이상 판정 비율 | **CP 보장: ≤ α** |
| TPR (재현율) | OOD 샘플 중 탐지 비율 | 높을수록 좋음 (보장 아님) |
| σ 비교 | MC Dropout σ 비율 vs 커널 스코어 | 커널 방식이 더 깔끔한 분리 보이는지 |
| 단계별 FPR | 각 비행 단계에서의 FPR | 단계별 보장 달성 여부 |

### 4.3 핵심 질문 (실험이 답해야 할 것)

1. 커널+CP 가 MC Dropout σ 임계값 대비 **동일 FPR 수준에서 더 높은 TPR**을 주는가?
2. CP 보장 P(FPR ≤ α) 가 시뮬 데이터에서 **실제로 성립**하는가? (marginal)
3. 단계별 분포 이동(부스트 vs 강하)에서 **marginal 보장이 어떻게 무너지고**, 단계별 calibration 이 어떻게 복구하는가?
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
| **χ² + 커널 + CP (하이브리드)** | 물리/ML 결합 | **이중 보장** | 중간 — 본 Step 의 목적 |

---

## 6. 교환성 위반 대응 (정직한 한계)

CP 의 보장은 교환성(exchangeability) 위에서 성립하는데, 비행은 시계열 + 단계별 분포 이동:

1. **비행 단위 분할** — 교환성의 기본 단위를 비행으로 (샘플이 아니라)
2. **단계별(phase-conditional) calibration** — 부스트/코스트/아포지/강하 각각 따로 calibration → 단계별 FPR 보장 (8/4 A1 인사이트의 구현)
3. **(선택) ACI (Adaptive Conformal Inference)** — Gibbs & Candès, 온라인 스트림에서 임계값 적응 — Step 5.5 에서는 소개만 하고 실제 구현은 future work

**문서화할 한계:**
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

1. **ν (OC-SVM) vs CP 임계값** — 둘 다 FPR 을 조절하는데, ν 를 사전 고정할지 CP 로 대체할지? (권장: ν 는 느슨하게, CP 가 최종 임계 결정)
2. **피처 차원** — 혁신 윈도우 통계만 쓸지 (저차원, 해석 가능) vs 원시 센서 포함 (고차원, 성능) — 4.3-Q4 로 실험 결정
3. **SVDD 직접 구현** — OC-SVM(RBF) 과 SVDD 는 밀접하나 완전 동일하지 않음. 교육적으로 SVDD 의 구(sphere) 해석을 직접 구현해 보여줄지 — 교육 가치 vs 구현 부담 트레이드오프
4. **χ² 검정과 커널의 결합 방식** — 혁신이 가우시안이면 χ² 로 먼저 필터링하고, 비가우시안 잔차만 커널 입력으로? 아니면 커널이 전체 혁신을 받고 χ² 는 별도 채널로? (실험 4.3-Q4 와 연결)
