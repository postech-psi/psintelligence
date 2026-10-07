# PSIntelligence 101: PSI 에비오닉스를 위한 AI 파이프라인
---

## 프로젝트 개요

본 프로젝트는 로켓 비행 중 실시간 아포지 (최고 고도) 예측을 목표로 하는 엔지니어링 AI 파이프라인 구축 교육 과정입니다. 단순한 머신러닝 모델 학습을 넘어, 실제 에비오닉스 (항공전자장비) 에 탑재 가능한 안전하고 신뢰할 수 있는 AI 시스템을 설계하는 데 중점을 둡니다.

### 핵심 테마

| 테마 | 설명 | 적용 단계 |
|------|------|----------|
| 물리 법칙 준수 | AI 예측이 에너지 보존, 운동 방정식 등 물리 법칙을 위반하지 않도록 제약 | Step 3 (PINN) |
| 시계열 맥락 학습 | 과거 비행 이력 (자이로 요동, 틸트 등) 을 기반으로 미래 예측 | Step 4 (GRU) |
| 불확실성 정량화 | AI 가 "자신의 무지"를 인식하고, 위험 상황에서 안전모드 진입 | Step 5 (UQ) |
| 온보드 배포 | 제한된 메모리/전력의 비행 컴퓨터에서 실시간 추론 가능 | Step 6 (Mamba+ONNX) |
| 안전성 우선 | 정확도보다 신뢰구간과 Fail-safe 로직을 우선시하는 엔지니어링 접근 | Step 5-6 |

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
  │         └─ 비선형 필터 (EKF/UKF)
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

필요 환경: 기본 `requirements.txt` 로 충분합니다. `data/simulated/` 가 동봉되어 있어
RocketPy 없이 Step 2~6 을 실행할 수 있습니다(데이터를 직접 재생성할 때만 RocketPy 필요).

> 동봉 데이터 스키마와 알려진 결함은 `data/SCHEMA.md` 를 참조하세요.

---


### [Step 2] EKF & UKF: 비선형 시스템 확장

목표: 비선형 로켓 역학 (추력, 항력, 중력) 을 고려한 상태 추정

핵심 내용:
- EKF: 자코비안 행렬 수동 유도
- UKF: Unscented Transform 로 비선형성 근사
- Bella Lui 실제 비행 데이터 검증

---

### [Step 3] PINN: 물리 법칙을 지키는 인공지능

목표: 데이터 부족 환경에서도 물리 법칙을 위반하지 않는 예측

핵심 내용:
- Hard Constraint: 출력층에 물리 식 내장 (아포지 = 현재고도 + v²/2g × r)
- Soft Constraint: Loss 함수에 물리 항 추가 (단조 감소, 에너지 보존)
- Stability Masking: 아포지 근처 (v→0) 수치 불안정성 제거

논문 기반:
- Raissi et al. (2019). *Physics-Informed Neural Networks*. Journal of Computational Physics.

---

### [Step 4] GRU: 과거의 결함을 기억하는 시계열 추정기

목표: 발사 직후의 핀 틀어짐, 자이로 요동 등 과거 이력이 현재 아포지에 미치는 영향 학습

핵심 내용:
- Sliding Window: (Batch, Seq_Len, Features) 3D 텐서 전처리
- Hidden State: 과거 100 스텝 (2 초) 비행 이력 압축 저장
- Physics Wrapper: Step 3 의 Hard Constraint 구조 재활용

논문 기반:
- Chung et al. (2014). *Empirical Evaluation of Gated Recurrent Neural Networks*. arXiv:1412.3555.
- Gers et al. (2000). *Learning to Forget: Continual Prediction with LSTM*. Neural Computation.

---

### [Step 5] Uncertainty Quantification: AI 의 자신감 측정

목표: AI 가 "모르는 상황 (OOD)"을 감지하고 안전모드로 진입

핵심 내용:
- Aleatoric Uncertainty: 센서 노이즈 등 데이터 고유 불확실성 (줄일 수 없음)
- Epistemic Uncertainty: 모델 무지 (데이터 추가 시 감소, OOD 감지용)
- MC Dropout: 추론 시 Dropout 켜고 T 번 샘플링하여 분산 계산
- 신뢰구간 기반 사출 로직: `현재고도 ≥ 예측아포지 - 2σ` 일 때만 사출 승인

논문 기반:
- Gal & Ghahramani (2016). *Dropout as a Bayesian Approximation*. ICML.
- Kendall & Gal (2017). *What Uncertainties Do We Need in Bayesian Deep Learning?* NIPS.

---

### [Step 5.5] 커널 기반 이상 탐지 + Conformal Prediction

목표: 물리로 잡을 수 있는 오류는 exact 보장으로, 못 잡는 것은 distribution-free 보장으로

핵심 내용:
- 명목 물리모델 잔차 — 명목 질량·Cd·발사각으로 동역학을 적분해 비행체 이상을 드러냄
  (Step 2 의 기구학 추적기는 측정 가속도를 입력으로 써서 파라미터 이탈을 못 잡음 — 실측 1.00x)
- L1 χ² 검정 — 잔차의 통계량. 이론 임계는 백색성 위반으로 성립하지 않아(자기상관 +0.40)
  calibration 창에서 재설정 (Mehra 1971 계열, 잔차 백색화는 남은 과제)
- L2 OC-SVM(RBF) + CP — 비가우시안·모델오차 잔차를 분포 무가정으로 처리
- 비행 단위 CP calibration — 창이 아니라 비행 단위. 창들은 상관되어 있어 실효 표본수가 비행 수에 불과
- Safe-Logic 2.0 연동 — L0/L1 결정적 우선, L2 는 N-of-M + 히스테리시스 (`safe_logic.py`)

실측 결과 (정상 200편 + OOD 50편, FPR 목표 ≤ 10%, 비행 단위):

| 방법 | FPR(비행) | TPR(비행) |
|:--|:--:|:--:|
| L1 χ² (보정임계 + N-of-M) | 6.0% | 68.0% |
| OC-SVM + CP | 6.0% | 54.0% |
| KDE + CP | 6.0% | 60.0% |
| kPCA + CP | 4.0% | 54.0% |

계열별로는 질량 93~100% · 항력 67~100% · 저각 11~56% · 강풍 0~9% 로 탐지가 계열 선택적입니다.

> ⚠️ 검출기 비교는 집계 규칙에 의존합니다. 위 표의 L1 은 N-of-M, 커널은 95분위 집계라
> 규칙이 다릅니다. 같은 N-of-M 규칙으로 통일하면 L1 χ² 6.0%/68.0% vs OC-SVM 22.0%/56.0%.
> 그러나 동일 FPR(10%) + 95분위 집계에서는 χ² 58% vs OC-SVM(크기 피처) 66% 로 뒤집힙니다.
>
> 더 중요한 것은 이상 유형과 피처의 정합입니다 (반증 실험 (a), 모양형 OOD 50편):
>
> | 이상 유형 | χ² (크기 통계) | OC-SVM (모양 피처) |
> |:--|:--:|:--:|
> | 크기 변화 (파라미터 이탈) | 58% | 18% |
> | 모양 변화 (분산 동일) | 18% | 98% |
>
> 모양형 OOD는 고도 잡음의 표본 σ를 1.5 m로 강제하고 모양만 바꾼 것입니다
> (검증: 창 Σz² 중앙 비 0.98x = 분산 동일, lag-1 자기상관 −0.03 → +0.68).
> → 검출기가 아니라 "피처 ∩ 이상 유형"이 지배하며, 두 채널은 실패 모드가 겹치지 않아
> 병치하는 것이 정답입니다. 상세는 `PLAN_step5.5.md` 참조.

L1 은 이론 임계를 쓸 수 없습니다. χ² 이론 임계는 잔차가 백색일 때만 성립하는데, 명목 모델
잔차의 lag-1 자기상관이 +0.40 으로 유의합니다(실측). 이론 임계로는 창 FPR 이 34.4% 였고,
임계를 calibration 창에서 재설정해 11.7% 로 맞췄습니다 (잔차 백색화는 남은 과제).

입력 생성:
```bash
python scripts/generate_ood.py      # 정상 범위를 벗어난 OOD 50편
python scripts/model_filter.py      # 명목 물리모델 잔차 → residuals.npz
```

논문 기반:
- Schölkopf et al. (2001). *Estimating the Support of a High-Dimensional Distribution*. Neural Computation. — OC-SVM
- Tax & Duin (2004). *Support Vector Data Description*. Machine Learning. — SVDD
- Vovk et al. (2005). *Algorithmic Learning in a Random World*. — Conformal Prediction
- Angelopoulos & Bates (2021). *A Gentle Introduction to Conformal Prediction*. arXiv 2107.07511.
- Mehra (1971). *On the Identification of Variances and Adaptive Kalman Filtering*. IEEE TAC.

> ⚠️ 정직한 한계: 시뮬 기반이라 CP 보장은 calibration 분포 안에서만 유효합니다. OOD 정의도
> 파라미터 범위 이탈로 협소하고, 위상은 데이터가 아포지에서 잘려 2개(부스트/관성)뿐입니다.
> 자세한 내용은 노트북 요약 절 참조.

---

### [Step 6] Quantize & Edge Deployment: 비행 컴퓨터를 위한 AI 경량화

목표: 제한된 온보드 자원 에서 실시간 추론

핵심 내용:
- Quantization: FP32 → INT8 양자화 (모델 크기 75% 감소)
- ONNX Export: PyTorch → C++ 배포용 포맷 변환
- 속도 벤치마크: 추론 시간 < 20ms (50Hz 요구사항 충족)

산출물:
- `rocket_apogee_model.onnx`: 온보드 배포용 모델

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
winget install mamba-org.micromamba
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
curl -LsSf https://astral.sh/uv/install.sh | sh

# 기본 설치 — 이 한 줄이면 Step 2~6 을 모두 실행할 수 있습니다
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

기본 경로 — 데이터는 이미 들어 있습니다. RocketPy 설치 없이 바로 시작하세요.

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
학습된 모델이 이미 포함되어 있어, Step 2~6 을 그대로 실행할 수 있습니다.

데이터를 직접 만들어보고 싶다면 두 가지 경로가 있습니다.

```bash
# (A) 자체 시뮬레이터 — RocketPy 불필요, 표준 라이브러리만 (200편 약 2.3초)
python scripts/generate_dataset.py
python scripts/verify_dataset.py --data-dir data/generated

# (B) 원본 경로 — RocketPy 로 step1.5 노트북 실행
uv pip install "rocketpy>=1.11" "seaborn>=0.13"   # 선택 의존성
```

데이터를 수정하거나 재생성한 뒤에는 무결성 검증을 실행하세요.

```bash
python scripts/verify_dataset.py                 # 정본 검사 (21항목)
python scripts/verify_dataset.py --data-dir <경로>  # 생성 데이터 검사
```

> 스키마·알려진 결함·재현 모델: [`data/SCHEMA.md`](data/SCHEMA.md)


---

## 교육적 목표

본 프로젝트는 단순한 코드 구현을 넘어 다음과 같은 엔지니어링 사고방식을 함양합니다:

1. 점진적 복잡도: 선형 (Step 1) → 비선형 (Step 2) → AI (Step 3-6) 로 단계적 학습
2. 물리 기반 AI: 데이터만 믿지 않고 물리 법칙을 제약 조건으로 활용
3. 안전성 우선: 정확도보다 불확실성 인식과 Fail-safe 로직을 우선시
4. 배포 고려: 연구실 PC 가 아닌 온보드 컴퓨터 제약을 고려한 모델 설계
5. 논문 기반: 각 Step 의 이론적 배경을 유명 논문에서 발췌하여 학술적 엄밀함 확보

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

1. Step 순서 준수: 각 Step 은 이전 Step 의 산출물을 사용하므로 순차적 실행 필요
2. 데이터 생성: Step 1.5 시뮬레이션 데이터 생성이 선행되어야 함
3. GPU 권장: Step 4-6 은 GPU 가 있으면 학습 속도가 5-10 배 빠름
4. 메모리 관리: Step 6 ONNX Export 시 모델 크기가 급증할 수 있으므로 주의

---

## 문의

- 개발자: 포항공과대학교 기계공학과 21학번 이승원
- 이메일: dongdong0615@postech.ac.kr / sungwon.lee.2002@gmail.com

---


> PSI 후배들이 AI를 접하고 공부하는데 도움이 되기를 바랍니다.
