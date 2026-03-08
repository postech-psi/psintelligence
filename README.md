# PSIntelligence 101: PSI 에비오닉스를 위한 AI 파이프라인
---

## 프로젝트 개요

본 프로젝트는 **로켓 비행 중 실시간 아포지 (최고 고도) 예측**을 목표로 하는 엔지니어링 AI 파이프라인 구축 교육 과정입니다. 단순한 머신러닝 모델 학습을 넘어, **실제 에비오닉스 (항공전자장비) 에 탑재 가능한 안전하고 신뢰할 수 있는 AI 시스템**을 설계하는 데 중점을 둡니다.

### 핵심 테마

| 테마 | 설명 | 적용 단계 |
|------|------|----------|
| **물리 법칙 준수** | AI 예측이 에너지 보존, 운동 방정식 등 물리 법칙을 위반하지 않도록 제약 | Step 3 (PINN) |
| **시계열 맥락 학습** | 과거 비행 이력 (자이로 요동, 틸트 등) 을 기반으로 미래 예측 | Step 4 (GRU) |
| **불확실성 정량화** | AI 가 "자신의 무지"를 인식하고, 위험 상황에서 안전모드 진입 | Step 5 (UQ) |
| **온보드 배포** | 제한된 메모리/전력의 비행 컴퓨터에서 실시간 추론 가능 | Step 6 (Mamba+ONNX) |
| **안전성 우선** | 정확도보다 신뢰구간과 Fail-safe 로직을 우선시하는 엔지니어링 접근 | Step 5-6 |

---

## 프로젝트 로드맵 (Step 1-6)

```
Step 1 → Step 2 → Step 3 → Step 4 → Step 5 → Step 6
  │         │         │         │         │         │
  │         │         │         │         │         └─ 경량화 + ONNX
  │         │         │         │         └─ 불확실성 정량화 (MC Dropout)
  │         │         │         └─ 시계열 모델링 (GRU)
  │         │         └─ 물리 법칙 통합 (PINN)
  │         └─ 비선형 필터 (EKF/UKF)
  └─ 선형 필터 (Kalman)
```

---

## 각 Step 상세 설명

### [Step 1] Linear Kalman Filter: 상태 추정의 기초

**목표:** 센서 노이즈가 제거된 로켓 상태 (고도, 속도) 추정

**핵심 내용:**
- 칼만 필터의 5 개 방정식 (예측/업데이트) 구현
- 공분산 행렬 (P, Q, R) 의 물리적 의미 이해
- 1 차원 수직 비행 모델에 적용

**산출물:**
- `101-1 기초확률론 & 선형칼만필터.pdf`: pdf 자료

---

### [Step 2] EKF & UKF: 비선형 시스템 확장

**목표:** 비선형 로켓 역학 (추력, 항력, 중력) 을 고려한 상태 추정

**핵심 내용:**
- EKF: 자코비안 행렬 수동 유도
- UKF: Unscented Transform 로 비선형성 근사
- Bella Lui 실제 비행 데이터 검증

---

### [Step 3] PINN: 물리 법칙을 지키는 인공지능

**목표:** 데이터 부족 환경에서도 물리 법칙을 위반하지 않는 예측

**핵심 내용:**
- **Hard Constraint**: 출력층에 물리 식 내장 (아포지 = 현재고도 + v²/2g × r)
- **Soft Constraint**: Loss 함수에 물리 항 추가 (단조 감소, 에너지 보존)
- **Stability Masking**: 아포지 근처 (v→0) 수치 불안정성 제거

**논문 기반:**
- Raissi et al. (2019). *Physics-Informed Neural Networks*. Journal of Computational Physics.

---

### [Step 4] GRU: 과거의 결함을 기억하는 시계열 추정기

**목표:** 발사 직후의 핀 틀어짐, 자이로 요동 등 과거 이력이 현재 아포지에 미치는 영향 학습

**핵심 내용:**
- **Sliding Window**: (Batch, Seq_Len, Features) 3D 텐서 전처리
- **Hidden State**: 과거 100 스텝 (2 초) 비행 이력 압축 저장
- **Physics Wrapper**: Step 3 의 Hard Constraint 구조 재활용

**논문 기반:**
- Chung et al. (2014). *Empirical Evaluation of Gated Recurrent Neural Networks*. arXiv:1412.3555.
- Gers et al. (2000). *Learning to Forget: Continual Prediction with LSTM*. Neural Computation.

---

### [Step 5] Uncertainty Quantification: AI 의 자신감 측정

**목표:** AI 가 "모르는 상황 (OOD)"을 감지하고 안전모드로 진입

**핵심 내용:**
- **Aleatoric Uncertainty**: 센서 노이즈 등 데이터 고유 불확실성 (줄일 수 없음)
- **Epistemic Uncertainty**: 모델 무지 (데이터 추가 시 감소, OOD 감지용)
- **MC Dropout**: 추론 시 Dropout 켜고 T 번 샘플링하여 분산 계산
- **신뢰구간 기반 사출 로직**: `현재고도 ≥ 예측아포지 - 2σ` 일 때만 사출 승인

**논문 기반:**
- Gal & Ghahramani (2016). *Dropout as a Bayesian Approximation*. ICML.
- Kendall & Gal (2017). *What Uncertainties Do We Need in Bayesian Deep Learning?* NIPS.

---

### [Step 6] Quantize & Edge Deployment: 비행 컴퓨터를 위한 AI 경량화

**목표:** 제한된 온보드 자원 에서 실시간 추론

**핵심 내용:**
- **Quantization**: FP32 → INT8 양자화 (모델 크기 75% 감소)
- **ONNX Export**: PyTorch → C++ 배포용 포맷 변환
- **속도 벤치마크**: 추론 시간 < 20ms (50Hz 요구사항 충족)

**산출물:**
- `rocket_apogee_model.onnx`: 온보드 배포용 모델

---

## 환경 설정 (micromamba & uv)

본 프로젝트는 **재현성 확보**와 **의존성 충돌 방지**를 위해 `micromamba` (가상환경) 와 `uv` (패키지 관리) 를 사용합니다.

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

```bash
# uv 설치 (pip 대체제) < 훨씬 빠르고 가벼움
curl -LsSf https://astral.sh/uv/install.sh | sh

# 의존성 설치 (pyproject.toml 기반)
uv pip install -r requirements.txt

# 또는 개별 설치
uv pip install torch torchvision torchaudio
uv pip install numpy pandas matplotlib scikit-learn
uv pip install jupyterlab ipywidgets
uv pip install mamba-ssm onnx onnxruntime
```

### 4. Jupyter Kernel 등록

```bash
# psintel 환경을 Jupyter 에서 선택 가능하도록 등록
python -m ipykernel install --user --name psintel --display-name "Python (psintel)"
```

### 5. 환경 변수 설정 (선택)

`.env` 파일 생성:

```bash
# .env
DATA_PATH=./data/simulated/
MODEL_PATH=./models/
DEVICE=cuda
WINDOW_SIZE=100
DROPOUT_SAMPLES=100
```

---

## 빠른 시작

```bash
# 1. 저장소 클론
git clone https://github.com/postech-psi/psintelligence.git
cd psintelligence

# 2. 가상환경 활성화
micromamba activate psintel

# 3. 데이터 생성 (Step 1.5)
python src/data/generate_trajectories.py --num_flights 100

# 4. Step 1.5 부터 순차적 실행
# Step 1 은 pdf 파일로 대체
jupyter notebook
```

---

## 교육적 목표

본 프로젝트는 단순한 코드 구현을 넘어 다음과 같은 **엔지니어링 사고방식**을 함양합니다:

1. **점진적 복잡도**: 선형 (Step 1) → 비선형 (Step 2) → AI (Step 3-6) 로 단계적 학습
2. **물리 기반 AI**: 데이터만 믿지 않고 물리 법칙을 제약 조건으로 활용
3. **안전성 우선**: 정확도보다 불확실성 인식과 Fail-safe 로직을 우선시
4. **배포 고려**: 연구실 PC 가 아닌 온보드 컴퓨터 제약을 고려한 모델 설계
5. **논문 기반**: 각 Step 의 이론적 배경을 유명 논문에서 발췌하여 학술적 엄밀함 확보

---

## 참고 문헌

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

1. **Step 순서 준수**: 각 Step 은 이전 Step 의 산출물을 사용하므로 순차적 실행 필요
2. **데이터 생성**: Step 1.5 시뮬레이션 데이터 생성이 선행되어야 함
3. **GPU 권장**: Step 4-6 은 GPU 가 있으면 학습 속도가 5-10 배 빠름
4. **메모리 관리**: Step 6 ONNX Export 시 모델 크기가 급증할 수 있으므로 주의

---

## 문의

- **개발자**: 포항공과대학교 기계공학과 21학번 이승원
- **이메일**: dongdong0615@postech.ac.kr / sungwon.lee.2002@gmail.com

---

> **PSI 후배들이 AI를 접하고 공부하는데 도움이 되기를 바랍니다.**