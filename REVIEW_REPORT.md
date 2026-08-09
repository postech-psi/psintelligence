# PSIntelligence 101 — 코드 리뷰 보고서

**레포지토리:** postech-psi/psintelligence
**검토 일자:** 2026-05-15
**검토 대상:** 모든 `.ipynb` 파일, `main.py`, `requirements.txt`, `config.json`, `.eng` 파일

---

## 1. 프로젝트 개요

POSTECH 로켓 동아리 PSI의 에비오닉스 팀이 제작한 **로켓 비행 중 실시간 아포지(최고 고도) 예측 AI 파이프라인** 교육 과정이다. 총 6개 Step의 Jupyter 노트북으로 구성되어 있으며, 기초 확률론(Kalman Filter)부터 Mamba/ONNX 배포까지 로켓 비행 데이터 기반 종단간(end-to-end) AI 파이프라인을 학습한다.

### Step 구성

| Step | 노트북 파일 | 핵심 주제 | 데이터 |
|------|------------|-----------|--------|
| 1.5 | step1.5-data generation.ipynb | RocketPy 시뮬레이션, 몬테카를로 샘플링, 추력 곡선 변환, 피처 엔지니어링 | motor/ (실험 데이터) + simulated/ (출력) |
| 2 | step2-EKF&UKF.ipynb | EKF/UKF 직접 구현, Unscented Transform, 필터 성능 평가 | simulated/ |
| 3 | step3-ANN&PINN.ipynb | Baseline ANN, Physics-Informed Neural Network, 물리 손실 함수, 자이로 해석 | simulated/ |
| 4 | step4-LSTM&GRU.ipynb | Sliding Window, GRU 시계열 모델, Physics-Informed Wrapper, Early Stopping | simulated/ |
| 5 | step5-UQ.ipynb | MC Dropout 기반 불확실성 정량화, OOD 탐지, Safe-Logic | simulated/ |
| 6 | step6-Mamba&Deploy.ipynb | Mamba 이론, Dynamic Quantization (FP32→INT8), ONNX Export | simulated/ (기학습 모델) |

### 핵심 Python 파일

| 파일 | 역할 | 상태 |
|------|------|------|
| main.py | 엔트리 포인트 | **사실상 빈 파일** (print("Hello from psintel!")만 있음) |
| requirements.txt | 의존성 목록 (154개 패키지) | pip freeze 결과의 전체 dump |
| motor_data.eng | OpenRocket 모터 파일 (PSI 50mm, 135N) | 1143 라인 추력 데이터 |

---

## 2. 주요 지표

| 항목 | 값 |
|------|-----|
| Python/노트북 파일 (비.git) | 8개 (노트북 6 + main.py + requirements.txt) |
| 총 노트북 코드 라인 수 | 약 1,900 라인 (코드 셀 기준) |
| 노트북 마크다운 라인 수 | 약 2,000+ 라인 (이론 설명 풍부) |
| 데이터 파일 | simulated/ 30개, motor/ 10개 |
| 시뮬레이션 비행 수 | 200 flights (config.json) |
| 학습된 모델 파일 | best_gru_model.pth, gru_int8.pth, mamba_fp32.pth, rocket_gru.onnx, input_scaler.pkl, model_metadata.pkl |
| 라이선스 | Apache 2.0 |
| PDF 참고 자료 | 101-1 기초확률론 & 선형칼만필터.pdf (1개) |
| git history | shallow clone (단일 커밋 추정) |

---

## 3. 강점 / 잘한 점

### 3.1 교육적 설계가 탁월함
- 각 Step이 이전 Step의 이론(분산, 공분산, 상태방정식)을 명시적으로 참조하며 점진적으로 난이도가 상승한다.
- Step 6의 Mamba 섹션은 칼만 필터의 상태방정식과의 연결성을 설명하여 제어공학→딥러닝으로의 개념적 확장을 자연스럽게 유도한다.
- 로켓 동아리라는 실질적 사용 사례(낙하산 전개 시점 결정)에 맞춰 모든 예제가 설계되었다.

### 3.2 Physics-Informed 설계 철학
- PINN에서 물리 손실 함수(단조 감소 제약 dr/dt ≤ 0, 범위 제약 r ∈ [0.8, 1.0])를 직접 구현하여 데이터가 부족한 구간에서도 물리적 타당성을 확보한다.
- GRU 모델 역시 `PhysicsInformedWrapper`를 통해 `h_apogee = h_curr + (v_z² / 2g) * r_pred` 하드 제약을 두어 출력을 물리적으로 해석 가능하게 만든다.
- "자이로의 비명 해석하기" 미션(Step 3 Cell 24)은 도메인 지식과 데이터 분석을 연결한 창의적인 교육 설계다.

### 3.3 실전 배포를 고려한 엔드-to-엔드 구성
- FP32 → INT8 동적 양자화 (3.8배 압축, 73.3% 메모리 절약)
- ONNX 변환 및 검증 (opset 18)
- MC Dropout 기반 불확실성 정량화 → OOD 탐지 → Safe-Logic 결정 파이프라인
- 실제 비행 데이터(112.txt)로 각 Step의 모델을 검증하는 일관된 평가 체계

### 3.4 코드 구성의 일관성
- 모든 노트북이 동일한 데이터 경로(`./data/simulated/`)를 참조한다.
- 피처 정의(`altitude, velocity_z, acceleration_z, tilt_angle, gyro_roll, dynamic_pressure`)가 Step 3~6 전반에 걸쳐 일관되게 유지된다.
- 시각화에 한글 폰트를 설정한 점이 한국 사용자에게 친화적이다.

### 3.5 참고 문헌 인용
- Gal & Ghahramani (2016), Gu & Dao (2023, Mamba), Zhu et al. (2024, Vision Mamba) 등 주요 논문을 마크다운 셀에서 인용하며 이론적 배경을 제공한다.

---

## 4. 발견된 이슈

### 🔴 Critical (심각)

#### C-1: requirements.txt가 pip freeze 전체 덤프
- **파일:** `/home/aero_groot/repos/psintelligence/requirements.txt`
- **문제:** `pip freeze` 결과를 그대로 저장하여 `pywinpty`, `pyreadline3` 등 Linux 환경과 무관한 패키지와 `file:///C:/miniconda3/...`, `file:///home/task_.../croot/pip/...` 같은 로컬 경로 pin이 포함되어 있다.
- **영향:** `pip install -r requirements.txt` 시 path 기반 패키지는 설치 실패하며, 불필요한 패키지 100+개로 가상환경이 비대해진다.
- **권장 조치:** 실제로 필요한 패키지만 명시 (`numpy, pandas, torch, rocketpy, matplotlib, scipy, scikit-learn, onnx, onnxruntime, joblib, filterpy` 등)

#### C-2: main.py가 사실상 비어 있음
- **파일:** `/home/aero_groot/repos/psintelligence/main.py`
- **문제:** `def main(): print("Hello from psintel!")` 외에 아무 로직도 없다. 저장된 모델 로드, 추론, 시각화 등 실행 가능한 데모가 전혀 없다.
- **영향:** 사용자가 Step 1.5~6을 순차 실행하지 않고는 이 프로젝트의 기능을 활용할 수 없다. CLI/API로의 진입점이 전혀 없다.
- **권장 조치:** 최소한 학습된 GRU 모델을 로드하여 입력 CSV에 대해 아포지 예측을 수행하는 스크립트를 추가하거나, 전체 파이프라인을 한 번에 실행하는 CLI 인터페이스를 구축할 것.

#### C-3: Mamba는 이론 설명만 있고 실제 구현/학습이 없음
- **파일:** `step6-Mamba&Deploy.ipynb`
- **문제:** Step 6 제목은 "Mamba & Edge Deployment"이나, 실제 코드 셀에서는 GRU 모델을 그대로 가져와 양자화/ONNX 변환만 수행한다. 마크다운 셀에서 Mamba 이론을 상세히 설명했지만, Mamba 모델의 정의, 학습, GRU와의 성능 비교 코드가 전혀 없다.
- **영향:** "Step 6 = Mamba"라는 기대와 실제 구현 간에 큰 괴리가 있다. 노트북 마지막에 "학습 난이도와 시간상의 이유로 GRU를 바탕으로 양자화, 배포 할 것임"이라는 안내가 있으나, Step 제목이 오해를 준다.
- **권장 조치:** (a) Step 제목을 "Model Quantization & Edge Deployment"로 변경하거나, (b) 최소한 `mamba-ssm` 패키지를 사용한 간단한 Mamba 모델 정의 및 추론 예제를 추가할 것.

---

### 🟡 Warning (주의)

#### W-1: Step 2 EKF/UKF에 실제 센서 바이어스/노이즈 모델이 단순화됨
- **파일:** `step2-EKF&UKF.ipynb`
- **문제:** EKF의 `predict()`에서 가속도 센서 값을 상태 전이에 직접 사용(`u`)한다. 이는 IMU의 가속도가 "측정값 = 실제 가속도 + 바이어스 + 노이즈"임을 고려하지 않은 설계다. 실제로는 가속도 바이어스를 상태 변수에 포함시켜 추정해야 한다.
- **코드:**
  ```python
  def _dynamics(self, x, u, dt):
      z, v = x[0], x[1]
      new_z = z + v * dt + 0.5 * u * dt**2  
      new_v = v + u * dt
      return np.array([new_z, new_v])
  ```
- **권장 조치:** 상태 벡터에 가속도 바이어스(b_a)를 추가하고, EKF/UKF가 이를 함께 추정하도록 확장할 것. 교육 자료이므로 "바이어스 추정을 포함한 확장"을 optional exercise로 제시해도 좋다.

#### W-2: Step 3 PINN의 시간 입력 처리 방식이 깨지기 쉬움
- **파일:** `step3-ANN&PINN.ipynb`, Cell 18 (compute_physics_loss) & Cell 24 (train_pinn_v2)
- **문제:** 모델 입력의 첫 번째 채널을 "시간"으로 가정하고 `input_features[:, 0] = time_norm`으로 교체한다. 그러나 `final_learning_cols` 정의에 `'time'`이 포함되기도 하고 안 되기도 한다. 피처 순서 의존성은 유지보수에 취약하다.
- **권장 조치:** 시간을 별도 입력으로 받거나, named feature dict를 사용하여 피처 순서에 의존하지 않도록 리팩토링할 것.

#### W-3: Step 4 GRU 모델과 Step 5/6 GRU 모델 간 불일치 가능성
- **파일:** `step4-LSTM&GRU.ipynb`, `step5-UQ.ipynb`, `step6-Mamba&Deploy.ipynb`
- **문제:** GRUModel 클래스가 Step 4~6에 걸쳐 중복 정의되어 있다. Step 6에서는 `self.fc`의 시퀀셜 인덱스가 0,1,2,3,4인데, Step 5에서는 동일 구조이나 완전 별도 정의다. 한 곳에서 수정하면 다른 곳과 불일치가 발생한다.
- **권장 조치:** 공통 모델 정의를 `model.py` 또는 `psintel/models/gru.py`로 분리하고, 모든 노트북이 import하여 사용하게 할 것.

#### W-4: Step 4/5/6에서 best_gru_model.pth의 키 매핑 문제
- **파일:** `step6-Mamba&Deploy.ipynb` Cell 3
- **문제:** 저장된 state_dict에 `'gru.'` 접두사가 있음(`step4` PhysicsInformedWrapper로 저장). Step 6에서는 이 접두사를 수동으로 제거하는 코드(`k[4:] if k.startswith('gru.') else k`)를 추가했다. 그러나 Step 5의 `load_state_dict` 코드는 이 매핑 없이 직접 로드하여 오류가 발생할 가능성이 있다.
- **권장 조치:** state_dict 저장 시 접두사를 통일하거나, 로드 시 자동으로 키 매핑을 시도하는 유틸리티 함수를 만들 것.

#### W-5: 시드(Seed) 고정이 Step 1.5에만 있고 나머지 Step에는 없음
- **파일:** `step1.5-data generation.ipynb`에는 `np.random.seed(42)`가 있으나, Step 2~6에는 시드 고정 코드가 없다.
- **영향:** GRU 학습 재현성이 보장되지 않는다. 같은 코드를 실행해도 매번 다른 결과가 나올 수 있다.
- **권장 조치:** PyTorch seed (`torch.manual_seed`, `torch.cuda.manual_seed_all`)를 각 Step의 첫 코드 셀에 추가할 것.

#### W-6: RockePy 경고 필터 무시
- **파일:** `step1.5-data generation.ipynb` 등
- **문제:** `warnings.filterwarnings('ignore')`를 전역으로 적용하여 모든 경고를 무시한다. 데이터 무결성 문제나 API 변경 경고가 사용자에게 전달되지 않는다.
- **권장 조치:** 특정 경고 카테고리만 필터링하거나, `warnings.filterwarnings('once')`를 사용할 것.

---

### 🟢 Suggestion (제안)

#### S-1: config.json에 시뮬레이션 파라미터를 두었지만 코드에 하드코딩된 값이 많음
- **파일:** `data/simulated/config.json` vs `step1.5-data generation.ipynb`
- **문제:** config.json에 `n_flights: 200`, `dt: 0.02`, 노이즈 스펙 등이 정의되어 있으나, 실제 시뮬레이션 코드에는 `burn_time = 2.99`, `m_propellant = 0.479` 등 하드코딩된 값이 있다.
- **권장 조치:** config.json을 모든 하이퍼파라미터의 단일 진실 공급원(Single Source of Truth)으로 사용하고, 노트북에서 파일을 로드하게 할 것.

#### S-2: 데이터 CSV에 대한 스키마 문서 부재
- `all_trajectories.csv.gz`의 40+ 컬럼 각각에 대한 설명이 없다. Step 3 Cell 12에서 `available_cols`를 출력하지만, 각 컬럼의 의미는 마크다운으로 문서화되어 있지 않다.
- **권장 조치:** `data/SCHEMA.md`를 추가하여 모든 컬럼의 이름, 단위, 의미, 생성 방식을 문서화할 것.

#### S-3: 노트북 내 코드 중복이 많음
- import 구문, 폰트 설정, 데이터 로드 코드가 모든 노트북에 중복되어 있다. 6개 노트북 각각에 30~50라인의 중복 셋업 코드가 있다.
- **권장 조치:** 공통 설정을 `utils.py`로 분리: `setup_korean_font()`, `load_simulation_data()`, `setup_seed()` 등.

#### S-4: Step 2 UKF의 수치 안정성
- `UKF.generate_sigma_points()`에서 `np.linalg.eigh`로 시그마 포인트를 생성하나, 수치적 불안정이 발생할 수 있다 (음수 고유값 → `np.maximum(vals, 1e-10)`으로 처리). Cholesky 분해가 더 안정적이다.
- **권장 조치:** `scipy.linalg.cholesky` 또는 `np.linalg.cholesky`로 변경하고, 실패 시 `np.linalg.eigh`로 fallback할 것.

#### S-5: .gitignore 파일 부재
- `data/simulated/`의 `.pth`, `.onnx`, `.pkl` 등 대용량 바이너리 파일이 git에 포함되어 있다. 이들은 git LFS 없이 일반 git으로 관리하면 저장소가 비대해진다.
- **권장 조치:** `.gitignore`를 추가하고, 바이너리 모델 파일은 Releases 또는 Hugging Face Hub로 이관할 것.

#### S-6: step3의 energy_ratio 계산 방식 검증 필요
- Step 3 Cell 12에서 `energy_ratio`가 없다면 재계산하나, 그 공식이 어떻게 계산되는지 명확하지 않다. `energy_ratio`가 Step 1.5에서 생성되었는지 Step 3에서 생성되었는지 혼란스럽다.
- **권장 조치:** `energy_ratio`의 생성 주체와 계산 공식을 명확히 문서화하거나, 전처리 코드를 Step 1.5로 일원화할 것.

---

### ⬜ Note (참고사항)

- **교육용 PDF:** `101-1 기초확률론 & 선형칼만필터.pdf`는 Step 1.5와 Step 2의 이론적 배경을 설명하는 교육 자료다. Step 1.5의 마크다운 표가 이 PDF의 챕터를 참조하고 있어 교육 과정 설계가 체계적이다.
- **motor_data.eng:** OpenRocket 형식의 모터 데이터로, PSI 동아리의 실제 50mm/135N 모터 스펙이 반영되어 있다. RocketPy 시뮬레이션에 사용된다.
- **학습된 모델들:** `best_gru_model.pth` (159,617 파라미터, 0.61MB), `gru_int8.pth` (0.16MB, 3.8x 압축)이 포함되어 있어 사용자가 별도 학습 없이 Step 5/6을 실행할 수 있다.
- **ONNX 모델:** `rocket_gru.onnx`와 메타데이터 파일(`rocket_gru.onnx.data`)이 포함되어 ONNX Runtime 기반 추론이 가능하다.
- **MC Dropout 샘플 수:** Step 5에서 `n_samples=100` (Gal & Ghahramani, 2016의 권장 범위: 50~100). 적절한 설정이다.

---

## 5. 종합 평가

### Total Score: ⭐⭐⭐⭐☆ (4.0 / 5.0)

| 평가 항목 | 점수 | 코멘트 |
|----------|------|--------|
| 교육적 완성도 | ★★★★★ | 단계별 구성, 이론-실습 연결, 도메인 특화가 탁월 |
| 코드 품질 | ★★★☆☆ | 중복 많음, 재현성 부족, requirements.txt 오염 |
| 확장성/유지보수성 | ★★☆☆☆ | 모듈 분리 부재, 하드코딩 다수, .gitignore 없음 |
| 문서화 | ★★★★☆ | 마크다운 설명 풍부하나 데이터 스키마 문서 부재 |
| 실전 배포 준비 | ★★★★☆ | ONNX/양자화/UQ/실제 데이터 검증 포함 |
| 재현성 | ★★★☆☆ | 시드 고정 불완전, 환경 의존성 관리 미흡 |

### 요약

PSIntelligence 101은 **교육 목적으로는 매우 훌륭한 프로젝트**다. 로켓 아포지 예측이라는 명확한 목표 아래 칼만 필터 → EKF/UKF → PINN → GRU → 불확실성 정량화 → 배포(양자화/ONNX)로 이어지는 AI 파이프라인의 전 과정을 실제 데이터와 함께 학습할 수 있다. 물리 법칙을 신경망에 통합하는 Physics-Informed 접근법과, MC Dropout으로 예측 불확실성을 정량화하고 OOD를 탐지하는 Safe-Logic 개념은 실제 항공우주 시스템에 적용 가능한 수준이다.

**핵심 개선이 필요한 부분:**
1. **requirements.txt 정비** (Critical) — path 기반 패키지 제거
2. **main.py 기능 추가** (Critical) — 최소한의 추론 데모 제공
3. **Step 6 제목/내용 불일치 해소** (Critical) — Mamba 실제 코드 또는 제목 변경
4. **공통 코드 모듈화** (Warning) — 모델 정의, 설정, 유틸리티를 별도 파일로 분리
5. **재현성 체계 구축** (Warning) — 모든 Step에 시드 고정, config.json 일원화

이 프로젝트는 POSTECH PSI 동아리의 실제 로켓 개발 경험과 최신 AI 기술을 결합한 독특한 교육 자료로, 위 이슈들만 해결된다면 오픈소스 교육 프로젝트로서 큰 가치를 지닐 것이다.
