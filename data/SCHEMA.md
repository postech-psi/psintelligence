# 데이터 스키마 — `data/`

PSIntelligence 파이프라인이 읽고 쓰는 데이터 파일의 컬럼 정의, 단위, 생성식입니다.
데이터를 다루기 전에 이 문서를 먼저 확인하세요.

---

## 1. 파일 세트 — 무엇이 정본인가

`data/simulated/` 에는 서로 다른 **두 세대**의 산출물이 함께 들어 있습니다. 이름이 비슷하지만 스키마가 다르므로 반드시 구분해야 합니다.

| 파일 | 행/편수 | 실제 형식 | 사용처 | 지위 |
|:--|:--|:--|:--|:--|
| `all_trajectories.csv.gz` | 67,424행 / 200편 | gzip CSV | Step 2, 3, 4, 5 | **정본 (현행)** |
| `flight_metadata.csv.gz` | 200행 / 200편 | gzip CSV | Step 2, 3, 4 | **정본 (현행)** |
| `all_trajectories.csv` | 17,294행 / 50편 | gzip CSV (확장자는 `.csv`) | 없음 | 레거시 |
| `flight_metadata.csv` | 50행 / 50편 | CSV | 없음 | 레거시 |
| `processed_trajectories.csv.gz` | — | gzip CSV | 없음 (Step 1.5가 쓰기만 함) | 레거시 |

> **주의:** 어떤 노트북도 `all_trajectories.csv` / `flight_metadata.csv` (확장자만 `.csv` 인 쪽)를 읽지 않습니다. 파이프라인이 읽는 것은 **`.csv.gz` 두 개**뿐입니다. 레거시 두 파일은 50편 규모의 이전 세대 산출물이며 스키마와 정의가 다릅니다.

학습된 모델·스케일러(Step 4~6에서 로드):

| 파일 | 설명 |
|:--|:--|
| `best_gru_model.pth` | Step 4에서 학습된 GRU (`PhysicsInformedWrapper` 로 저장, state_dict 키에 `gru.` 접두사) |
| `gru_int8.pth` | 위 모델의 INT8 양자화본 |
| `mamba_fp32.pth` | Step 6 배포 실습용 |
| `rocket_gru.onnx`, `rocket_gru.onnx.data` | ONNX 내보내기 결과 |
| `input_scaler.pkl` | 입력 정규화 스케일러 |
| `model_metadata.pkl` | 모델 메타데이터 |

입력 원자료(모터):

| 파일 | 설명 |
|:--|:--|
| `data/motor/thrust_curve_from_pressure.csv` | 연소실험 압력 → 추력 변환 결과. **현행 200편 데이터셋이 사용한 추력 소스** (연소 0~2.99 s, 총임펄스 286.8 N·s) |
| `data/motor/241108.xlsx`, `241114.csv` | 연소실험 원자료 |
| `../motor_data.eng` | OpenRocket RASP 형식 모터 파일 (연소 0~2.36 s, 총임펄스 467.9 N·s). 레거시 50편 데이터셋이 사용한 곡선으로 보임 |

---

## 2. `all_trajectories.csv.gz` — 정본 궤적 (200편)

아포지까지의 비행 1편을 시간순으로 기록. `dt = 0.02 s` 격자.

| 컬럼 | 단위 | 정의 |
|:--|:--|:--|
| `time` | s | 비행 시작(0)부터의 시간. 파일 내 최대 8.14 s |
| `altitude` | m (해발) | 고도. 발사대 고도 407 m 에서 시작 |
| `velocity_z` | m/s | 수직 속도 성분 |
| `velocity_total` | m/s | 속도 크기 |
| `acceleration_z` | m/s² | 수직 가속도 |
| `tilt_angle` | 도(추정) | 기체 기울기. 0 = 수직, 음수 = 진행 방향으로 기움 |
| `gyro_roll`, `gyro_pitch`, `gyro_yaw` | 도/s(추정) | 3축 각속도 |
| `acceleration_noisy` | m/s² | `acceleration_z` 에 센서 노이즈·해상도 적용 |
| `altitude_noisy` | m | `altitude` 에 노이즈 적용 (std 1.5 m, 해상도 0.5 m) |
| `gyro_*_noisy` | 도/s | 자이로 3축 노이즈 적용 |
| `h_theoretical` | m (해발) | **검증됨:** `altitude + velocity_z² / (2·9.81)` — 현재 상태의 탄도 아포지 추정 |
| `energy_ratio` | — | **검증됨:** `flight_metadata.apogee_altitude / h_theoretical`. 아포지에서 정확히 1 |
| `time_to_apogee` | s | 아포지까지 남은 시간 |
| `flight_phase` | 문자열 | `powered_ascent` \| `coasting` (강하는 기록되지 않음) |
| `flight_id` | — | 0~199 |

**검증:** `h_theoretical` 식은 200편 전체에서 오차 < 1e-3 m (저장 정밀도 한계) 로 일치. `energy_ratio` 도 오차 < 1e-4 로 일치.

## 3. `flight_metadata.csv.gz` — 정본 비행별 파라미터 (200편)

비행 1편당 1행. 몬테카를로 샘플링된 입력 파라미터와 결과 요약.

| 컬럼 | 단위 | 정의 |
|:--|:--|:--|
| `flight_id` | — | 0~199 |
| `apogee_altitude` | m (해발) | 최고 고도. 500.1~663.9 m |
| `apogee_time` | s | 아포지 도달 시각 |
| `burnout_time` | s | 연소 종료 시각. **전 200편 2.99 s 로 동일** — `thrust_curve_from_pressure.csv` 의 시간 범위와 일치 |
| `launch_angle` | 도 | 수평면 기준 발사각. 70.1~90.0 |
| `wind_speed` | m/s | 0.09~9.98 |
| `avg_drag_coeff` | — | 항력계수. 0.503~1.493 |
| `mass_total` | kg | **이륙 총질량** (추진제 포함). 3.010~4.997 |
| `max_speed` | m/s | 최대 속도 |
| `cg_offset_x`, `cg_offset_y` | m(추정) | 무게중심 편심 |
| `cant_angle` | 도 | 핀 캔트각 |

> **추진제 질량은 이 파일에 없습니다.** Step 1.5 노트북이 `mass_dry = mass_total − motor.propellant_initial_mass` 로 계산합니다. 즉 모터가 고정이므로 **추진제 질량은 전 비행 상수**입니다.

### 3.1 생성에 쓰인 물리 파라미터 (코드 + 실측 역산으로 확정)

| 항목 | 값 | 근거 |
|:--|:--|:--|
| 추력 곡선 | `data/motor/thrust_curve_from_pressure.csv` | 메타 `burnout_time = 2.99 s` 가 곡선의 시간 범위와 일치 |
| 총임펄스 | 286.8 N·s | 곡선 적분 (실연소 구간 0.79~2.57 s, 이후 0) |
| 추진제 질량 | **0.478856 kg** | 그레인 2개 × 1815 kg/m³ × π(0.0265² − 0.0075²) × 0.065 |
| 기준 직경 | **0.15 m** (반경 0.075) | 재현 검증: d=0.15 → 잔차 0.33%, d=0.10 → 6.6% |
| 발사대 고도 | 407 m (ASL) | 궤적의 t=0 고도 |
| 레일 | 2.0 m, 이탈 전 후방 이동 구속 | 데이터는 t ≈ 1.05 s 까지 정지 상태 유지 |
| 항력 모델 | 상수 `avg_drag_coeff` | 노트북이 `power_off_drag = power_on_drag = float(Cd)` 로 전달 |

> 위 파라미터로 3-DOF 점질량 시뮬레이터를 돌리면 **200편 아포지가 \|median\| 0.33% 로 재현**됩니다.
> 재생성 후 이 값이 크게 벌어지면 물리 파라미터가 바뀐 것이므로 `verify_dataset.py` 와 함께 확인하세요.

### 3.2 자세 컬럼 — 정의 확정 + 재현 모델

정의는 RocketPy 공식 문서로 확정했습니다 ([Flight Class](https://docs.rocketpy.org/en/latest/reference/classes/Flight.html)).
본체 좌표계에서 **z축이 로켓 대칭축**입니다.

| 컬럼 | RocketPy 속성 | 공식 정의 | 재현 모델 / 결과 |
|:--|:--|:--|:--|
| `tilt_angle` | `flight.theta(t)` | 3-2-3 너테이션 각 [도] — 수직 기준 기울기 | t=0 에서 정확히 `−(90 − launch_angle)` (corr −1.0000). 동압 의존 2차 모델로 추종, **비행별 평균오차 median 0.92도** (119/200편 1도 이내, 188/200편 2도 이내) |
| `gyro_pitch` | `flight.w1(t)` | 본체 x축 각속도 = **피치율 q** | dθ/dt 와 상관 +0.86, 범위 ±2.30 = max\|dθ/dt\| 2.302 |
| `gyro_yaw` | `flight.w2(t)` | 본체 y축 각속도 = **요율 r** | dθ/dt 와 상관 −0.75 (오일러 결합) |
| `gyro_roll` | `flight.w3(t)` | 본체 z축(대칭축) 각속도 = **롤율 p** | 롤 감쇠 ODE 로 **비행별 상관 median +0.934** (111/200편이 개별 0.9 이상) |

재현 모델 (3-DOF 점질량 상태 + 아래 두 식이면 자세 채널을 생성할 수 있습니다):

```
tilt :  ψ̈ = −C·q·(ψ − ψ_fpa) − 2ζ√(C·q)·ψ̇        C = 0.05, ζ = 0.2
        ψ_fpa = −atan2(|v_h − w|, |v_z|)  [도],  레일 구간은 ψ = −(90 − launch_angle) 고정
        (정적 안정 로켓의 복원 모멘트 ∝ 동압 → q → 0 이면 자세가 얼어붙음)
        → 행 단위 median 0.35도 / 비행별 평균 median 0.92도

roll :  ṗ = a·cant·q − b·q·p                     a = 1e-2, b = 3e-4
        (정상상태 p_ss = (a/b)·cant ∝ cant,  시상수 ∝ 1/q)
```

> 잔여 한계 — **축소 모델은 바닥에 도달했습니다.** 오차는 두 구간에 몰려 있습니다:
> 레일 이탈 직후(t/t_ap 0.2~0.3)와 근-아포지(0.9~1.0). 중간 구간은 median 0.27도입니다.
> 다음 가설들은 실측으로 **기각**되어, 남은 오차는 축소 모델로 흡수할 수 없는 6-DOF 효과로 판단됩니다.
>
> | 가설 | 검정 결과 |
> |:--|:--|
> | 강성이 관성에 반비례 (`C ∝ 1/I_yy`) | 기각 — 0.349도 vs 전역 0.345도 |
> | 미기록 풍향 (`wind_direction`) | 기각 — 무제한 피팅에서 개선 0.002도 |
> | 레일 이탈 초기 피치율 (ψ̇₀ ≈ −5 deg/s) | 기각 — 지표가 ψ̇₀ 에 평평 (0.351 vs 0.352) |
> | 레일 이탈 시각 추정 (2 m 기준 vs 궤적 역산) | 기각 — 0.722도 vs 0.733도 |
>
> 또한 최적 강성 C=0.05 는 물리 추정값(A·C_Nα·Δx/I ≈ 0.02)의 약 2.5배로, 축소 모델 형태 자체가
> 근사임을 시사합니다. 더 낮추려면 강체 6-DOF(유한 레일 천이, cg 편심·캔트 결합)가 필요합니다.
> 참고로 롤이 전 비행에서 1회전 미만(최대 6 rad)이라 코닝 위상 검정은 불가능했습니다.

### 3.3 센서 노이즈 모델 (생성식 + 실측 검증)

| 컬럼 | 생성식 | 실측 |
|:--|:--|:--|
| `acceleration_noisy` | `round((a_z + 9.81 + b + e)/0.05)×0.05`, b~N(0, 0.5), e~N(0, 0.2) | bias σ **0.5015**, 노이즈 σ **0.1997**, 격자 이탈 0/67424 |
| `altitude_noisy` | `round((altitude + e)/0.5)×0.5`, e~N(0, 1.5) | σ **1.5107**, 비행별 bias 없음(0.081), 격자 이탈 0/67424 |
| `gyro_*_noisy` | `round((w + b + e)/0.001)×0.001`, b~N(0, 0.03), e~N(0, 0.08) | σ **0.0853** ≈ √(0.08²+0.03²), bias σ **0.0312**, 격자 이탈 0/67424 |

> 노이즈 컬럼은 **양자화가 실제로 적용**되어 있습니다(clean 컬럼은 격자 위에 있지 않음).
> `acceleration_noisy` 는 중력이 포함된 specific force 입니다 (Step 2 의 IMU 입력).

---

## 4. 레거시 50편 세트 — 정본과 정의가 다름

`all_trajectories.csv`(16컬럼) / `flight_metadata.csv`(15컬럼) 는 200편 세트와 **컬럼 이름·개수·파생식이 모두 다릅니다.**

| 항목 | 정본 (200편) | 레거시 (50편) |
|:--|:--|:--|
| 질량 규모 | `mass_total` 3.0~5.0 kg | 6.0~8.0 kg |
| 추력 소스 | `thrust_curve_from_pressure.csv` (0~2.99 s) | `motor_data.eng` (0~2.36 s) |
| `burnout_time` | 2.99 s (전편 동일) | 2.36 s (전편 동일) |
| `h_theoretical` | 매 시점 `altitude + velocity_z²/2g` (행마다 변함) | **비행당 상수** = `h_burnout + v_burnout²/2g` |
| `energy_ratio` | `apogee / h_theoretical(t)` | **비행당 상수** = `apogee / h_theoretical` |
| 속도 컬럼 | `velocity_z`, `velocity_total` | `velocity` |
| 고유 컬럼 | `tilt_angle`, `flight_phase`, `time_to_apogee` | `mass`, `h_burnout`, `v_burnout` |

**두 세트를 합쳐서 쓰지 마세요.** `energy_ratio` 와 `h_theoretical` 의 정의가 달라 그대로 concat 하면 라벨이 섞입니다.

레거시 세트의 `mass` 컬럼은 비행당 상수이며 **연소 종료 질량**입니다. 정본 메타데이터에는 대응 컬럼이 없습니다.

---

## 5. 알려진 결함

| # | 대상 | 내용 | 상태 |
|:--|:--|:--|:--|
| D1 | `all_trajectories.csv`, `flight_metadata.csv` | 확장자가 `.csv` 인데 내용이 gzip (정본은 `.csv.gz`) | `flight_metadata.csv` 는 일반 CSV 로 정정. `all_trajectories.csv` 는 레거시 잔존 |
| D2 | `flight_metadata.csv` (레거시) | `mass_dry` 가 17/50편에서 `mass_total` 이상 → 추진제 질량 0 또는 음수 (물리 불가능) | **정정 완료** — `mass_dry = mass_total − 1.18236` |
| D3 | `flight_metadata.csv` (레거시) | `mass_total` 고유값이 33개뿐 (50행) → 파라미터 조합 중복 | 미해결 (레거시) |
| D4 | 정본 세트 | `mass_dry` 컬럼 자체가 없어 추진제 질량을 재계산해야 함 | 〃 (문서화로 대응) |
| D5 | 저장소 전체 | 데이터 스키마 문서 부재 | **본 문서로 해소** |
| D6 | `step1.5` 노트북 | `diameter=0.15` 를 전달하지만 `create_rocket` 기본값은 0.10 | **해소** — 재현 검증 결과 기준 직경 = **0.15 m** (d=0.15 → 잔차 0.33%, d=0.10 → 6.6%) |
| D7 | `step1.5` 노트북 | 샘플링 분포가 MD 표 / `EnvironmentSampler.params` / 실제 `np.random.uniform` 호출 3곳에서 서로 다름 | 미해결 |
| D8 | `step1.5` → Step 4/5 | `step1.5` 는 스케일러를 `./models/input_scaler.pkl` 에 쓰지만 Step 4/5 는 `./data/simulated/input_scaler.pkl` 를 읽음 (Step 5 는 두 경로를 혼용) | 미해결 |
| D9 | `step1.5` 노트북 | 재실행 시 정본 파일을 덮어씀 (`_regen` 접미사 없음) | 문서화로 대응 |
| D10 | 정본 메타데이터 | `wind_direction` 이 샘플링되어 시뮬레이션에 쓰였으나 컬럼으로 기록되지 않음 | **부분 해결** — 기존 200편은 궤적에서 복원 불가(식별 시도 실패). Step 1.5 에 기록 1줄 추가(신규 생성분부터 기록됨) |
| D11 | `step1.5` 노트북 | 노이즈 σ 가 본문 표(0.5/0.1/0.01)와 코드(1.5/0.2/0.08)에서 불일치 — **데이터는 코드 기준** | 문서화로 대응 (§3.3) |
| D12 | `step1.5` 노트북 | 천음속 이상현상 분기(`0.85 < Mach < 1.15`)가 도달 불가 — 데이터의 최대 Mach 는 0.228 | 문서화로 대응 |

---

## 6. 무결성 검증

데이터를 수정하거나 새로 생성한 뒤에는 반드시 실행하세요.

```bash
python scripts/verify_dataset.py
```

검사 항목:

1. 정본 파일 존재·형식 (`.csv.gz` 실제 gzip 여부)
2. 200편 세트의 `flight_id` 연속성, 궤적↔메타데이터 커버리지 일치
3. `h_theoretical == altitude + velocity_z²/2g` (전 행)
4. `energy_ratio == apogee_altitude / h_theoretical`
5. `burnout_time` 전편 동일 여부 (모터 고정 가정)
6. 물리 정합성: 질량 > 0, 아포지 ≥ 발사대 고도, 속도·가속도 유한
7. 레거시 파일이 정본과 혼입되지 않았는지 (컬럼 집합 대조)

---

## 7. 데이터 재생성 — 두 경로

### (A) 자체 시뮬레이터 — **RocketPy 불필요** (권장)

```bash
python scripts/generate_dataset.py                       # 200편 -> data/generated/
python scripts/generate_dataset.py --n 50 --seed 7       # 소규모/재현
python scripts/verify_dataset.py --data-dir data/generated
```

- 표준 라이브러리만 사용. 200편을 **약 2.3초**에 생성합니다.
- 정본과 동일한 스키마(`all_trajectories.csv.gz` + `flight_metadata.csv.gz`)로 씁니다.
- 기본 출력이 `data/generated/` 이므로 **정본을 덮어쓰지 않습니다.**
- 정본 대비 실측: 아포지 \|median\| 0.33%, tilt 비행별 평균오차 median 0.92도,
  gyro_roll 비행별 상관 median 0.934, 노이즈 스펙 3자리 일치.
- **알려진 차이** (SCHEMA.md §3.2 의 축소 모델 한계):
  · `gyro_yaw` 는 2D 모델이라 **0** 으로 채워집니다 (파이프라인 미사용 컬럼)
  · `tilt_angle` 이 드물게 +2도까지 나옵니다 (정본은 전부 음수) — 감쇠비 0.2 의 과도 응답
  · `gyro_pitch`·`gyro_roll` 의 범위가 정본보다 좁습니다 (정본 ±2.30 / ±0.85 → 생성 ±1.3 / ±0.64)
- `wind_direction` 컬럼이 메타데이터에 포함됩니다 (정본의 결함 D10 대응).

### (B) 원본 경로 — RocketPy

```bash
pip install -r requirements-full.txt   # rocketpy, seaborn 포함
jupyter notebook "step1.5-data generation.ipynb"
```

- Step 1.5 는 **선택 단계**입니다. 동봉 데이터로 Step 2~6 을 바로 실행할 수 있습니다.
- ⚠️ **재생성은 기존 정본을 덮어씁니다.** `all_trajectories.csv.gz`, `flight_metadata.csv.gz`,
  `data/motor/thrust_curve_from_pressure.csv` 를 같은 경로에 다시 씁니다. 먼저 백업하세요.
- 재생성 후 반드시 `scripts/verify_dataset.py` 를 돌리고, 실패하면 백업으로 되돌리세요.
