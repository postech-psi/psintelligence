"""psintel_sim — RocketPy 없이 PSIntelligence 학습 데이터를 생성하는 시뮬레이터.

표준 라이브러리만 사용합니다 (numpy·scipy·RocketPy 불필요).

모델
----
1) 병진 (3-DOF 점질량, 수직면)
     m(t)·dv/dt = T(t)·û − ½ρ(z)·Cd·A·|v_rel|·v_rel + m·g
   · 레일 구속: 레일 방향 1-D. 추력이 중력 성분을 넘기 전에는 정지 유지(후방 이동 금지),
     레일길이(2.0 m) 이동 후 자유 비행.
   · 대기: ISA 1976 대류권.  · 적분: 고정 스텝 RK4.

2) 자세 (축소 모델, 6-DOF 아님)
   tilt  :  ψ̈ = −C·q·(ψ − ψ_fpa) − 2ζ√(C·q)·ψ̇            C=0.05, ζ=0.2
   roll  :  ṗ = a·cant·q − b·q·p                            a=1e-2, b=3e-4
   · ψ_fpa = −atan2(|v_h − w_along|, |v_z|)  [도]
   · gyro_pitch = dψ/dt (피치율),  gyro_yaw = 0 (2D 모델 한계)

3) 센서 노이즈 (실측 검증된 스펙)
   acceleration_noisy = round((a_z + 9.81 + b + e)/0.05)·0.05,  b~N(0,0.5), e~N(0,0.2)
   altitude_noisy     = round((altitude + e)/0.5)·0.5,          e~N(0,1.5)
   gyro_*_noisy       = round((w + b + e)/0.001)·0.001,         b~N(0,0.03), e~N(0,0.08)

검증 (정본 200편 대조)
---------------------
   아포지    |median| 0.33%, RMSE 0.91%
   tilt      비행별 평균오차 median 0.92도
   gyro_roll 비행별 상관 median +0.934
   노이즈    스펙 3자리 일치, 양자화 적용 확인

자세 모델의 잔여 한계와 기각된 가설은 `data/SCHEMA.md` §3.2 를 참조하세요.
"""
from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass, field

# ---- 물리 상수 (정본 데이터에서 역산·검증으로 확정) ----
G = 9.80665
R_AIR = 287.05287
L_LAPSE = 0.0065
T0, P0 = 288.15, 101325.0

DIAMETER = 0.15            # 기준 직경 [m] (정본: d=0.15 -> 잔차 0.33%, d=0.10 -> 6.6%)
REF_AREA = math.pi * DIAMETER ** 2 / 4
PROPELLANT_MASS = 0.478856  # 그레인 2 x 1815 kg/m3 x pi(0.0265^2-0.0075^2) x 0.065
RAIL_LENGTH = 2.0          # [m]
ELEVATION = 407.0          # 발사대 해발고도 [m]

# 자세 모델 상수 (정본 대조로 적합)
C_TILT, ZETA_TILT = 0.05, 0.2
A_ROLL, B_ROLL = 1e-2, 3e-4

# ---- 몬테카를로 샘플링 (Step 1.5 노트북의 실제 샘플링 코드와 동일) ----
#   launch_angle U(70,90) / wind_speed U(0,10) / wind_direction U(0,360)
#   mass_total U(3,5) / Cd U(0.5,1.5)
#   cg_offset_x,y N(0, 0.002) / cant_angle N(0, 0.5)
SAMPLING = {
    "mass_total": ("uniform", 3.0, 5.0),          # kg
    "launch_angle": ("uniform", 70.0, 90.0),      # deg (수평 기준)
    "wind_speed": ("uniform", 0.0, 10.0),         # m/s
    "wind_direction": ("uniform", 0.0, 360.0),    # deg
    "avg_drag_coeff": ("uniform", 0.5, 1.5),
    "cg_offset_x": ("normal", 0.0, 0.002),        # m
    "cg_offset_y": ("normal", 0.0, 0.002),        # m
    "cant_angle": ("normal", 0.0, 0.5),           # deg
}

NOISE = {
    "accel": {"std": 0.2, "bias_std": 0.5, "resolution": 0.05},
    "altitude": {"std": 1.5, "resolution": 0.5},
    "gyro": {"std": 0.08, "bias_std": 0.03, "resolution": 0.001},
}


# ----------------------------------------------------------------------
# 모터
# ----------------------------------------------------------------------
@dataclass
class Motor:
    """추력 곡선 (OpenRocket .eng 또는 (time, thrust) CSV)."""
    time: list
    thrust: list

    @classmethod
    def from_csv(cls, path: str) -> "Motor":
        t, f = [], []
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                k = list(row.keys())
                t.append(float(row[k[0]])); f.append(float(row[k[1]]))
        return cls(t, f)

    @classmethod
    def from_eng(cls, path: str) -> "Motor":
        t, f = [], []
        with open(path, encoding="utf-8") as fh:
            fh.readline()                       # RASP 헤더 1줄
            for line in fh:
                p = line.split()
                if len(p) >= 2:
                    try:
                        t.append(float(p[0])); f.append(float(p[1]))
                    except ValueError:
                        pass
        return cls(t, f)

    @property
    def t_end(self) -> float:
        return self.time[-1]

    @property
    def impulse(self) -> float:
        return sum((self.thrust[i] + self.thrust[i + 1]) / 2 * (self.time[i + 1] - self.time[i])
                   for i in range(len(self.time) - 1))

    def at(self, t: float) -> float:
        if t <= self.time[0] or t >= self.time[-1]:
            return 0.0
        lo, hi = 0, len(self.time) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.time[mid] <= t:
                lo = mid
            else:
                hi = mid
        w = (t - self.time[lo]) / (self.time[hi] - self.time[lo])
        return self.thrust[lo] + w * (self.thrust[hi] - self.thrust[lo])


# ----------------------------------------------------------------------
# 대기
# ----------------------------------------------------------------------
def isa_density(h_asl: float) -> float:
    """ISA 1976 대류권 밀도 [kg/m3]. h_asl = 해발고도 [m]."""
    h = min(max(h_asl, -611.0), 20000.0)
    T = T0 - L_LAPSE * h
    p = P0 * (T / T0) ** (G / (L_LAPSE * R_AIR))
    return p / (R_AIR * T)


# ----------------------------------------------------------------------
# 비행 1회
# ----------------------------------------------------------------------
@dataclass
class FlightSample:
    mass_total: float
    launch_angle: float
    wind_speed: float
    wind_direction: float
    avg_drag_coeff: float
    cg_offset_x: float
    cg_offset_y: float
    cant_angle: float

    @classmethod
    def draw(cls, rng: random.Random) -> "FlightSample":
        def draw_one(key):
            dist = SAMPLING[key]
            if dist[0] == "uniform":
                return rng.uniform(dist[1], dist[2])
            return rng.gauss(dist[1], dist[2])

        return cls(**{k: draw_one(k) for k in SAMPLING})


def simulate(motor: Motor, s: FlightSample, dt: float = 0.02) -> dict:
    """비행 1회 적분 -> 아포지까지의 시계열."""
    th = math.radians(s.launch_angle)
    ux, uz = math.cos(th), math.sin(th)
    # 수직면 근사: 풍속의 발사 방위 성분만 반영 (횡성분 무시 — SCHEMA.md 참조)
    w_along = s.wind_speed * math.cos(math.radians(s.wind_direction))

    m0 = s.mass_total
    m_dry = m0 - PROPELLANT_MASS
    mdot_scale = PROPELLANT_MASS / motor.impulse
    Cd, A = s.avg_drag_coeff, REF_AREA

    x = z = vx = vz = 0.0
    m = m0
    t = 0.0
    on_rail, s_rail, v_rail = True, 0.0, 0.0
    psi = -(90.0 - s.launch_angle)
    dpsi = 0.0
    p_roll = 0.0
    prev_psi = psi

    cols = {k: [] for k in ("time", "altitude", "velocity_z", "velocity_total", "acceleration_z",
                            "tilt_angle", "gyro_roll", "gyro_pitch", "gyro_yaw")}

    def rho_at(h_agl):
        return isa_density(ELEVATION + h_agl)

    while t < 120.0:
        T = motor.at(t)
        vt = math.hypot(vx, vz)
        q = 0.5 * rho_at(z) * vt * vt

        # ---- 자세 ----
        vh = math.sqrt(max(vt * vt - vz * vz, 0.0))
        if vt < 0.5 or on_rail:
            fpa = -(90.0 - s.launch_angle)
        else:
            fpa = -math.degrees(math.atan2(abs(vh - w_along), abs(vz)))
        if on_rail:
            psi, dpsi = -(90.0 - s.launch_angle), 0.0
        else:
            wn = math.sqrt(max(C_TILT * q, 0.0))
            dpsi += (-wn * wn * (psi - fpa) - 2 * ZETA_TILT * wn * dpsi) * dt
            psi += dpsi * dt
        # 롤 (cant 유도 + 감쇠)
        cant = math.radians(s.cant_angle)
        p_roll += (A_ROLL * cant * q - B_ROLL * q * p_roll) * dt

        cols["time"].append(t)
        cols["altitude"].append(ELEVATION + z)
        cols["velocity_z"].append(vz)
        cols["velocity_total"].append(vt)
        cols["acceleration_z"].append(0.0)      # 아래에서 채움
        cols["tilt_angle"].append(psi)
        cols["gyro_roll"].append(p_roll)
        cols["gyro_pitch"].append(math.radians((psi - prev_psi) / dt) if dt > 0 else 0.0)
        cols["gyro_yaw"].append(0.0)
        prev_psi = psi

        # ---- 병진 ----
        if on_rail:
            v_rel = v_rail - w_along
            D = 0.5 * rho_at(s_rail * uz) * Cd * A * abs(v_rel) * v_rel
            f = T - D - m * G * uz
            if v_rail <= 0.0 and f <= 0.0:
                a_rail, v_rail = 0.0, 0.0
            else:
                a_rail = f / m
            v_rail += a_rail * dt
            s_rail += v_rail * dt
            x, z, vx, vz = s_rail * ux, s_rail * uz, v_rail * ux, v_rail * uz
            az_kin = a_rail * uz
            if s_rail >= RAIL_LENGTH:
                on_rail = False
        else:
            def rhs(st):
                rr = math.hypot(st[2] - w_along, st[3])
                return [st[2], st[3],
                        (T * ux - 0.5 * rho_at(st[1]) * Cd * A * rr * (st[2] - w_along)) / m,
                        (T * uz - 0.5 * rho_at(st[1]) * Cd * A * rr * st[3]) / m - G]
            st = [x, z, vx, vz]
            k1 = rhs(st)
            k2 = rhs([st[i] + dt / 2 * k1[i] for i in range(4)])
            k3 = rhs([st[i] + dt / 2 * k2[i] for i in range(4)])
            k4 = rhs([st[i] + dt * k3[i] for i in range(4)])
            new = [st[i] + dt / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(4)]
            az_kin = (new[3] - vz) / dt if dt > 0 else 0.0
            x, z, vx, vz = new

        cols["acceleration_z"][-1] = az_kin

        if t < motor.t_end:
            m = max(m - mdot_scale * T * dt, m_dry)

        t += dt
        if t > motor.t_end + 0.3 and vz < 0:
            break

    # 아포지까지 자르기
    apogee = max(cols["altitude"])
    i_ap = cols["altitude"].index(apogee)
    for k in cols:
        cols[k] = cols[k][:i_ap + 1]

    out: dict = {k: v for k, v in cols.items()}
    out["apogee_altitude"] = apogee
    out["apogee_time"] = cols["time"][i_ap]
    out["max_speed"] = max(cols["velocity_total"])
    out["burnout_time"] = motor.t_end
    out["mass_dry"] = m_dry
    return out


# ----------------------------------------------------------------------
# 센서 노이즈
# ----------------------------------------------------------------------
def add_sensor_noise(flight: dict, rng: random.Random) -> dict:
    """검증된 노이즈 스펙 + 양자화를 적용해 *_noisy 컬럼을 만든다."""
    n = len(flight["time"])
    sp = NOISE

    bias = rng.gauss(0, sp["accel"]["bias_std"])
    acc = [flight["acceleration_z"][i] + 9.81 + bias + rng.gauss(0, sp["accel"]["std"])
           for i in range(n)]
    res = sp["accel"]["resolution"]
    flight["acceleration_noisy"] = [round(v / res) * res for v in acc]

    alt = [flight["altitude"][i] + rng.gauss(0, sp["altitude"]["std"]) for i in range(n)]
    res = sp["altitude"]["resolution"]
    flight["altitude_noisy"] = [round(v / res) * res for v in alt]

    for col in ("gyro_roll", "gyro_pitch", "gyro_yaw"):
        gb = rng.gauss(0, sp["gyro"]["bias_std"])
        g = [flight[col][i] + gb + rng.gauss(0, sp["gyro"]["std"]) for i in range(n)]
        res = sp["gyro"]["resolution"]
        flight[f"{col}_noisy"] = [round(v / res) * res for v in g]
    return flight


# ----------------------------------------------------------------------
# 파생 컬럼 (정본과 동일한 정의 — SCHEMA.md §3)
# ----------------------------------------------------------------------
def add_derived(flight: dict) -> dict:
    ap = flight["apogee_altitude"]
    t_ap = flight["apogee_time"]
    bt = flight["burnout_time"]
    h_theo, e_ratio, t2ap, phase = [], [], [], []
    for i, t in enumerate(flight["time"]):
        h = flight["altitude"][i]
        vz = flight["velocity_z"][i]
        ht = h + vz * vz / (2 * 9.81)
        h_theo.append(ht)
        e_ratio.append(ap / ht if ht > 0 else 0.0)
        t2ap.append(max(t_ap - t, 0.0))
        phase.append("powered_ascent" if t < bt else "coasting")
    flight["h_theoretical"] = h_theo
    flight["energy_ratio"] = e_ratio
    flight["time_to_apogee"] = t2ap
    flight["flight_phase"] = phase
    return flight
