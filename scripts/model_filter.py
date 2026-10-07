#!/usr/bin/env python3
"""명목 물리모델 기반 잔차 필터 — Step 5.5 의 L1/L2 공통 입력을 만든다.

왜 필요한가
-----------
Step 2 의 EKF/UKF 는 **측정된 가속도를 입력으로 쓰는 기구학 추적기**입니다.
따라서 어떤 궤적이든 "그게 정답"으로 따라가고, 혁신에는 센서 잡음만 남습니다.
실측: 파라미터 이탈(OOD) 비행과 정상 비행의 혁신 크기 비 = **1.00x** (탐지 불가).

이 모듈은 대신 **명목(nominal) 파라미터 물리 모델**로 상태를 예측합니다.
비행체가 명목에서 벗어나면 모델 예측이 실제와 갈라져 잔차가 커집니다.
실측: 같은 OOD 세트에서 비 = **2.05x**, TPR 52% @ FPR 5%.

    파라미터                     정상 데이터 범위
    M_NOM  = 4.0 kg              mass_total  3.0 ~ 5.0
    CD_NOM = 1.0                 Cd          0.5 ~ 1.5
    LAUNCH_NOM = 80도            launch_angle 70 ~ 90

  → 정상 데이터는 명목 주변에 퍼져 있어 하나의 one-class 모델이 "정상 봉투"를
    학습할 수 있고, 봉투를 벗어난 OOD 가 드러난다.

    python scripts/model_filter.py                     # 정상 200 + OOD 50
    python scripts/model_filter.py --out ... --Q 1e-3 --R 4.0

출력: data/simulated/residuals.npz
    flat    (M, 6) float64  flight_id, time, y, S, z(=y/sqrt(S)), is_powered
    cols    (6,)   str
    fl_ids  (N,)   int    비행 id
    fl_ood  (N,)   int    0=정상, 1=OOD
    fl_type (N,)   str    'normal' 또는 ood_type
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from psintel_sim import (  # noqa: E402
    Motor, isa_density, ELEVATION, G, PROPELLANT_MASS, REF_AREA,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOTOR_CSV = os.path.join(ROOT, "data", "motor", "thrust_curve_from_pressure.csv")

# 명목 파라미터 (모니터가 사전에 아는 값)
M_NOM, CD_NOM, LAUNCH_NOM, RAIL_LEN = 4.0, 1.0, 80.0, 2.0

COLS = ["flight_id", "time", "y", "S", "z", "is_powered"]


def _nominal_step(motor: Motor, z, v, t, dt):
    """명목 모델로 한 스텝 전진. 레일 구속(정지 유지) 포함."""
    uz = math.sin(math.radians(LAUNCH_NOM))
    m = M_NOM - PROPELLANT_MASS * min(t / motor.t_end, 1.0)
    rho = isa_density(ELEVATION + z)
    f = motor.at(t) - 0.5 * rho * CD_NOM * REF_AREA * abs(v) * v - m * G * uz
    if z < RAIL_LEN and v <= 0.0 and f <= 0.0:
        return 0.0, 0.0                      # 레일 위 정지 (후방 이동 금지)
    a = (motor.at(t) - 0.5 * rho * CD_NOM * REF_AREA * v * abs(v)) / m - G
    return z + v * dt + 0.5 * a * dt * dt, v + a * dt


def _jacobian(motor: Motor, z, v, t, dt):
    """해석적 야코비안 F. ∂v'/∂v = 1 - ρ Cd A |v| dt / m."""
    m = M_NOM - PROPELLANT_MASS * min(t / motor.t_end, 1.0)
    rho = isa_density(ELEVATION + z)
    return np.array([[1.0, dt],
                     [0.0, 1.0 - rho * CD_NOM * REF_AREA * abs(v) * dt / m]])


def run_flight(motor: Motor, fd: pd.DataFrame, Qd: float, Rd: float) -> tuple:
    """비행 1편 → (시각, y, S, is_powered)."""
    z0 = float(fd["altitude"].iloc[0])
    zm = fd["altitude_noisy"].to_numpy() - z0
    ts = fd["time"].to_numpy()
    powered = (fd["flight_phase"].astype(str).to_numpy() == "powered_ascent").astype(float) \
        if "flight_phase" in fd.columns else np.zeros(len(fd))

    x = np.array([float(zm[0]), 0.0])
    P = np.diag([4.0, 4.0])
    H = np.array([[1.0, 0.0]])
    ys, Ss = [], []

    for i in range(len(fd)):
        if i > 0:
            dt = float(ts[i] - ts[i - 1])
            zn, vn = _nominal_step(motor, x[0], x[1], float(ts[i - 1]), dt)
            F = _jacobian(motor, zn, vn, float(ts[i - 1]), dt)
            x = np.array([zn, vn])
            P = F @ P @ F.T + np.diag([Qd, Qd])
        y = float((zm[i] - (H @ x))[0])
        S = float((H @ P @ H.T)[0, 0]) + Rd
        K = P @ H.T / S
        x = x + (K * y).flatten()
        P = (np.eye(2) - K @ H) @ P
        ys.append(y)
        Ss.append(S)

    return ts, np.array(ys), np.array(Ss), powered


def collect(motor: Motor, df: pd.DataFrame, Qd: float, Rd: float,
            ood_flag: int, types: dict) -> tuple:
    rows, fl_ids, fl_ood, fl_type = [], [], [], []
    for fid in sorted(df["flight_id"].unique()):
        fd = df[df["flight_id"] == fid].reset_index(drop=True)
        ts, ys, Ss, pw = run_flight(motor, fd, Qd, Rd)
        z = ys / np.sqrt(Ss)
        rows.append(np.column_stack([np.full(len(ys), float(fid)), ts, ys, Ss, z, pw]))
        fl_ids.append(int(fid))
        fl_ood.append(ood_flag)
        fl_type.append(types.get(int(fid), "normal" if not ood_flag else "ood"))
    return np.vstack(rows), np.array(fl_ids), np.array(fl_ood), np.array(fl_type)


def main() -> int:
    ap = argparse.ArgumentParser(description="명목 물리모델 잔차 생성 (Step 5.5 용)")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "simulated", "residuals.npz"))
    ap.add_argument("--Q", type=float, default=1e-3, help="프로세스 노이즈 (기본 1e-3)")
    ap.add_argument("--R", type=float, default=4.0, help="측정 노이즈 (기본 4.0 = 2.0^2 m^2)")
    ap.add_argument("--no-ood", action="store_true", help="정상 데이터만 처리")
    args = ap.parse_args()

    motor = Motor.from_csv(MOTOR_CSV)
    ndf = pd.read_csv(os.path.join(ROOT, "data", "simulated", "all_trajectories.csv.gz"))
    nf, nid, nood, ntyp = collect(motor, ndf, args.Q, args.R, 0, {})
    print(f"정상 {len(nid)}편 / {len(nf)}행")

    if args.no_ood:
        flat, fid, ood, typ = nf, nid, nood, ntyp
    else:
        odf = pd.read_csv(os.path.join(ROOT, "data", "simulated", "ood", "ood_trajectories.csv.gz"))
        ometa = pd.read_csv(os.path.join(ROOT, "data", "simulated", "ood", "ood_metadata.csv.gz"))
        types = dict(zip(ometa["flight_id"].astype(int), ometa["ood_type"]))
        of, oid, oood, otyp = collect(motor, odf, args.Q, args.R, 1, types)
        print(f"OOD  {len(oid)}편 / {len(of)}행")
        flat = np.vstack([nf, of])
        fid = np.concatenate([nid, oid])
        ood = np.concatenate([nood, oood])
        typ = np.concatenate([ntyp, otyp])

    np.savez_compressed(args.out, flat=flat.astype(np.float64), cols=np.array(COLS),
                        fl_ids=fid, fl_ood=ood, fl_type=typ)
    print(f"\n저장: {os.path.relpath(args.out, ROOT)}  ({flat.shape[0]}행, {len(fid)}편)")

    # ── 빠른 탐지력 리포트 (정상 95분위 단일 임계 기준선) ──
    per = pd.DataFrame(flat, columns=COLS).groupby("flight_id")["z"].apply(
        lambda s: float(np.median(np.abs(s)))).rename("med")
    m = pd.DataFrame({"fid": fid, "ood": ood, "typ": typ}).merge(per, left_on="fid", right_index=True)
    thr = float(np.percentile(m.loc[m.ood == 0, "med"], 95))
    fpr = float((m.loc[m.ood == 0, "med"] > thr).mean() * 100)
    tpr = float((m.loc[m.ood == 1, "med"] > thr).mean() * 100)
    print(f"\n[기준선: 정상 95분위 단일 임계 {thr:.3f}]  FPR {fpr:.1f}% / TPR {tpr:.1f}%")
    for k, g in m[m.ood == 1].groupby("typ"):
        print(f"    {k:6s} n={len(g):2d}  중앙 {g['med'].median():7.3f}  탐지 {float((g['med']>thr).mean()*100):5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
