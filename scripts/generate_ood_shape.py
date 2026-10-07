#!/usr/bin/env python3
"""모양(shape)형 OOD 생성 — (a) 반증 실험용.

목적
----
기존 OOD(`generate_ood.py`)는 파라미터 이탈이라 잔차의 **크기**가 커진다.
χ²(Σz²)는 크기 변화에 정합된 검출기라 커널을 이긴다 — 이미 측정했다.

이 스크립트는 **분산은 정상과 동일하게 맞추고 분포의 모양만 바꾼** OOD 를 만든다.
그러면 Σz² 는 원리적으로 볼 수 없고, 모양을 보는 검출기(커널 one-class)만 잡을 수 있다.

계열 (전부 표본 표준편차를 정상과 동일한 sigma=1.5 m 로 강제 스케일)
--------------------------------------------------------------------
  ar1      AR(1) 상관 잡음 (rho=0.9) — 같은 분산, 시간 상관만 다름 (가장 순수한 모양 변화)
  t3       Student-t(3) 잡음        — 같은 분산, 첨도만 큼 (heavy tail)
  sine     협대역 정현파 + 소량 백색  — 같은 분산, 전력이 특정 주파수에 집중

고도 채널만 바꾼다 (가속도·자이로는 정상 스펙 유지) → 이상이 한 채널에 격리된다.

    python scripts/generate_ood_shape.py                    # 50편 -> data/simulated/ood_shape/
    python scripts/generate_ood_shape.py --n 50 --seed 7707
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from psintel_sim import (  # noqa: E402
    Motor, FlightSample, simulate, add_sensor_noise, add_derived, ELEVATION,
)
from generate_dataset import TRAJ_COLS, META_COLS, META_EXTRA, write_gz_csv  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOTOR_CSV = os.path.join(ROOT, "data", "motor", "thrust_curve_from_pressure.csv")

SHAPE_ID_BASE = 2000          # 정상 0~199, OOD(범위) 1000~1049 와 충돌 방지
SIGMA_ALT = 1.5               # 정상 고도 잡음 스펙 (m) — 이 값을 강제로 맞춘다
QUANT_ALT = 0.5               # 정상 양자화 스텝 (m)
FAMILIES = ["ar1", "t3", "sine"]


def shaped_noise(fam: str, n: int, rng: random.Random, dt: float) -> np.ndarray:
    """정상과 표본 표준편차가 정확히 같은 모양 변형 잡음을 만든다."""
    r = np.random.default_rng(rng.randrange(1 << 30))
    if fam == "ar1":
        rho = 0.9
        w = r.standard_normal(n)
        e = np.zeros(n)
        for i in range(1, n):
            e[i] = rho * e[i - 1] + math.sqrt(1 - rho ** 2) * w[i]
        e[0] = w[0]
    elif fam == "t3":
        e = r.standard_t(3, n)
    elif fam == "sine":
        f = r.uniform(1.0, 4.0)                      # 1~4 Hz 협대역
        t = np.arange(n) * dt
        e = np.sin(2 * math.pi * f * t + r.uniform(0, 2 * math.pi))
        e = e + 0.3 * r.standard_normal(n)           # 소량 백색 (광대역 꼬리)
    else:
        raise ValueError(fam)
    e = e - e.mean()
    s = e.std()
    if s < 1e-12:
        return np.zeros(n)
    return e * (SIGMA_ALT / s)                       # ← 표본 분산을 정확히 일치시킴


def main() -> int:
    ap = argparse.ArgumentParser(description="모양형 OOD 생성 (반증 실험 (a))")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7707)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "simulated", "ood_shape"))
    ap.add_argument("--dt", type=float, default=0.02)
    args = ap.parse_args()

    motor = Motor.from_csv(MOTOR_CSV)
    rng = random.Random(args.seed)
    traj, meta = [], []
    t0 = time.perf_counter()

    while len(meta) < args.n:
        fam = FAMILIES[len(meta) % len(FAMILIES)]
        sample = FlightSample.draw(rng)                       # 파라미터는 전부 정상 범위
        try:
            fl = simulate(motor, sample, dt=args.dt)
        except Exception as e:                                # noqa: BLE001
            print(f"  시뮬 실패: {e}", file=sys.stderr)
            continue
        if fl["apogee_altitude"] <= ELEVATION + 10:
            continue

        fl = add_sensor_noise(fl, rng)                        # 정상 노이즈 먼저
        # 고도 채널만 모양 변형 잡음으로 교체 (양자화는 정상과 동일하게 적용)
        n = len(fl["time"])
        shaped = shaped_noise(fam, n, rng, args.dt)
        alt = np.asarray(fl["altitude"], dtype=float)
        # numpy 스칼라를 그대로 넘기면 repr() 이 'np.float64(...)' 문자열이 되어
        # CSV 에 박힌다 (numpy 2.x). 반드시 Python float 로 변환한다.
        noisy = np.round((alt + shaped) / QUANT_ALT) * QUANT_ALT
        fl["altitude_noisy"] = [float(x) for x in noisy]

        fl = add_derived(fl)
        fid = SHAPE_ID_BASE + len(meta)
        for i in range(n):
            traj.append({c: fl[c][i] for c in TRAJ_COLS if c != "flight_id"} | {"flight_id": fid})
        meta.append({
            "apogee_altitude": fl["apogee_altitude"], "apogee_time": fl["apogee_time"],
            "burnout_time": fl["burnout_time"], "launch_angle": sample.launch_angle,
            "wind_speed": sample.wind_speed, "avg_drag_coeff": sample.avg_drag_coeff,
            "mass_total": sample.mass_total, "max_speed": fl["max_speed"],
            "cg_offset_x": sample.cg_offset_x, "cg_offset_y": sample.cg_offset_y,
            "cant_angle": sample.cant_angle, "flight_id": fid,
            "wind_direction": sample.wind_direction, "ood_type": fam,
        })

    wall = time.perf_counter() - t0
    write_gz_csv(os.path.join(args.out, "ood_trajectories.csv.gz"), TRAJ_COLS, traj)
    write_gz_csv(os.path.join(args.out, "ood_metadata.csv.gz"),
                 META_COLS + META_EXTRA + ["ood_type"], meta)
    print(f"모양형 OOD {len(meta)}편 / {len(traj)}행 / {wall:.1f}s → {os.path.relpath(args.out, ROOT)}/")
    print("  계열 분포:", dict(Counter(m["ood_type"] for m in meta)))
    print(f"  flight_id: {SHAPE_ID_BASE}~{SHAPE_ID_BASE + len(meta) - 1}")
    print(f"  고도 잡음: 표본 sigma = {SIGMA_ALT} m 로 강제 (정상과 동일)")
    print("  ⚠️ 파라미터는 전부 정상 범위 — 크기가 아니라 '모양'만 다르다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
