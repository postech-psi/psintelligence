#!/usr/bin/env python3
"""Step 5.5 용 OOD(분포 이탈) 비행 데이터 생성.

정상 데이터(data/simulated/)는 전부 정상 범위 샘플이라 이상 탐지 실험에 쓸
OOD 비행이 없습니다. 이 스크립트가 정상 범위를 **의도적으로 벗어난**
파라미터로 비행을 생성해 OOD 세트를 만듭니다.

    python scripts/generate_ood.py                       # 50편 -> data/simulated/ood/
    python scripts/generate_ood.py --n 50 --seed 5505

OOD 계열 (정상 범위 대비)
------------------------
  mass      질량 초과      mass_total  6.0~8.0  (정상 3.0~5.0)
  drag      항력 초과      Cd          1.8~2.6  (정상 0.5~1.5)
  wind      강풍           wind_speed 14~24 m/s  (정상 0~10), 발사 방위 근처
  loft      저각 발사      launch_angle 50~65도 (정상 70~90)

⚠️ wind 계열 주의: 자체 시뮬레이터는 발사 방위 성분만 반영하므로
   wind_direction 을 0 근처로 두어야 실제로 영향이 생깁니다
   (90도 = 횡성분 = 영향 0).
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from psintel_sim import (  # noqa: E402
    Motor, FlightSample, simulate, add_sensor_noise, add_derived, ELEVATION,
)
from generate_dataset import TRAJ_COLS, META_COLS, META_EXTRA, write_gz_csv  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOTOR_CSV = os.path.join(ROOT, "data", "motor", "thrust_curve_from_pressure.csv")

# flight_id 충돌 방지: 정상 0~199, 생성기 0~N, OOD 는 1000 부터
OOD_ID_BASE = 1000

# (계열명 → 이탈시킬 파라미터와 (분포, 모수))
OOD_FAMILIES = {
    "mass": {"mass_total": ("uniform", 6.0, 8.0)},
    "drag": {"avg_drag_coeff": ("uniform", 1.8, 2.6)},
    "wind": {"wind_speed": ("uniform", 14.0, 24.0), "wind_direction": ("uniform", 0.0, 30.0)},
    "loft": {"launch_angle": ("uniform", 50.0, 65.0)},
}
# 계열 선택 확률 (균등)
FAMILY_WEIGHTS = [1.0] * len(OOD_FAMILIES)


def draw_ood(rng: random.Random) -> tuple[str, FlightSample]:
    """정상 범위 기준으로 뽑은 뒤 한 계열만 이탈시킨다."""
    name = rng.choices(list(OOD_FAMILIES), weights=FAMILY_WEIGHTS)[0]
    overrides = OOD_FAMILIES[name]
    base = FlightSample.draw(rng)                       # 정상 분포
    kw = {}
    for key, spec in overrides.items():
        if spec[0] == "uniform":
            kw[key] = rng.uniform(spec[1], spec[2])
        else:
            kw[key] = rng.gauss(spec[1], spec[2])
    return name, replace(base, **kw)


def main() -> int:
    ap = argparse.ArgumentParser(description="OOD 비행 데이터 생성 (Step 5.5 용)")
    ap.add_argument("--n", type=int, default=50, help="OOD 편수 (기본 50)")
    ap.add_argument("--seed", type=int, default=5505, help="난수 시드 (기본 5505)")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "simulated", "ood"))
    ap.add_argument("--dt", type=float, default=0.02)
    args = ap.parse_args()

    motor = Motor.from_csv(MOTOR_CSV)
    rng = random.Random(args.seed)
    traj, meta = [], []
    t0 = time.perf_counter()

    while len(meta) < args.n:
        name, sample = draw_ood(rng)
        try:
            fl = simulate(motor, sample, dt=args.dt)
        except Exception as e:                                   # noqa: BLE001
            print(f"  시뮬 실패: {e}", file=sys.stderr)
            continue
        if fl["apogee_altitude"] <= ELEVATION + 10:
            continue
        fl = add_derived(add_sensor_noise(fl, rng))
        fid = OOD_ID_BASE + len(meta)

        for i in range(len(fl["time"])):
            traj.append({c: fl[c][i] for c in TRAJ_COLS if c != "flight_id"} | {"flight_id": fid})
        meta.append({
            "apogee_altitude": fl["apogee_altitude"], "apogee_time": fl["apogee_time"],
            "burnout_time": fl["burnout_time"], "launch_angle": sample.launch_angle,
            "wind_speed": sample.wind_speed, "avg_drag_coeff": sample.avg_drag_coeff,
            "mass_total": sample.mass_total, "max_speed": fl["max_speed"],
            "cg_offset_x": sample.cg_offset_x, "cg_offset_y": sample.cg_offset_y,
            "cant_angle": sample.cant_angle, "flight_id": fid,
            "wind_direction": sample.wind_direction, "ood_type": name,
        })

    wall = time.perf_counter() - t0
    write_gz_csv(os.path.join(args.out, "ood_trajectories.csv.gz"), TRAJ_COLS, traj)
    write_gz_csv(os.path.join(args.out, "ood_metadata.csv.gz"),
                 META_COLS + META_EXTRA + ["ood_type"], meta)

    from collections import Counter
    cnt = Counter(m["ood_type"] for m in meta)
    print(f"OOD {len(meta)}편 / {len(traj)}행 / {wall:.1f}s  → {os.path.relpath(args.out, ROOT)}/")
    print("  계열 분포:", dict(cnt))
    print(f"  flight_id: {OOD_ID_BASE}~{OOD_ID_BASE + len(meta) - 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
