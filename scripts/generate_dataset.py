#!/usr/bin/env python3
"""PSIntelligence 학습 데이터 생성 (RocketPy 대체 경로).

    python scripts/generate_dataset.py                    # 200편 -> data/generated/
    python scripts/generate_dataset.py --n 50 --seed 7
    python scripts/generate_dataset.py --out data/mytest

정본(data/simulated/)과 **동일한 스키마**로 all_trajectories.csv.gz,
flight_metadata.csv.gz 를 씁니다. 기본 출력은 data/generated/ 이므로
정본을 덮어쓰지 않습니다.

생성 후 검증:
    python scripts/verify_dataset.py --data-dir data/generated
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from psintel_sim import (  # noqa: E402
    Motor, FlightSample, simulate, add_sensor_noise, add_derived,
    PROPELLANT_MASS, ELEVATION,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOTOR_CSV = os.path.join(ROOT, "data", "motor", "thrust_curve_from_pressure.csv")

# 정본과 동일한 컬럼 순서
TRAJ_COLS = ["time", "altitude", "velocity_z", "velocity_total", "acceleration_z",
             "tilt_angle", "gyro_roll", "gyro_pitch", "gyro_yaw",
             "acceleration_noisy", "altitude_noisy",
             "gyro_roll_noisy", "gyro_pitch_noisy", "gyro_yaw_noisy",
             "h_theoretical", "energy_ratio", "time_to_apogee", "flight_phase", "flight_id"]
META_COLS = ["apogee_altitude", "apogee_time", "burnout_time", "launch_angle", "wind_speed",
             "avg_drag_coeff", "mass_total", "max_speed", "cg_offset_x", "cg_offset_y",
             "cant_angle", "flight_id"]
# 정본에 없던 컬럼 (결함 D10 대응). 마지막에 덧붙인다.
META_EXTRA = ["wind_direction"]


def fmt(v):
    if isinstance(v, float):
        return repr(round(v, 9))
    return v


def write_gz_csv(path: str, header: list[str], rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for r in rows:
        w.writerow([fmt(r.get(c, "")) for c in header])
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as fh:
        fh.write(buf.getvalue())


def main() -> int:
    ap = argparse.ArgumentParser(description="PSIntelligence 데이터셋 생성 (RocketPy 불필요)")
    ap.add_argument("--n", type=int, default=200, help="비행 편수 (기본 200)")
    ap.add_argument("--seed", type=int, default=42, help="난수 시드 (기본 42)")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "generated"),
                    help="출력 디렉터리 (기본 data/generated)")
    ap.add_argument("--dt", type=float, default=0.02, help="적분/샘플 간격 (기본 0.02 s)")
    args = ap.parse_args()

    if not os.path.exists(MOTOR_CSV):
        print(f"추력 곡선이 없습니다: {MOTOR_CSV}", file=sys.stderr)
        return 1
    motor = Motor.from_csv(MOTOR_CSV)
    print(f"모터: 총임펄스 {motor.impulse:.1f} N·s, 연소 {motor.t_end:.2f} s")
    print(f"추진제 {PROPELLANT_MASS:.6f} kg, 발사대 {ELEVATION:.0f} m, 시드 {args.seed}, dt {args.dt}s")

    rng = random.Random(args.seed)
    traj_rows: list[dict] = []
    meta_rows: list[dict] = []
    t0 = time.perf_counter()

    for fid in range(args.n):
        sample = FlightSample.draw(rng)
        try:
            fl = simulate(motor, sample, dt=args.dt)
        except Exception as e:                                  # noqa: BLE001
            print(f"  fid {fid} 시뮬레이션 실패: {e}", file=sys.stderr)
            continue
        if fl["apogee_altitude"] <= ELEVATION + 10:
            continue
        fl = add_derived(add_sensor_noise(fl, rng))
        fid_out = len(meta_rows)          # 건너뛴 편이 있어도 0..N-1 연속 유지

        for i in range(len(fl["time"])):
            traj_rows.append({c: fl[c][i] for c in TRAJ_COLS if c != "flight_id"}
                             | {"flight_id": fid_out})
        meta_rows.append({
            "apogee_altitude": fl["apogee_altitude"], "apogee_time": fl["apogee_time"],
            "burnout_time": fl["burnout_time"], "launch_angle": sample.launch_angle,
            "wind_speed": sample.wind_speed, "avg_drag_coeff": sample.avg_drag_coeff,
            "mass_total": sample.mass_total, "max_speed": fl["max_speed"],
            "cg_offset_x": sample.cg_offset_x, "cg_offset_y": sample.cg_offset_y,
            "cant_angle": sample.cant_angle, "flight_id": fid_out,
            "wind_direction": sample.wind_direction,
        })

    wall = time.perf_counter() - t0
    write_gz_csv(os.path.join(args.out, "all_trajectories.csv.gz"), TRAJ_COLS, traj_rows)
    write_gz_csv(os.path.join(args.out, "flight_metadata.csv.gz"), META_COLS + META_EXTRA, meta_rows)

    print(f"\n생성 완료: {len(meta_rows)}편 / {len(traj_rows)}행 / {wall:.1f}s")
    print(f"  {os.path.join(args.out, 'all_trajectories.csv.gz')}")
    print(f"  {os.path.join(args.out, 'flight_metadata.csv.gz')}")
    print(f"\n검증:  python scripts/verify_dataset.py --data-dir {os.path.relpath(args.out, ROOT)}")
    print("\n주의: gyro_yaw 는 2D 축소 모델이라 0 으로 채워집니다 (SCHEMA.md §3.2 참조).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
