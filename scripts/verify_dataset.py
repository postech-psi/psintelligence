#!/usr/bin/env python3
"""PSIntelligence 데이터셋 무결성 검증.

    python scripts/verify_dataset.py

data/SCHEMA.md §6 의 검사 항목을 수행합니다.
데이터를 수정하거나 Step 1.5 로 재생성한 뒤 반드시 실행하세요.
종료 코드 0 = 전체 통과, 1 = 실패.
"""
from __future__ import annotations

import csv
import gzip
import io
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM = os.path.join(ROOT, "data", "simulated")
G = 9.81

# 정본(파이프라인이 읽는 파일)
CANON_TRAJ = os.path.join(SIM, "all_trajectories.csv.gz")
CANON_META = os.path.join(SIM, "flight_metadata.csv.gz")
# 레거시(노트북이 읽지 않음)
LEGACY_TRAJ = os.path.join(SIM, "all_trajectories.csv")
LEGACY_META = os.path.join(SIM, "flight_metadata.csv")

CANON_TRAJ_COLS = {
    "time", "altitude", "velocity_z", "velocity_total", "acceleration_z", "tilt_angle",
    "gyro_roll", "gyro_pitch", "gyro_yaw", "acceleration_noisy", "altitude_noisy",
    "gyro_roll_noisy", "gyro_pitch_noisy", "gyro_yaw_noisy", "h_theoretical",
    "energy_ratio", "time_to_apogee", "flight_phase", "flight_id",
}
CANON_META_COLS = {
    "apogee_altitude", "apogee_time", "burnout_time", "launch_angle", "wind_speed",
    "avg_drag_coeff", "mass_total", "max_speed", "cg_offset_x", "cg_offset_y",
    "cant_angle", "flight_id",
}

_fail: list[str] = []
_pass = 0


def check(name: str, ok: bool, detail: str = "") -> bool:
    global _pass
    if ok:
        _pass += 1
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
    else:
        _fail.append(name)
        print(f"  FAIL  {name}" + (f"  -> {detail}" if detail else ""))
    return ok


def is_gzip(path: str) -> bool:
    with open(path, "rb") as f:
        return f.read(2) == b"\x1f\x8b"


def load(path: str, expect_gzip: bool | None = None) -> list[dict]:
    raw = open(path, "rb").read()
    if raw[:2] == b"\x1f\x8b":
        if expect_gzip is False:
            raise ValueError(f"{os.path.basename(path)}: 내용은 gzip 인데 일반 CSV 로 기대됨")
        raw = gzip.decompress(raw)
    elif expect_gzip is True:
        raise ValueError(f"{os.path.basename(path)}: gzip 이 아님")
    return list(csv.DictReader(io.StringIO(raw.decode())))


print("=" * 62)
print("PSIntelligence 데이터셋 무결성 검증")
print("=" * 62)

# ---- 1. 존재 / 형식 -------------------------------------------------
print("\n[1] 정본 파일 존재·형식")
for p in (CANON_TRAJ, CANON_META):
    if not check(f"{os.path.basename(p)} 존재", os.path.exists(p)):
        print("\n정본 파일이 없습니다. 중단.")
        sys.exit(1)
check("정본 궤적이 실제 gzip", is_gzip(CANON_TRAJ))
check("정본 메타가 실제 gzip", is_gzip(CANON_META))

traj = load(CANON_TRAJ, expect_gzip=True)
meta = load(CANON_META, expect_gzip=True)

# ---- 2. 커버리지 -----------------------------------------------------
print("\n[2] 스키마·커버리지")
check("정본 궤적 컬럼 집합 일치", set(traj[0]) == CANON_TRAJ_COLS,
      f"차이 {set(traj[0]) ^ CANON_TRAJ_COLS}" if set(traj[0]) != CANON_TRAJ_COLS else "")
check("정본 메타 컬럼 집합 일치", set(meta[0]) == CANON_META_COLS,
      f"차이 {set(meta[0]) ^ CANON_META_COLS}" if set(meta[0]) != CANON_META_COLS else "")

mids = sorted(int(r["flight_id"]) for r in meta)
tids = sorted({int(r["flight_id"]) for r in traj})
check("메타 flight_id 중복 없음", len(mids) == len(set(mids)))
check("메타 flight_id 연속 (0..N-1)", mids == list(range(len(mids))), f"n={len(mids)}")
check("궤적↔메타 flight_id 완전 일치", set(tids) == set(mids),
      f"궤적만 {sorted(set(tids) - set(mids))[:5]} / 메타만 {sorted(set(mids) - set(tids))[:5]}")

# ---- 3~4. 파생식 ----------------------------------------------------
print("\n[3] h_theoretical == altitude + velocity_z^2 / (2*9.81)")
worst = 0.0
for r in traj:
    h = float(r["altitude"]); vz = float(r["velocity_z"]); ht = float(r["h_theoretical"])
    worst = max(worst, abs(ht - (h + vz * vz / (2 * G))))
check("전 행 일치 (오차 < 1e-3, 저장 정밀도 한계)", worst < 1e-3, f"최대오차 {worst:.3e}")

print("\n[4] energy_ratio == apogee_altitude / h_theoretical")
apogee = {int(r["flight_id"]): float(r["apogee_altitude"]) for r in meta}
worst = 0.0
for r in traj:
    f = int(r["flight_id"]); ht = float(r["h_theoretical"])
    if ht <= 0:
        continue
    worst = max(worst, abs(float(r["energy_ratio"]) - apogee[f] / ht))
check("전 행 일치 (오차 < 1e-4)", worst < 1e-4, f"최대오차 {worst:.3e}")

# ---- 5. 모터 고정 ----------------------------------------------------
print("\n[5] burnout_time 전편 동일 (모터 고정 가정)")
bt = {round(float(r["burnout_time"]), 6) for r in meta}
check("고유값 1개", len(bt) == 1, f"고유값 {sorted(bt)[:5]}")

# ---- 6. 물리 정합성 --------------------------------------------------
print("\n[6] 물리 정합성")
first_seen: dict[int, tuple[float, float]] = {}
for r in traj:
    f = int(r["flight_id"])
    t = float(r["time"])
    if f not in first_seen or t < first_seen[f][0]:
        first_seen[f] = (t, float(r["altitude"]))
h0s = {round(v[1], 1) for v in first_seen.values()}
check("모든 비행이 동일 발사대 고도에서 시작", len(h0s) == 1, f"고유값 {sorted(h0s)[:3]}")

neg = [r["flight_id"] for r in meta if float(r["mass_total"]) <= 0]
check("mass_total > 0", not neg, f"위반 fid {neg[:5]}")
bad_ap = [r["flight_id"] for r in meta
          if float(r["apogee_altitude"]) < min(h0s or [0])]
check("apogee_altitude >= 발사대 고도", not bad_ap, f"위반 fid {bad_ap[:5]}")

nonfinite = 0
for r in traj:
    for k in ("altitude", "velocity_z", "acceleration_z", "h_theoretical"):
        if not math.isfinite(float(r[k])):
            nonfinite += 1
check("궤적 수치 유한", nonfinite == 0, f"비유한 {nonfinite}건")

dt = {round(float(b["time"]) - float(a["time"]), 6)
      for a, b in zip(traj[:-1], traj[1:])
      if int(a["flight_id"]) == int(b["flight_id"])}
check("궤적 시간 간격 일정", len(dt) <= 1, f"고유 간격 {sorted(dt)[:4]}")

# ---- 7. 레거시 혼입 --------------------------------------------------
print("\n[7] 레거시 파일이 정본과 혼입되지 않았는지")
try:
    if os.path.exists(LEGACY_TRAJ):
        lt = load(LEGACY_TRAJ)
        check("레거시 궤적 컬럼 != 정본 컬럼", set(lt[0]) != CANON_TRAJ_COLS)
except Exception as e:  # noqa: BLE001
    check("레거시 궤적 로드", False, str(e))
try:
    if os.path.exists(LEGACY_META):
        lm = load(LEGACY_META)
        check("레거시 메타가 일반 CSV (확장자/내용 일치)", not is_gzip(LEGACY_META))
        check("레거시 메타에 mass_dry 존재", "mass_dry" in lm[0])
        if "mass_dry" in lm[0]:
            bad = [r["flight_id"] for r in lm if float(r["mass_dry"]) >= float(r["mass_total"])]
            check("레거시 mass_dry < mass_total (추진제 > 0)", not bad, f"위반 fid {bad[:5]}")
except Exception as e:  # noqa: BLE001
    check("레거시 메타 로드", False, str(e))

# ---- 요약 ------------------------------------------------------------
print("\n" + "=" * 62)
print(f"통과 {_pass} · 실패 {len(_fail)}")
if _fail:
    print("실패 항목:")
    for f in _fail:
        print(f"  - {f}")
    print("=" * 62)
    sys.exit(1)
print("전체 통과 ✅")
print("=" * 62)
