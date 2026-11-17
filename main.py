"""PSIntelligence minimal inference demo.

Usage:
    python main.py --csv data/112.txt

Parses a raw flight log, converts it to the shared AGL/inertial feature
contract, and prints raw apogee diagnostics.  Projection below the current
altitude is an optional diagnostic, not the default model output.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

import apogee_pipeline as ap
from apogee_models import (
    GRUApogeeModel,
    project_apogee,
    raw_apogee,
    validate_physics_inputs,
)

WINDOW = 100
LOG_RE = re.compile(
    r"Time:\s*([\d\.]+)s.*"
    r"Euler:\s*\(([-\d\.]+),\s*([-\d\.]+),\s*([-\d\.]+)\).*"
    r"Gyro:\s*\(([-\d\.]+),\s*([-\d\.]+),\s*([-\d\.]+)\).*"
    r"Accel:\s*\([^,]+,\s*[^,]+,\s*([-\d\.]+)\).*"
    r"kal:\s*\(alt\s*:\s*([\d\.]+),\s*velo\s*:\s*([\d\.]+)"
)


def parse_log(path: Path) -> pd.DataFrame:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    launch_time = None
    rows = []
    for line in lines:
        m = LOG_RE.search(line)
        if not m:
            continue
        t_sys = float(m.group(1))
        acc_g = float(m.group(8))
        if launch_time is None and acc_g > 0.5:
            launch_time = t_sys
        if launch_time is None:
            continue
        alt_abs = float(m.group(9))
        vel = float(m.group(10))
        acc_ms = acc_g * 9.80665
        pitch = float(m.group(3))
        gyro_roll = np.deg2rad(float(m.group(7)))
        rows.append({
            "flight_id": 0,
            "time": t_sys - launch_time,
            "altitude": alt_abs,
            "altitude_noisy": alt_abs,
            "velocity_z": vel,
            "accel_g": acc_g,  # already gravity-removed in the logger
            "acceleration_z": acc_ms,
            "tilt_angle": -abs(pitch),
            "gyro_roll": gyro_roll,
            "gyro_roll_noisy": gyro_roll,
        })
    if not rows:
        raise SystemExit("no usable rows in log")
    return pd.DataFrame(rows)


def _prepare_scaler(scaler, accel_encoding: str, metadata: dict | None):
    if accel_encoding == "auto":
        encoding = ap.resolve_acceleration_encoding(metadata)
    elif accel_encoding == "legacy":
        encoding = "legacy_specific_force"
    elif accel_encoding == "inertial":
        encoding = "inertial"
    else:
        raise SystemExit(f"unsupported acceleration encoding: {accel_encoding}")
    try:
        return ap.prepare_scaler_for_encoding(scaler, encoding), encoding
    except ap.PipelineError as exc:
        raise SystemExit(str(exc)) from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/112.txt", help="raw flight log")
    parser.add_argument("--model", default="data/simulated/best_gru_model.pth")
    parser.add_argument("--scaler", default="data/simulated/input_scaler.pkl")
    parser.add_argument("--metadata", default="data/simulated/model_metadata.pkl")
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--show-projected", action="store_true",
                        help="also print max(H_raw, h) diagnostics")
    parser.add_argument(
        "--accel-encoding",
        choices=["auto", "legacy", "inertial"],
        default="auto",
        help="auto reads metadata; legacy adapts once; inertial assumes new scaler",
    )
    args = parser.parse_args()

    metadata = joblib.load(args.metadata) if Path(args.metadata).exists() else None
    if metadata:
        metadata_features = list(metadata.get("feature_cols", []))
        if metadata_features and metadata_features not in (
            list(ap.FEATURE_COLUMNS), list(ap.LEGACY_FEATURE_COLUMNS)
        ):
            raise SystemExit("metadata feature_cols do not match a supported contract")
    scaler = joblib.load(args.scaler)
    scaler, resolved_encoding = _prepare_scaler(scaler, args.accel_encoding, metadata)
    hidden_dim = int((metadata or {}).get("hidden_dim", 128))
    num_layers = int((metadata or {}).get("num_layers", 2))
    dropout = float((metadata or {}).get("dropout", 0.2))
    model = GRUApogeeModel(
        input_dim=len(ap.FEATURE_COLUMNS),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    state = torch.load(args.model, map_location="cpu")
    model.load_legacy_state_dict(state)
    model.eval()

    raw = parse_log(Path(args.csv))
    frame = ap.make_real_features(raw)
    ap.validate_time_order(frame)
    columns = tuple(dict.fromkeys([*ap.FEATURE_COLUMNS, "altitude_rel"]))
    frame = ap.resample_causal(frame, columns=columns)
    if len(frame) < args.window:
        raise SystemExit(f"log too short: {len(frame)} rows < window {args.window}")

    endpoints = np.arange(args.window - 1, len(frame), 10, dtype=int)
    if endpoints.size == 0:
        raise SystemExit("no complete causal window")
    values = frame[list(ap.FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
    scaled = scaler.transform(values)
    x = torch.tensor(ap.slice_history(scaled, endpoints, args.window), dtype=torch.float32)
    h = torch.tensor(frame["altitude_rel"].values[endpoints], dtype=torch.float32)
    v = torch.tensor(frame["velocity_z"].values[endpoints], dtype=torch.float32)
    denom = validate_physics_inputs(h, v)

    with torch.no_grad():
        r_raw, _ = model(x)
        h_raw = raw_apogee(h, v, r_raw)
        h_projected, r_projected, below = project_apogee(h, v, r_raw)

    logged_max = float(frame["altitude_rel"].max())
    print(f"rows={len(frame)} window={args.window}")
    print(f"raw r (last window):            {r_raw[-1].item():.4f}")
    print(f"raw apogee AGL (last window):   {h_raw[-1].item():.2f} m")
    print(f"raw error vs logged max:        {h_raw[-1].item() - logged_max:.2f} m")
    print(f"constant-r (r=1) AGL:           {denom[-1].item():.2f} m")
    print(f"below-current flag (last):      {bool(below[-1].item())}")
    print(f"below-current rate (all):       {float(below.float().mean()) * 100:.1f}%")
    if args.show_projected:
        print(f"projected r (last window):      {r_projected[-1].item():.4f}")
        print(f"projected apogee AGL:           {h_projected[-1].item():.2f} m")
        print(f"projected error vs logged max:  {h_projected[-1].item() - logged_max:.2f} m")


if __name__ == "__main__":
    main()
