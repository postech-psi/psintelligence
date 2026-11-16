"""PSIntelligence minimal inference demo.

Usage:
    python main.py --csv data/112.txt

Loads the Step 4 GRU checkpoint and scaler, parses a raw flight log, and
prints the predicted apogee at the last available window.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

FEATURES = [
    "altitude_noisy",
    "velocity_z",
    "acceleration_noisy",
    "tilt_angle",
    "gyro_roll_noisy",
    "dynamic_pressure",
]
WINDOW = 100
G = 9.81
LOG_RE = re.compile(
    r"Time:\s*([\d\.]+)s.*"
    r"Euler:\s*\(([-\d\.]+),\s*([-\d\.]+),\s*([-\d\.]+)\).*"
    r"Gyro:\s*\(([-\d\.]+),\s*([-\d\.]+),\s*([-\d\.]+)\).*"
    r"Accel:\s*\([^,]+,\s*[^,]+,\s*([-\d\.]+)\).*"
    r"kal:\s*\(alt\s*:\s*([\d\.]+),\s*velo\s*:\s*([\d\.]+)"
)


class GRUModel(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1), nn.Tanh(),
        )

    def forward(self, x, h_curr, v_z):
        out, _ = self.gru(x)
        r = 1.0 + 0.7 * self.fc(out[:, -1, :]).squeeze(-1)
        return h_curr + (v_z ** 2) / (2 * G) * r, r


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

        alt = float(m.group(9))
        vel = float(m.group(10))
        acc_ms = acc_g * 9.80665
        pitch = float(m.group(3))
        gyro_roll = np.deg2rad(float(m.group(7)))
        rho = 1.225 * np.exp(-0.00012 * abs(alt))
        rows.append({
            "time": t_sys - launch_time,
            "altitude": alt,
            "velocity_z": vel,
            "acceleration_z": acc_ms,
            "tilt_angle": -abs(pitch),
            "gyro_roll": gyro_roll,
            "dynamic_pressure": 0.5 * rho * vel ** 2,
            "altitude_noisy": alt,
            "acceleration_noisy": acc_ms,
            "gyro_roll_noisy": gyro_roll,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/112.txt", help="raw flight log")
    ap.add_argument("--model", default="data/simulated/best_gru_model.pth")
    ap.add_argument("--scaler", default="data/simulated/input_scaler.pkl")
    ap.add_argument("--window", type=int, default=WINDOW)
    args = ap.parse_args()

    scaler = joblib.load(args.scaler)
    model = GRUModel(input_dim=len(FEATURES))
    state = torch.load(args.model, map_location="cpu")
    # Checkpoints are saved from PhysicsInformedGRU, so strip the "gru." prefix.
    state = {k[4:] if k.startswith("gru.") else k: v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    df = parse_log(Path(args.csv))
    if len(df) < args.window:
        raise SystemExit(f"log too short: {len(df)} rows < window {args.window}")

    x = torch.tensor(scaler.transform(df[FEATURES].values), dtype=torch.float32)
    x = x.unfold(0, args.window, 1).transpose(-1, -2)  # (N, window, features)
    h = torch.tensor(df["altitude"].values[args.window - 1:], dtype=torch.float32)
    v = torch.tensor(df["velocity_z"].values[args.window - 1:], dtype=torch.float32)

    with torch.no_grad():
        apogee, ratio = model(x, h, v)

    # Constant-r baseline (r=1): physics-only prediction.
    baseline = h + (v ** 2) / (2 * G)
    print(f"rows={len(df)} window={args.window}")
    print(f"predicted apogee (last window): {apogee[-1].item():.2f} m")
    print(f"predicted r (last window):      {ratio[-1].item():.4f}")
    print(f"constant-r (r=1) baseline:      {baseline[-1].item():.2f} m")
    print(f"physics gain:                   {((v[-1] ** 2) / (2 * G)).item():.2f} m")


if __name__ == "__main__":
    main()
