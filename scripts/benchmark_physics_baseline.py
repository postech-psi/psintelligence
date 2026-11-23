#!/usr/bin/env python
"""Compare the coasting physics baseline with the saved GRU on matched endpoints.

The physics baseline (``apogee_physics``) needs no training: it detects coast
causally, fits the drag constant ``k`` from gravity-removed acceleration and
evaluates the closed-form ballistic apex.  This script

1. builds the common 50 Hz grid and endpoint manifest (window 100, stride 10),
2. runs the physics baseline on the chosen state source,
3. optionally runs the saved GRU checkpoint on the same endpoints (needs torch),
4. scores both only where both are defined (coast endpoints), per split and by
   time-to-apogee band, and
5. builds an honest reference for the real log (``data/112.txt``): the log
   stops while climbing, so its maximum is a lower bound, not the apogee.

Outputs go only to ``--output-dir``; no artifact under data/simulated changes.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import apogee_pipeline as ap  # noqa: E402
import apogee_physics as phy  # noqa: E402

BANDS = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 10.0)
LOG_PATTERN = re.compile(
    r"Time:\s*([\d\.]+)s.*"
    r"Euler:\s*\(([-\d\.]+),\s*([-\d\.]+),\s*([-\d\.]+)\).*"
    r"Gyro:\s*\(([-\d\.]+),\s*([-\d\.]+),\s*([-\d\.]+)\).*"
    r"Accel:\s*\([^,]+,\s*[^,]+,\s*([-\d\.]+)\).*"
    r"kal:\s*\(alt\s*:\s*([\d\.]+),\s*velo\s*:\s*([-\d\.]+)"
)
BARO_PATTERN = re.compile(r"Alt:\s*([-\d\.]+)m")


def _rel(path: Path) -> str:
    """Repo-relative POSIX path when possible (keeps local folders out of outputs)."""
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(path).name


def load_sim_raw(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, compression="infer")
    frame = ap.add_relative_labels(ap.make_sim_features(frame))
    ap.validate_features(frame)
    return frame


def m7_ukf_grid(raw: pd.DataFrame, executable: Path) -> pd.DataFrame:
    """Replay the flight firmware UKF (``flight-computer/tools/ukf_replay``).

    Every simulated row offers one accelerometer and one barometer sample, as
    in the M7 replay contract (``flight_id,time_us,accel_z_g,baro_alt``).  The
    NN keeps the noisy barometer channel; h/v/q and the physics inputs use the
    filter's altitude and velocity.  Rows the replay marks invalid (velocity
    recovery after re-initialisation) become invalid grid rows.

    Uses the same replay and grid contract as ``benchmark_apogee.py
    --state-source m7_ukf`` so both benchmarks see identical states.
    """
    import benchmark_apogee as bench

    ordered = raw.sort_values(["flight_id", "time"]).reset_index(drop=True)
    replay = bench._run_ukf_replay(ordered, ordered["altitude_noisy_rel"], executable)
    grid, _ = bench._grid_with_m7_state(ordered, replay)
    return grid


def apply_state_source(sampled: pd.DataFrame, source: str) -> pd.DataFrame:
    """Return a grid whose h/v/q columns come from ``source``.

    ``estimated`` is ``ap.causal_state_estimate`` (dead reckoning from the
    first sample), not the firmware UKF; use ``m7_ukf`` for that.  Labels stay
    the true apogee.
    """
    if source == "simulation_clean":
        return sampled
    if source != "estimated":
        raise ValueError(f"unsupported state source: {source}")
    state = ap.causal_state_estimate(sampled)
    out = sampled.merge(state, on=["flight_id", "time"], how="left", validate="one_to_one")
    out["altitude_rel"] = out["h_est"]
    out["altitude_noisy_rel"] = out["h_est"]
    out["velocity_z"] = out["v_est"]
    out["dynamic_pressure"] = 0.5 * 1.225 * np.exp(-0.00012 * out["altitude_rel"].abs()) * out["velocity_z"] ** 2
    return out


def gru_predictions(grid: pd.DataFrame, endpoints: pd.DataFrame, checkpoint: Path,
                    scaler_path: Path) -> np.ndarray:
    import joblib
    import torch

    from apogee_models import GRUApogeeModel

    state = torch.load(checkpoint, map_location="cpu")
    keys = [k for k in state if k.endswith("weight_hh_l0")]
    if len(keys) != 1:
        raise RuntimeError(f"cannot infer GRU size from {checkpoint}")
    hidden = int(state[keys[0]].shape[1])
    layers = len([k for k in state if re.search(r"weight_hh_l\d+$", k)])
    model = GRUApogeeModel(hidden_dim=hidden, num_layers=layers)
    model.load_legacy_state_dict(state)
    model.eval()
    scaler = joblib.load(scaler_path)
    window = 100
    out = np.full(len(endpoints), np.nan)
    for flight_id, rows in endpoints.groupby("flight_id", sort=False):
        g = grid[grid["flight_id"] == flight_id].reset_index(drop=True)
        x = scaler.transform(g[list(ap.FEATURE_COLUMNS)].to_numpy(dtype=float))
        idx = rows["endpoint_index"].to_numpy(dtype=int)
        seq = torch.as_tensor(ap.slice_history(x, idx, window), dtype=torch.float32)
        with torch.no_grad():
            r, _ = model(seq)
        h = g["altitude_rel"].to_numpy(dtype=float)[idx]
        v = g["velocity_z"].to_numpy(dtype=float)[idx]
        out[rows.index.to_numpy()] = (h + v**2 / (2 * ap.SIM_G)) * r.numpy()
    return out


def summarize(frame: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    rows = []
    for (split, band), group in frame.groupby(["split", "band"], observed=True):
        row = {"split": split, "time_to_apogee_s": str(band), "n": len(group)}
        for name, col in columns.items():
            err = group[col] - group["apogee_rel"]
            row[f"{name}_mae_m"] = float(err.abs().mean())
            row[f"{name}_bias_m"] = float(err.mean())
            row[f"{name}_p95_abs_m"] = float(err.abs().quantile(0.95))
        rows.append(row)
    for split, group in frame.groupby("split"):
        row = {"split": split, "time_to_apogee_s": "all", "n": len(group)}
        for name, col in columns.items():
            err = group[col] - group["apogee_rel"]
            row[f"{name}_mae_m"] = float(err.abs().mean())
            row[f"{name}_bias_m"] = float(err.mean())
            row[f"{name}_p95_abs_m"] = float(err.abs().quantile(0.95))
        rows.append(row)
    return pd.DataFrame(rows)


def load_real_log(path: Path) -> pd.DataFrame:
    """Parse the legacy-firmware text log onto the common 50 Hz grid (AGL)."""
    records, launch, baro_pad = [], None, None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LOG_PATTERN.search(line)
        if not m:
            continue
        t, accel_g = float(m.group(1)), float(m.group(8))
        if launch is None and accel_g > 0.5:
            launch = t
        if launch is None:
            continue
        alt = float(m.group(9))
        baro = BARO_PATTERN.search(line)
        baro_alt = float(baro.group(1)) if baro else np.nan
        baro_pad = baro_alt if baro_pad is None else baro_pad
        records.append({
            "baro_raw_rel": baro_alt - baro_pad,
            "flight_id": 0, "time": t - launch, "altitude": alt, "altitude_noisy": alt,
            "velocity_z": float(m.group(10)), "accel_g": accel_g,
            "tilt_angle": -abs(float(m.group(3))),
            "gyro_roll_noisy": np.deg2rad(float(m.group(7))),
        })
    frame = ap.make_real_features(pd.DataFrame(records))
    return ap.resample_causal(
        frame, columns=tuple(dict.fromkeys([*ap.FEATURE_COLUMNS, "altitude_rel", "baro_raw_rel"]))
    )


def real_flight(path: Path, config: phy.CoastConfig) -> tuple[dict, pd.DataFrame]:
    log = load_real_log(path)
    t = log["time"].to_numpy(dtype=float)
    h = log["altitude_rel"].to_numpy(dtype=float)
    v = log["velocity_z"].to_numpy(dtype=float)
    a = log["acceleration_inertial"].to_numpy(dtype=float)
    # Real gravity for the logger's g->m/s^2 constant; sim tuning otherwise unchanged.
    cfg = phy.CoastConfig(**{**config.__dict__, "gravity": ap.LOG_G})
    ref = phy.apogee_reference(t, h, v, a, config=cfg)
    # The legacy logger fed the UKF gravity-free g where it expects specific
    # force in m/s^2, so its logged velocity is biased.  The kinematic smoother
    # rebuilds h/v from the raw barometer and the measured acceleration.
    smooth = phy.kinematic_smoother(log, baro_col="baro_raw_rel")
    h_s = smooth["h_smooth"].to_numpy(dtype=float)
    v_s = smooth["v_smooth"].to_numpy(dtype=float)
    ref_smooth = phy.apogee_reference(t, np.where(np.isfinite(h_s), h_s, h), v_s, a, config=cfg)
    trace = phy.online_ballistic_apogee(log, config=cfg)
    trace["apogee_ballistic_smoothed"] = phy.online_ballistic_apogee(
        log.assign(altitude_rel=h_s, velocity_z=v_s), config=cfg)["apogee_ballistic"].to_numpy()
    trace["altitude_rel"] = h
    trace["velocity_z"] = v
    trace["v_smooth"] = v_s
    trace["energy_height"] = h + v**2 / (2 * ap.LOG_G)
    summary = {
        "log": _rel(path),
        "apogee_observed": ref.apogee_observed,
        "observed_max_m": ref.observed_max,
        "note": "observed_max is a lower bound when apogee_observed is false",
        "extrapolated_kf_velocity_m": ref.extrapolated,
        "extrapolated_smoothed_state_m": ref_smooth.extrapolated,
        "final_kf_velocity_mps": ref.final_velocity,
        "final_smoothed_velocity_mps": ref_smooth.final_velocity,
        "k_fit_per_m": ref.k_fit,
        "coast_samples": ref.coast_samples,
        "physics_final_m": float(trace["apogee_ballistic"].dropna().iloc[-1])
        if trace["apogee_ballistic"].notna().any() else None,
        "physics_smoothed_final_m": float(trace["apogee_ballistic_smoothed"].dropna().iloc[-1])
        if trace["apogee_ballistic_smoothed"].notna().any() else None,
        "legacy_ukf_note": "112 was logged by legacy firmware that fed gravity-free g to a UKF "
                           "expecting specific force in m/s^2; its logged velocity is biased",
    }
    return summary, trace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=ROOT / "data/simulated/all_trajectories.csv.gz")
    parser.add_argument("--split", type=Path, default=ROOT / "data/experiments/apogee-v2/split_manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "data/simulated/best_gru_model.pth")
    parser.add_argument("--scaler", type=Path, default=ROOT / "data/simulated/input_scaler.pkl")
    parser.add_argument("--real-log", type=Path, default=ROOT / "data/112.txt")
    parser.add_argument("--state-source", choices=("simulation_clean", "estimated", "m7_ukf"),
                        default="simulation_clean")
    parser.add_argument("--ukf-replay", type=Path, default=None,
                        help="flight-computer ukf_replay executable (for --state-source m7_ukf)")
    parser.add_argument("--tag", default=None, help="output subdirectory (default: state source)")
    parser.add_argument("--no-gru", action="store_true", help="physics baseline only (no torch)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/experiments/physics-baseline")
    args = parser.parse_args(argv)

    config = phy.CoastConfig()
    raw = load_sim_raw(args.data)
    sampled = ap.resample_causal(raw)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    ap.validate_split_manifest(split, sampled["flight_id"].unique())
    split_of = {int(f): name for name, ids in split.items() for f in ids}

    if args.state_source == "m7_ukf":
        if args.ukf_replay is None or not args.ukf_replay.exists():
            parser.error("--state-source m7_ukf needs --ukf-replay pointing at the built ukf_replay")
        grid = m7_ukf_grid(raw, args.ukf_replay)
    else:
        grid = apply_state_source(sampled, args.state_source)
    endpoints = ap.make_endpoint_manifest(grid).reset_index(drop=True)
    endpoints = endpoints[endpoints["window_valid"]].reset_index(drop=True)
    physics = phy.online_ballistic_apogee(grid, config=config)
    grid = grid.join(physics[["coast", "k_est", "apogee_ballistic"]])
    # Same physics on a barometer+accelerometer least-squares state instead of
    # the source's h/v (matters when that velocity is noisy, e.g. the M7 UKF).
    smooth = phy.kinematic_smoother(grid)
    smooth_grid = grid.assign(altitude_rel=smooth["h_smooth"], velocity_z=smooth["v_smooth"])
    grid["apogee_ballistic_smoothed"] = phy.online_ballistic_apogee(
        smooth_grid, config=config)["apogee_ballistic"]
    pick = grid.set_index(["flight_id", "time"])
    keyed = list(zip(endpoints["flight_id"], endpoints["endpoint_time"]))
    endpoints["physics_m"] = pick.loc[keyed, "apogee_ballistic"].to_numpy()
    endpoints["physics_smoothed_m"] = pick.loc[keyed, "apogee_ballistic_smoothed"].to_numpy()
    endpoints["k_est"] = pick.loc[keyed, "k_est"].to_numpy()
    endpoints["energy_height_m"] = (
        pick.loc[keyed, "altitude_rel"].to_numpy() + pick.loc[keyed, "velocity_z"].to_numpy() ** 2 / (2 * ap.SIM_G)
    )
    apogee_time = sampled.loc[sampled.groupby("flight_id")["altitude_rel"].idxmax(), ["flight_id", "time"]]
    endpoints["time_to_apogee_s"] = endpoints["flight_id"].map(apogee_time.set_index("flight_id")["time"]) - endpoints["endpoint_time"]
    endpoints["band"] = pd.cut(endpoints["time_to_apogee_s"], BANDS, include_lowest=True)
    endpoints["split"] = endpoints["flight_id"].map(split_of)

    columns = {"physics": "physics_m", "physics_smoothed": "physics_smoothed_m",
               "energy_height": "energy_height_m"}
    if not args.no_gru:
        endpoints["gru_m"] = gru_predictions(grid, endpoints, args.checkpoint, args.scaler)
        columns["gru"] = "gru_m"

    matched = endpoints.dropna(subset=list(columns.values()))
    summary = summarize(matched, columns)
    coverage = (endpoints.assign(physics_defined=endpoints["physics_m"].notna())
                .groupby("split")["physics_defined"].mean().rename("physics_coverage").reset_index())

    real_summary, real_trace = real_flight(args.real_log, config)

    out = args.output_dir / (args.tag or args.state_source)
    out.mkdir(parents=True, exist_ok=True)
    endpoints.drop(columns=["band"]).to_csv(out / "predictions.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    coverage.to_csv(out / "coverage.csv", index=False)
    real_trace.to_csv(args.output_dir / "112_physics_trace.csv", index=False)
    ap.write_json(args.output_dir / "112_reference.json", real_summary)
    ap.write_json(out / "run_config.json", {
        "coast_config": config.__dict__, "state_source": args.state_source,
        "data_sha256": ap.sha256_file(args.data), "split": _rel(args.split),
        "checkpoint": None if args.no_gru else _rel(args.checkpoint),
        "checkpoint_sha256": None if args.no_gru else ap.sha256_file(args.checkpoint),
        "matched_rule": "endpoints where every compared model is defined (coast only)",
    })

    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summary[summary["split"] == "test"].round(3).to_string(index=False))
        print(coverage.round(3).to_string(index=False))
    print(json.dumps(real_summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
