"""Coasting-only r = H/(h + v_z²/(2g)) contracts, without torch.

The hard head assumes no thrust and nonpositive vertical nongravitational
work during ascent. It is not a bound for arbitrary wind/lift/estimator errors.
All h and H are launch-relative (AGL); the original energy_ratio is untouched.
"""
from __future__ import annotations

from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

import apogee_pipeline as ap

FEATURE_COLUMNS = ap.FEATURE_COLUMNS
TARGET_COLUMN = "r_target"
MIN_ASCENT_SPEED = 3.0
EXPERIMENT_DIR = Path("data/experiments/step3-coasting")


def _require(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [name for name in columns if name not in frame]
    if missing:
        raise ap.PipelineError(f"missing columns: {missing}")


def _gravity(gravity: float) -> None:
    if not np.isscalar(gravity) or not np.isfinite(gravity) or gravity <= 0:
        raise ap.PipelineError("gravity must be finite and positive")


def _domain(h, v, gravity: float):
    _gravity(gravity)
    h, v = np.broadcast_arrays(np.asarray(h, dtype=float), np.asarray(v, dtype=float))
    with np.errstate(over="ignore", invalid="ignore"):
        denom = h + v**2 / (2 * gravity)
    if not (np.isfinite(h).all() and np.isfinite(v).all()
            and np.isfinite(denom).all() and (h >= 0).all()
            and (v > MIN_ASCENT_SPEED).all() and (denom > 0).all()):
        raise ap.PipelineError("head requires finite h >= 0, v_z > 3 and finite D > 0")
    return h, v, denom


def constrained_r_head(logit, h, v, *, gravity: float = ap.SIM_G):
    """NumPy reference: state-dependent r floor, not a posthoc projection.

    Returns (r, H). Finite logits may saturate at a boundary in floating point.
    The sigmoid parameter is not a replacement training target.
    """
    h, _, denom = _domain(h, v, gravity)
    logit, h, denom = np.broadcast_arrays(np.asarray(logit, dtype=float), h, denom)
    if not np.isfinite(logit).all():
        raise ap.PipelineError("logit must be finite")
    # exp(-abs(logit)) never overflows, including extreme finite logits.
    small = np.exp(-np.abs(logit))
    sigmoid = np.where(logit >= 0, 1.0 / (1.0 + small), small / (1.0 + small))
    floor = h / denom
    ratio = floor + (1.0 - floor) * sigmoid
    return ratio, denom * ratio


def reconstruct_apogee(r, h, v, *, gravity: float = ap.SIM_G):
    """Invert the unchanged ratio label; do not clip inconsistent labels."""
    _, _, denom = _domain(h, v, gravity)
    r, denom = np.broadcast_arrays(np.asarray(r, dtype=float), denom)
    if not np.isfinite(r).all():
        raise ap.PipelineError("ratio must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        height = denom * r
    if not np.isfinite(height).all():
        raise ap.PipelineError("reconstructed apogee must be finite")
    return height


def make_coasting_frame(frame: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Shared AGL/inertial features, metadata labels, and auditable row masks.

    candidate is input-valid AND label-finite, independent of target bounds.
    strict_consistent additionally enforces exact h <= H <= D, with NO epsilon
    or clipping. Use strict only for training/validation and secondary test.
    Missing/nonfinite labels are separate from physical counterexamples.
    """
    _require(frame, ("flight_id", "flight_phase", "velocity_z",
                     "tilt_angle", "gyro_roll_noisy"))
    _require(metadata, ("flight_id", "apogee_altitude"))
    if frame["flight_id"].isna().any() or metadata["flight_id"].isna().any():
        raise ap.PipelineError("missing flight_id")
    if metadata["flight_id"].duplicated().any():
        raise ap.PipelineError("duplicate flight_id in metadata")
    out = ap.make_sim_features(frame)
    # Same pad as make_sim_features, not the sampled maximum or noisy altitude.
    pad = frame.groupby("flight_id", sort=False)["altitude"].transform("first")
    first = frame.groupby("flight_id", sort=False).head(1).set_index("flight_id")["altitude"]
    first_pad = frame["flight_id"].map(first)
    missing_pad = first_pad.isna().to_numpy()
    nonfinite_pad = (~np.isfinite(first_pad.to_numpy(dtype=float))) & ~missing_pad
    lookup = metadata.set_index("flight_id")["apogee_altitude"]
    missing_metadata = ~out["flight_id"].isin(lookup.index)
    apogee_asl = out["flight_id"].map(lookup)
    out["pad_altitude_asl"] = pad
    out["apogee_agl"] = apogee_asl - pad
    h = out["altitude_rel"].to_numpy(dtype=float)
    v = out["velocity_z"].to_numpy(dtype=float)
    target = out["apogee_agl"].to_numpy(dtype=float)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        gain = v**2 / (2 * ap.SIM_G)
        denom = h + gain
        ratio = np.divide(target, denom, out=np.full(len(out), np.nan),
                          where=np.isfinite(denom) & (denom > 0))
    out["gain"], out["denom"], out[TARGET_COLUMN] = gain, denom, ratio
    state = out[["altitude_rel", "velocity_z"]].to_numpy(dtype=float)
    features = out[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    # Primary input reasons are exclusive. Label statuses are separate, so
    # an input rejection never conceals a missing or inconsistent label.
    reasons = np.select([
        out["flight_phase"].isna().to_numpy(),
        frame[["altitude", "velocity_z"]].isna().any(axis=1).to_numpy() | missing_pad,
        (~np.isfinite(state)).any(axis=1) | nonfinite_pad,
        out["flight_phase"].ne("coasting").to_numpy(),
        v <= MIN_ASCENT_SPEED,
        h < 0,
        ~np.isfinite(denom) | (denom <= 0),
        np.isnan(features).any(axis=1),
        (~np.isfinite(features)).any(axis=1),
    ], ["missing_phase", "missing_state", "nonfinite_state", "not_coasting",
        "not_ascending", "negative_altitude", "invalid_denominator",
        "missing_feature", "nonfinite_feature"], default="accepted")
    statuses = np.select([
        missing_metadata.to_numpy(),
        apogee_asl.isna().to_numpy() | missing_pad,
        ~np.isfinite(target),
        ~np.isfinite(ratio),
        target < h,
        target > denom,
    ], ["missing_metadata", "missing_label", "nonfinite_label", "invalid_ratio",
        "below_current", "above_ballistic"], default="consistent")
    out["input_rejection"] = reasons
    out["label_status"] = statuses
    out["input_valid"] = reasons == "accepted"
    out["candidate"] = out["input_valid"] & np.isfinite(target) & np.isfinite(ratio)
    out["strict_consistent"] = out["candidate"] & (target >= h) & (target <= denom)
    return out


def partition_coasting(
    frame: pd.DataFrame, split: Mapping[str, Sequence[int]] | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Shared flight split; NEVER filter the primary test by the true bound."""
    _require(frame, ("flight_id", "input_rejection", "label_status", "input_valid",
                     "candidate", "strict_consistent"))
    ids = frame["flight_id"].unique()
    if split is None:
        split = ap.build_split_manifest(ids)
    ap.validate_split_manifest(split, ids)
    parts, audits = {}, []
    for name, flight_ids in split.items():
        group = frame[frame["flight_id"].isin(flight_ids)]
        candidate = group["candidate"]
        strict = group["strict_consistent"]
        selected = strict if name in ("train", "validation") else candidate
        parts[name] = group[selected].copy().reset_index(drop=True)
        row = {"split": name, "flights": len(flight_ids), "raw_rows": len(group),
               "input_valid_rows": int(group["input_valid"].sum()),
               "candidate_rows": int(candidate.sum()), "strict_rows": int(strict.sum()),
               "selected_rows": int(selected.sum()),
               "excluded_inconsistent": int((candidate & ~strict).sum()) if name in ("train", "validation") else 0,
               "unscorable_input_valid": int((group["input_valid"] & ~candidate).sum())}
        for reason in ("missing_phase", "missing_state", "nonfinite_state", "not_coasting",
                       "not_ascending", "negative_altitude", "invalid_denominator",
                       "missing_feature", "nonfinite_feature"):
            row[reason] = int(group["input_rejection"].eq(reason).sum())
        for status in ("missing_metadata", "missing_label", "nonfinite_label", "invalid_ratio"):
            row[status] = int(group["label_status"].eq(status).sum())
        for status in ("below_current", "above_ballistic"):
            row[status] = int((candidate & group["label_status"].eq(status)).sum())
        row["r_gt_one"] = int((candidate & (group[TARGET_COLUMN] > 1)).sum())
        audits.append(row)
    parts["test_strict"] = parts["test"][parts["test"]["strict_consistent"]].copy().reset_index(drop=True)
    return parts, pd.DataFrame(audits)


def score_predictions(frame: pd.DataFrame, ratio) -> dict:
    """Row- and flight-weighted errors on the exact supplied population.

    The caller must label full test vs strict subset. Empty subsets report
    None, never a fabricated zero. Violation counts use floating-point tolerance.
    """
    _require(frame, ("flight_id", "altitude_rel", "velocity_z", TARGET_COLUMN,
                     "apogee_agl", "strict_consistent"))
    ratio = np.asarray(ratio, dtype=float)
    if ratio.shape != (len(frame),) or not np.isfinite(ratio).all():
        raise ap.PipelineError("predictions must be finite, one-dimensional and match rows")
    result = {"rows": len(frame), "flights": int(frame["flight_id"].nunique()),
              "strict_rows": int(frame["strict_consistent"].sum())}
    if frame.empty:
        return {**result, **dict.fromkeys(("r_mae", "r_rmse", "H_mae_m", "H_rmse_m",
                                          "flight_H_mae_m", "r_bound_violations",
                                          "H_lower_violations", "H_upper_violations"))}
    labels = frame[[TARGET_COLUMN, "apogee_agl"]].to_numpy(dtype=float)
    if not np.isfinite(labels).all():
        raise ap.PipelineError("score requires finite labels")
    h = frame["altitude_rel"].to_numpy(dtype=float)
    v = frame["velocity_z"].to_numpy(dtype=float)
    _, _, denom = _domain(h, v, ap.SIM_G)
    height = reconstruct_apogee(ratio, h, v)
    r_error = ratio - labels[:, 0]
    h_error = height - labels[:, 1]
    by_flight = pd.DataFrame({"flight_id": frame["flight_id"].to_numpy(),
                              "absolute_error": np.abs(h_error)})
    tol = 32 * np.finfo(float).eps * np.maximum(1.0, denom)
    return {**result, "r_mae": float(np.mean(np.abs(r_error))),
            "r_rmse": float(np.sqrt(np.mean(r_error**2))),
            "H_mae_m": float(np.mean(np.abs(h_error))),
            "H_rmse_m": float(np.sqrt(np.mean(h_error**2))),
            "flight_H_mae_m": float(by_flight.groupby("flight_id").absolute_error.mean().mean()),
            "r_bound_violations": int(((ratio < h / denom - 1e-12) | (ratio > 1 + 1e-12)).sum()),
            "H_lower_violations": int((height < h - tol).sum()),
            "H_upper_violations": int((height > denom + tol).sum())}


_NUMBER = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
_REAL_PATTERN = re.compile(
    r"Time:\s*" + _NUMBER + r"s.*?Euler:\s*\(\s*" + _NUMBER + r",\s*" + _NUMBER + r",\s*" + _NUMBER
    + r"\).*?Gyro:\s*\(\s*" + _NUMBER + r",\s*" + _NUMBER + r",\s*" + _NUMBER
    + r"\).*?Accel:\s*\(\s*" + _NUMBER + r",\s*" + _NUMBER + r",\s*" + _NUMBER
    + r"\).*?kal:\s*\(alt\s*:\s*" + _NUMBER + r",\s*velo\s*:\s*" + _NUMBER
)


def load_real_coasting(path: str | Path, *, launch_time_s: float | None,
                       coast_start_s: float | None) -> tuple[pd.DataFrame | None, str]:
    """Explicit event gate; coast start is seconds after configured launch.

    No simulator burnout fallback or automatic motor-event inference. The
    gyro axis/tilt mapping remains an unverified diagnostic convention.
    The observed log maximum is descriptive, not independent ground truth.
    """
    if coast_start_s is None:
        return None, "skip: REAL_COAST_START_S is None; real burnout is not established"
    if not np.isfinite(coast_start_s) or coast_start_s < 0:
        raise ap.PipelineError("REAL_COAST_START_S must be finite and nonnegative")
    if launch_time_s is None:
        return None, "skip: REAL_LAUNCH_TIME_S is None; launch datum is not established"
    if not np.isfinite(launch_time_s):
        raise ap.PipelineError("REAL_LAUNCH_TIME_S must be finite")
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="cp949")
    records = []
    for line in text.splitlines():
        match = _REAL_PATTERN.search(line)
        if match is None:
            continue
        values = [float(x) for x in match.groups()]
        time, _, pitch, _, _, _, gyro_z, _, _, accel_g, altitude, velocity = values
        if time >= launch_time_s:
            records.append({"flight_id": 0, "time": time - launch_time_s,
                            "altitude": altitude, "altitude_noisy": altitude,
                            "velocity_z": velocity, "accel_g": accel_g,
                            "tilt_angle": -abs(pitch), "gyro_roll_noisy": np.deg2rad(gyro_z)})
    if not records:
        raise ap.PipelineError("no parsed observations at/after configured launch")
    real = ap.make_real_features(pd.DataFrame(records))
    observed_max = float(real["altitude_rel"].max())
    values = real[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        denom = real["altitude_rel"] + real["velocity_z"]**2 / (2 * ap.SIM_G)
    valid = ((real["time"] >= coast_start_s) & (real["velocity_z"] > MIN_ASCENT_SPEED)
             & (real["altitude_rel"] >= 0) & np.isfinite(denom) & (denom > 0)
             & np.isfinite(values).all(axis=1))
    real = real[valid].copy().reset_index(drop=True)
    real.attrs["observed_max_agl"] = observed_max
    if real.empty:
        return None, "skip: no valid ascending observations after configured coast start"
    return real, "configured coast-only diagnostic; no independent apogee reference"
