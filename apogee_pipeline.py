"""Shared, deterministic preprocessing contracts for apogee experiments.

The module deliberately has no PyTorch dependency.  It defines the feature
order, acceleration convention, AGL labels, causal resampling, common
endpoints, and split manifests used by notebooks and CLI tools.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SCHEMA_VERSION = 1
SIM_G = 9.81
LOG_G = 9.80665

FEATURE_COLUMNS = (
    "altitude_noisy_rel",
    "velocity_z",
    "acceleration_inertial",
    "tilt_angle",
    "gyro_roll_noisy",
    "dynamic_pressure",
)
LEGACY_FEATURE_COLUMNS = (
    "altitude_noisy_rel",
    "velocity_z",
    "acceleration_noisy",
    "tilt_angle",
    "gyro_roll_noisy",
    "dynamic_pressure",
)
FEATURE_UNITS = {
    "altitude_noisy_rel": "m AGL",
    "velocity_z": "m/s",
    "acceleration_inertial": "m/s^2 (gravity removed)",
    "tilt_angle": "deg",
    "gyro_roll_noisy": "rad/s",
    "dynamic_pressure": "Pa (AGL ISA proxy)",
}
ACCELERATION_CONVENTION = {
    "sim": "acceleration_inertial = acceleration_noisy - 9.81",
    "real": "acceleration_inertial = logged_accel_g * 9.80665",
    "body_axis_mapping": "unverified; only gravity removal is confirmed",
}
ALTITUDE_DATUM = "AGL; zero is the first parsed observation at/after launch"
STATE_SOURCE = "simulation_clean"
MANIFEST_REQUIRED_KEYS = (
    "schema_version",
    "commit",
    "data_sha256",
    "scaler_sha256",
    "model_sha256",
    "paths",
    "artifact_source",
    "features",
    "feature_units",
    "artifact_feature_columns",
    "artifact_acceleration_encoding",
    "acceleration_convention",
    "altitude_datum",
    "state_source",
    "dt",
    "window_size",
    "stride",
    "seed",
    "environment",
    "split",
)


class PipelineError(ValueError):
    """Raised when a data contract is violated."""


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise PipelineError(f"missing columns: {missing}")


def validate_time_order(
    frame: pd.DataFrame,
    *,
    flight_col: str = "flight_id",
    time_col: str = "time",
) -> None:
    """Reject duplicate or backward timestamps within a flight."""
    _require_columns(frame, (flight_col, time_col))
    for flight_id, group in frame.groupby(flight_col, sort=False):
        times = group[time_col].to_numpy(dtype=float)
        if not np.isfinite(times).all():
            raise PipelineError(f"non-finite time in flight {flight_id}")
        if len(times) > 1 and np.any(np.diff(times) <= 0):
            raise PipelineError(
                f"time must be strictly increasing within flight {flight_id}"
            )


def _pad_altitude(
    frame: pd.DataFrame,
    *,
    flight_col: str,
    altitude_col: str,
) -> pd.Series:
    _require_columns(frame, (flight_col, altitude_col))
    return frame.groupby(flight_col, sort=False)[altitude_col].transform("first")


def _relative_dynamic_pressure(altitude_rel: pd.Series, velocity_z: pd.Series) -> pd.Series:
    # This is a launch-referenced ISA proxy, not an absolute-density product.
    rho = 1.225 * np.exp(-0.00012 * altitude_rel.abs())
    return 0.5 * rho * velocity_z**2


def make_sim_features(
    frame: pd.DataFrame,
    *,
    flight_col: str = "flight_id",
    time_col: str = "time",
    altitude_col: str = "altitude",
    altitude_noisy_col: str = "altitude_noisy",
    velocity_col: str = "velocity_z",
    acceleration_noisy_col: str = "acceleration_noisy",
) -> pd.DataFrame:
    """Create AGL/sim features without mutating the input frame.

    The original ``acceleration_noisy`` is specific force (g + acceleration);
    the new ``acceleration_inertial`` subtracts the simulator's 9.81 m/s^2.
    """
    _require_columns(
        frame,
        (flight_col, time_col, altitude_col, altitude_noisy_col, velocity_col, acceleration_noisy_col),
    )
    validate_time_order(frame, flight_col=flight_col, time_col=time_col)
    out = frame.copy()
    pad = _pad_altitude(out, flight_col=flight_col, altitude_col=altitude_col)
    out["altitude_rel"] = out[altitude_col] - pad
    out["altitude_noisy_rel"] = out[altitude_noisy_col] - pad
    out["acceleration_inertial"] = out[acceleration_noisy_col] - SIM_G
    out["dynamic_pressure"] = _relative_dynamic_pressure(
        out["altitude_rel"], out[velocity_col]
    )
    return out


def make_real_features(
    frame: pd.DataFrame,
    *,
    flight_col: str = "flight_id",
    time_col: str = "time",
    altitude_col: str = "altitude",
    altitude_noisy_col: str = "altitude_noisy",
    velocity_col: str = "velocity_z",
    accel_g_col: str = "accel_g",
) -> pd.DataFrame:
    """Create AGL/sim features for real logs.

    ``accel_g_col`` is the logged, gravity-removed acceleration in g.  It is
    converted once with the logger constant 9.80665; it is never subtracted
    from 9.81 again.
    """
    _require_columns(
        frame,
        (flight_col, time_col, altitude_col, altitude_noisy_col, velocity_col, accel_g_col),
    )
    validate_time_order(frame, flight_col=flight_col, time_col=time_col)
    out = frame.copy()
    pad = _pad_altitude(out, flight_col=flight_col, altitude_col=altitude_col)
    out["altitude_rel"] = out[altitude_col] - pad
    out["altitude_noisy_rel"] = out[altitude_noisy_col] - pad
    out["acceleration_inertial"] = out[accel_g_col] * LOG_G
    out["dynamic_pressure"] = _relative_dynamic_pressure(
        out["altitude_rel"], out[velocity_col]
    )
    return out


def add_relative_labels(
    frame: pd.DataFrame,
    *,
    flight_col: str = "flight_id",
    altitude_rel_col: str = "altitude_rel",
    velocity_col: str = "velocity_z",
    gravity: float = SIM_G,
) -> pd.DataFrame:
    """Add AGL target columns; rows with a degenerate denominator stay NaN."""
    _require_columns(frame, (flight_col, altitude_rel_col, velocity_col))
    out = frame.copy()
    out["h_theoretical_gain"] = out[velocity_col] ** 2 / (2 * gravity)
    out["h_theoretical_rel"] = out[altitude_rel_col] + out["h_theoretical_gain"]
    out["apogee_rel"] = out.groupby(flight_col, sort=False)[altitude_rel_col].transform("max")
    denom = out["h_theoretical_rel"].to_numpy(dtype=float)
    out["energy_ratio_rel"] = np.where(
        denom > 1e-9,
        out["apogee_rel"].to_numpy(dtype=float) / np.maximum(denom, 1e-9),
        np.nan,
    )
    return out


def validate_features(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
    allow_nan: bool = False,
) -> None:
    _require_columns(frame, feature_columns)
    values = frame[list(feature_columns)].to_numpy(dtype=float)
    if allow_nan:
        if np.isinf(values).any():
            raise PipelineError("inf feature value")
    else:
        if not np.isfinite(values).all():
            raise PipelineError("non-finite feature value")


def resample_causal(
    frame: pd.DataFrame,
    *,
    dt: float = 0.02,
    max_age_s: float = 0.04,
    flight_col: str = "flight_id",
    time_col: str = "time",
    columns: Sequence[str] = ("altitude_rel", "altitude_noisy_rel", "velocity_z",
                              "acceleration_inertial", "tilt_angle",
                              "gyro_roll_noisy", "dynamic_pressure",
                              "energy_ratio_rel", "apogee_rel"),
) -> pd.DataFrame:
    """Backward as-of resampling onto a causal uniform grid."""
    if dt <= 0 or max_age_s < 0:
        raise PipelineError("dt must be positive and max_age_s non-negative")
    _require_columns(frame, (flight_col, time_col, *columns))
    validate_time_order(frame, flight_col=flight_col, time_col=time_col)

    rows: list[pd.DataFrame] = []
    for flight_id, group in frame.groupby(flight_col, sort=False):
        group = group.sort_values(time_col, kind="mergesort").reset_index(drop=True)
        obs_t = group[time_col].to_numpy(dtype=float)
        end = float(obs_t[-1])
        grid = np.arange(0.0, end + dt * 1e-6, dt)
        idx = np.searchsorted(obs_t, grid, side="right") - 1
        valid = idx >= 0
        age = np.full(len(grid), np.inf)
        safe_idx = np.maximum(idx, 0)
        age[valid] = grid[valid] - obs_t[safe_idx[valid]]
        valid &= age <= max_age_s + 1e-12
        selected = group.iloc[safe_idx].reset_index(drop=True)
        out = pd.DataFrame({
            flight_col: flight_id,
            time_col: grid,
            "observation_time": obs_t[safe_idx],
            "age_s": age,
            "valid": valid,
        })
        for name in columns:
            out[name] = selected[name].to_numpy()
        rows.append(out)

    if not rows:
        return pd.DataFrame(columns=(flight_col, time_col, "observation_time", "age_s", "valid", *columns))
    return pd.concat(rows, ignore_index=True)


def make_endpoint_manifest(
    frame: pd.DataFrame,
    *,
    max_window: int = 100,
    stride: int = 10,
    flight_col: str = "flight_id",
    time_col: str = "time",
    h_col: str = "altitude_rel",
    v_col: str = "velocity_z",
    target_col: str = "energy_ratio_rel",
    apogee_col: str = "apogee_rel",
) -> pd.DataFrame:
    """Create common endpoints shared by every history length."""
    if max_window < 1 or stride < 1:
        raise PipelineError("max_window and stride must be positive")
    _require_columns(frame, (flight_col, time_col, "valid", h_col, v_col, target_col, apogee_col))
    rows = []
    for flight_id, group in frame.groupby(flight_col, sort=False):
        group = group.sort_values(time_col, kind="mergesort").reset_index(drop=True)
        valid = group["valid"].to_numpy(dtype=bool)
        for index in range(max_window - 1, len(group), stride):
            if not valid[index]:
                continue
            start = index - max_window + 1
            rows.append({
                flight_col: flight_id,
                "endpoint_index": index,
                "endpoint_time": float(group.iloc[index][time_col]),
                "h": float(group.iloc[index][h_col]),
                "v": float(group.iloc[index][v_col]),
                "target": float(group.iloc[index][target_col]),
                "apogee_rel": float(group.iloc[index][apogee_col]),
                "window_valid": bool(valid[start:index + 1].all()),
            })
    manifest = pd.DataFrame(rows)
    if not manifest.empty:
        manifest["denom"] = manifest["h"] + manifest["v"] ** 2 / (2 * SIM_G)
    return manifest


def slice_history(
    values: np.ndarray,
    endpoints: Sequence[int],
    window_size: int,
) -> np.ndarray:
    """Slice causal histories ending at each endpoint."""
    values = np.asarray(values)
    endpoints = np.asarray(endpoints, dtype=int)
    if window_size < 1:
        raise PipelineError("window_size must be positive")
    if values.ndim not in (2,):
        raise PipelineError("values must be 2D for a single flight")
    if endpoints.size == 0:
        return np.empty((0, window_size, values.shape[1]), dtype=values.dtype)
    if endpoints.min() < window_size - 1 or endpoints.max() >= len(values):
        raise PipelineError("endpoint outside the available causal history")
    return np.stack([values[e - window_size + 1:e + 1] for e in endpoints], axis=0)


def build_split_manifest(
    flight_ids: Sequence[int],
    *,
    seed: int = 42,
) -> dict[str, list[int]]:
    ids = np.sort(np.asarray(flight_ids, dtype=int))
    if len(ids) == 0 or len(np.unique(ids)) != len(ids):
        raise PipelineError("flight_ids must be non-empty and unique")
    train, temp = train_test_split(ids, test_size=0.4, random_state=seed)
    val, rest = train_test_split(temp, test_size=0.625, random_state=seed)
    calibration, test = train_test_split(rest, test_size=0.5, random_state=seed)
    manifest = {
        "train": train.tolist(),
        "validation": val.tolist(),
        "calibration": calibration.tolist(),
        "test": test.tolist(),
    }
    validate_split_manifest(manifest, ids)
    return manifest


def validate_split_manifest(manifest: Mapping[str, Sequence[int]], all_ids: Sequence[int]) -> None:
    expected = {"train", "validation", "calibration", "test"}
    if set(manifest) != expected:
        raise PipelineError(f"split keys must be {sorted(expected)}")
    sets = {k: set(int(x) for x in v) for k, v in manifest.items()}
    for name, values in sets.items():
        if len(values) != len(manifest[name]):
            raise PipelineError(f"duplicate ids in {name}")
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if sets[a] & sets[b]:
                raise PipelineError(f"split overlap: {a}/{b}")
    union = set().union(*sets.values())
    if union != set(int(x) for x in all_ids):
        raise PipelineError("split union does not equal all flight ids")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_baseline_manifest(
    *,
    commit: str,
    data_path: str | Path,
    scaler_path: str | Path,
    model_path: str | Path,
    split: Mapping[str, Sequence[int]],
    dt: float,
    window_size: int,
    stride: int,
    seed: int,
    acceleration_convention: Mapping[str, str] | None = None,
    altitude_datum: str = ALTITUDE_DATUM,
    state_source: str = STATE_SOURCE,
    artifact_feature_columns: Sequence[str] | None = None,
    artifact_acceleration_encoding: str = "inertial",
    last_model_path: str | Path | None = None,
    artifact_source: str = "working_tree",
) -> dict:
    last_model_path = Path(last_model_path) if last_model_path else None
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "commit": commit,
        "data_sha256": sha256_file(data_path),
        "scaler_sha256": sha256_file(scaler_path),
        "model_sha256": sha256_file(model_path),
        "last_model_sha256": sha256_file(last_model_path) if last_model_path else None,
        "paths": {
            "data": Path(data_path).as_posix(),
            "scaler": Path(scaler_path).as_posix(),
            "model": Path(model_path).as_posix(),
            "last_model": Path(last_model_path).as_posix() if last_model_path else None,
        },
        "artifact_source": artifact_source,
        "features": list(FEATURE_COLUMNS),
        "feature_units": dict(FEATURE_UNITS),
        "artifact_feature_columns": list(artifact_feature_columns or FEATURE_COLUMNS),
        "artifact_acceleration_encoding": artifact_acceleration_encoding,
        "acceleration_convention": dict(acceleration_convention or ACCELERATION_CONVENTION),
        "altitude_datum": altitude_datum,
        "state_source": state_source,
        "dt": dt,
        "window_size": window_size,
        "stride": stride,
        "seed": seed,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": importlib.metadata.version("numpy"),
            "pandas": importlib.metadata.version("pandas"),
            "scikit-learn": importlib.metadata.version("scikit-learn"),
        },
        "split": {k: list(v) for k, v in split.items()},
    }
    validate_baseline_manifest(manifest)
    return manifest


def validate_baseline_manifest(manifest: Mapping) -> None:
    missing = [key for key in MANIFEST_REQUIRED_KEYS if key not in manifest]
    if missing:
        raise PipelineError(f"manifest missing keys: {missing}")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise PipelineError("unsupported manifest schema_version")
    if manifest["features"] != list(FEATURE_COLUMNS):
        raise PipelineError("manifest features do not match FEATURE_COLUMNS")
    if list(manifest["artifact_feature_columns"]) not in (
        list(FEATURE_COLUMNS), list(LEGACY_FEATURE_COLUMNS)
    ):
        raise PipelineError("artifact feature order is not a supported contract")
    if manifest["artifact_acceleration_encoding"] not in ("inertial", "legacy_specific_force"):
        raise PipelineError("unsupported artifact acceleration encoding")
    if manifest["state_source"] not in (STATE_SOURCE, "estimated"):
        raise PipelineError("unsupported state_source")
    if manifest["dt"] <= 0 or manifest["window_size"] < 1 or manifest["stride"] < 1:
        raise PipelineError("invalid window/dt/stride")


def write_json(path: str | Path, payload: Mapping) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Stable key order and explicit UTF-8 for experiment review.
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def adapt_legacy_scaler_to_inertial(scaler, *, feature_name: str = "acceleration_inertial"):
    """Return a scaler adapted from the legacy specific-force encoding.

    The legacy StandardScaler was fit on ``acceleration_noisy = a_inertial +
    9.81``. Shifting its mean by -9.81 yields exactly the scaler that would
    have been fit on the inertial channel, so existing weights can be reused
    for an input-encoding A/B.
    """
    import copy

    if getattr(scaler, "_psintelligence_accel_encoding", None) == "inertial":
        raise PipelineError("scaler is already inertial; refusing double adaptation")
    adapted = copy.deepcopy(scaler)
    names = getattr(scaler, "feature_names_in_", None)
    if names is not None:
        names = np.asarray(names)
        candidates = [feature_name, "acceleration_noisy"]
        matches = [i for i, name in enumerate(names) if name in candidates]
        if len(matches) != 1:
            raise PipelineError(
                f"scaler must have exactly one acceleration feature, found {matches}"
            )
        index = int(matches[0])
    else:
        # Known order: altitude, velocity, acceleration, tilt, gyro, q.
        index = 2
    adapted.mean_ = np.asarray(adapted.mean_, dtype=float).copy()
    adapted.mean_[index] -= SIM_G
    adapted._psintelligence_accel_encoding = "inertial"
    return adapted


def causal_state_estimate(
    frame: pd.DataFrame,
    *,
    flight_col: str = "flight_id",
    time_col: str = "time",
    altitude_rel_col: str = "altitude_rel",
    velocity_col: str = "velocity_z",
    accel_col: str = "acceleration_inertial",
) -> pd.DataFrame:
    """Simple causal double integration for an observable-state proxy.

    This is not the flight firmware KF.  It uses only the current and prior
    samples and is exposed so clean-state and estimated-state experiments are
    not confused.
    """
    _require_columns(frame, (flight_col, time_col, altitude_rel_col, velocity_col, accel_col))
    validate_time_order(frame, flight_col=flight_col, time_col=time_col)
    rows = []
    for flight_id, group in frame.groupby(flight_col, sort=False):
        g = group.sort_values(time_col, kind="mergesort").reset_index(drop=True)
        t = g[time_col].to_numpy(dtype=float)
        a = g[accel_col].to_numpy(dtype=float)
        h = np.empty(len(g), dtype=float)
        v = np.empty(len(g), dtype=float)
        h[0] = float(g.iloc[0][altitude_rel_col])
        v[0] = float(g.iloc[0][velocity_col])
        for i in range(1, len(g)):
            dt = t[i] - t[i - 1]
            if not np.isfinite(dt) or dt <= 0:
                raise PipelineError("non-positive dt in state estimate")
            # semi-implicit Euler; uses only observations available at i
            v[i] = v[i - 1] + a[i] * dt
            h[i] = h[i - 1] + v[i] * dt
        rows.append(pd.DataFrame({
            flight_col: flight_id,
            time_col: t,
            "h_est": h,
            "v_est": v,
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def resolve_acceleration_encoding(metadata: Mapping | None) -> str:
    """Resolve artifact encoding, defaulting to the known legacy baseline."""
    if not metadata:
        return "legacy_specific_force"
    encoding = metadata.get("acceleration_encoding") or metadata.get("artifact_acceleration_encoding")
    if encoding in ("inertial", "legacy_specific_force"):
        return encoding
    # No explicit marker: current repository baseline is legacy.  Do not infer
    # from feature names alone, because metadata and scaler can be out of sync.
    return "legacy_specific_force"


def prepare_scaler_for_encoding(scaler, encoding: str):
    if encoding == "inertial":
        return scaler
    if encoding == "legacy_specific_force":
        return adapt_legacy_scaler_to_inertial(scaler)
    raise PipelineError(f"unsupported acceleration encoding: {encoding}")


def load_scaler_for_contract(
    scaler_path: str | Path,
    metadata_path: str | Path | None = None,
) -> tuple[object, str]:
    import joblib

    scaler = joblib.load(scaler_path)
    metadata = joblib.load(metadata_path) if metadata_path and Path(metadata_path).exists() else None
    encoding = resolve_acceleration_encoding(metadata)
    return prepare_scaler_for_encoding(scaler, encoding), encoding


def validate_artifact_hashes(manifest: Mapping, root: str | Path = ".") -> None:
    """Verify the files named in a baseline manifest match its hashes."""
    root = Path(root)
    paths = manifest.get("paths", {})
    checks = (
        ("data", "data_sha256"),
        ("scaler", "scaler_sha256"),
        ("model", "model_sha256"),
        ("last_model", "last_model_sha256"),
    )
    for path_key, hash_key in checks:
        relative = paths.get(path_key)
        expected = manifest.get(hash_key)
        if relative is None or expected is None:
            continue
        path = Path(relative)
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            raise PipelineError(f"manifest artifact missing: {path}")
        if sha256_file(path) != expected:
            raise PipelineError(f"manifest hash mismatch: {path}")
