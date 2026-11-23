#!/usr/bin/env python
"""Benchmark common endpoints and history lengths without touching artifacts.

``--dry-run`` is NumPy/pandas-only.  Actual training imports torch lazily and
writes only under --output-dir.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from io import StringIO
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import apogee_pipeline as ap


def _read_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, compression="infer")
    frame = ap.make_sim_features(frame)
    frame = ap.add_relative_labels(frame)
    ap.validate_features(frame)
    return frame


def _load_split(path: Path | None, flight_ids: np.ndarray) -> dict:
    if path is None:
        return ap.build_split_manifest(flight_ids)
    split = json.loads(path.read_text(encoding="utf-8"))
    ap.validate_split_manifest(split, flight_ids)
    return split


def _resample_and_endpoints(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sampled = ap.resample_causal(frame)
    endpoints = ap.make_endpoint_manifest(sampled)
    return sampled, endpoints


def _prepare_state_contract(
    sampled: pd.DataFrame,
    endpoints: pd.DataFrame,
    state_source: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return NN feature frame and endpoint head/label source.

    Clean mode uses simulated states for both.  Estimated mode replaces the
    NN's altitude/velocity/dynamic-pressure features and the head h/v with the
    same causal estimate, and recomputes r_target from the true apogee.
    """
    if state_source == "simulation_clean":
        return sampled, endpoints
    if state_source != "estimated":
        raise ValueError(f"unsupported state_source: {state_source}")
    state = ap.causal_state_estimate(sampled)
    features = sampled.merge(
        state, on=["flight_id", "time"], how="left", validate="one_to_one"
    )
    if features[["h_est", "v_est"]].isna().any().any():
        raise RuntimeError("estimated state missing on the causal grid")
    features["altitude_rel"] = features["h_est"]
    features["altitude_noisy_rel"] = features["h_est"]
    features["velocity_z"] = features["v_est"]
    features["dynamic_pressure"] = 0.5 * 1.225 * np.exp(
        -0.00012 * features["altitude_rel"].abs()
    ) * features["velocity_z"] ** 2
    head_endpoints = endpoints.merge(
        state, how="left",
        left_on=["flight_id", "endpoint_time"], right_on=["flight_id", "time"],
    )
    if head_endpoints[["h_est", "v_est"]].isna().any().any():
        raise RuntimeError("estimated state missing at an endpoint")
    head_endpoints["h"] = head_endpoints["h_est"]
    head_endpoints["v"] = head_endpoints["v_est"]
    head_endpoints["target"] = head_endpoints["apogee_rel"] / (
        head_endpoints["h"] + head_endpoints["v"] ** 2 / (2 * ap.SIM_G)
    )
    return features, head_endpoints


def _grid_with_m7_state(frame: pd.DataFrame, replay: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Keep the noisy barometer channel separate from the filter's altitude.
    features = frame[["flight_id", "acceleration_inertial", "tilt_angle",
                      "gyro_roll_noisy", "apogee_rel"]].copy()
    features["time"] = replay["time_us"].to_numpy() / 1_000_000
    features["altitude_noisy_rel"] = replay["baro_alt"].to_numpy()
    features["altitude_rel"] = replay["kf_alt"].to_numpy()
    features["velocity_z"] = replay["kf_vz"].to_numpy()
    features["dynamic_pressure"] = ap._relative_dynamic_pressure(
        features["altitude_rel"], features["velocity_z"]
    )
    features["ukf_valid"] = (replay["valid"].to_numpy() == 1) & np.isfinite(
        features[[*ap.FEATURE_COLUMNS, "altitude_rel"]].to_numpy()
    ).all(axis=1)
    denom = (features["altitude_rel"] + features["velocity_z"] ** 2 / (2 * ap.SIM_G)).to_numpy()
    features["energy_ratio_rel"] = np.divide(
        features["apogee_rel"].to_numpy(), denom,
        out=np.full(len(features), np.nan),
        where=features["ukf_valid"].to_numpy() & (denom > 1e-9),
    )

    # Replay before resampling. A held grid row is not a new barometer update.
    sampled = ap.resample_causal(features, columns=(
        *ap.FEATURE_COLUMNS, "altitude_rel", "energy_ratio_rel", "apogee_rel", "ukf_valid",
    ))
    sampled["valid"] &= sampled.pop("ukf_valid")
    endpoints = ap.make_endpoint_manifest(sampled)
    if not endpoints.empty:
        endpoints = endpoints.loc[np.isfinite(endpoints["target"])].reset_index(drop=True)
    return sampled, endpoints


def _run_ukf_replay(events: pd.DataFrame, baro_alt, executable: Path) -> pd.DataFrame:
    # An empty barometer cell means no new observation at this event.
    sensor_rows = pd.DataFrame({
        "flight_id": events["flight_id"],
        "time_us": np.rint(events["time"].to_numpy() * 1_000_000).astype(np.int64),
        "accel_z_g": events["acceleration_inertial"] / ap.LOG_G,
        "baro_alt": baro_alt,
    })
    try:
        result = subprocess.run(
            [str(executable.resolve())], input=sensor_rows.to_csv(index=False, lineterminator="\n"),
            encoding="utf-8", capture_output=True, check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ukf_replay failed: {exc.stderr.strip()}") from exc
    replay = pd.read_csv(StringIO(result.stdout))
    expected = ["flight_id", "time_us", "baro_alt", "kf_alt", "kf_vz", "valid"]
    if (list(replay.columns) != expected or not np.array_equal(
            replay[expected[:2]].to_numpy(), sensor_rows[expected[:2]].to_numpy())):
        raise RuntimeError("ukf_replay output does not match the sensor rows")
    return replay


def _benchmark_inputs(frame: pd.DataFrame, args) -> tuple[pd.DataFrame, pd.DataFrame]:
    if args.state_source != "m7_ukf":
        return _prepare_state_contract(*_resample_and_endpoints(frame), args.state_source)

    ordered = frame.sort_values(["flight_id", "time"]).reset_index(drop=True)
    replay = _run_ukf_replay(ordered, ordered["altitude_noisy_rel"], args.ukf_replay)
    sampled, endpoints = _grid_with_m7_state(ordered, replay)
    if endpoints.empty or not endpoints["window_valid"].any():
        raise RuntimeError("M7 UKF replay produced no valid common windows")
    return sampled, endpoints


def _select_endpoints(frame: pd.DataFrame, endpoints: pd.DataFrame, phase: str) -> pd.DataFrame:
    if phase == "all":
        return endpoints
    # Only the prediction instant is filtered. Earlier observations, including
    # powered ascent, remain available to construct each causal history.
    phase_grid = ap.resample_causal(frame, columns=("flight_phase",))
    coast = phase_grid.loc[
        phase_grid["valid"] & phase_grid["flight_phase"].eq("coasting"), ["flight_id", "time"],
    ].rename(columns={"time": "endpoint_time"})
    return endpoints.merge(coast, on=["flight_id", "endpoint_time"],
                           how="inner", validate="one_to_one")


def _state_metadata(args) -> dict:
    estimator = {"simulation_clean": "simulation_clean",
                 "estimated": "causal_acceleration_integration",
                 "m7_ukf": "m7_variable_dt_ukf"}[args.state_source]
    metadata = {"state_source": args.state_source, "state_estimator": estimator}
    metadata["endpoint_phase"] = args.endpoint_phase
    if args.state_source == "m7_ukf":
        metadata["ukf_replay_sha256"] = ap.sha256_file(args.ukf_replay)
    return metadata


def _fit_endpoint_scaler(
    features: pd.DataFrame,
    endpoints: pd.DataFrame,
    train_ids: Sequence[int],
):
    from sklearn.preprocessing import StandardScaler

    rows = []
    for flight_id in train_ids:
        flight = features[features["flight_id"] == flight_id].reset_index(drop=True)
        ep = endpoints[
            (endpoints["flight_id"] == flight_id) & endpoints["window_valid"]
        ]
        idx = ep["endpoint_index"].to_numpy(dtype=int)
        if len(idx):
            rows.append(flight.iloc[idx][list(ap.FEATURE_COLUMNS)].to_numpy(dtype=np.float32))
    if not rows:
        raise RuntimeError("no common valid endpoints for scaler fit")
    return StandardScaler().fit(np.concatenate(rows))


def _samples_for_window(
    features: pd.DataFrame,
    endpoints: pd.DataFrame,
    flight_ids: list[int],
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs, hs, vs, ys, fids = [], [], [], [], []
    for flight_id in flight_ids:
        flight = features[features["flight_id"] == flight_id].reset_index(drop=True)
        ep = endpoints[endpoints["flight_id"] == flight_id]
        # The common 100-step validity mask is fixed before history length is
        # chosen, so L=25/50/100 always use the same endpoint set.
        ep = ep[ep["window_valid"]]
        ep = ep[(ep["endpoint_index"] >= window - 1) & (ep["endpoint_index"] < len(flight))]
        if ep.empty:
            continue
        values = flight[list(ap.FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
        xs.append(ap.slice_history(values, ep["endpoint_index"].to_numpy(), window))
        hs.append(ep["h"].to_numpy(dtype=np.float32))
        vs.append(ep["v"].to_numpy(dtype=np.float32))
        ys.append(ep["target"].to_numpy(dtype=np.float32))
        fids.append(np.full(len(ep), flight_id, dtype=int))
    if not xs:
        return (np.empty((0, window, len(ap.FEATURE_COLUMNS)), dtype=np.float32),
                np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32), np.empty(0, dtype=int))
    return (np.concatenate(xs), np.concatenate(hs), np.concatenate(vs),
            np.concatenate(ys), np.concatenate(fids))


def _flight_mae(errors: np.ndarray, fids: np.ndarray) -> float:
    values = [np.mean(np.abs(errors[fids == f])) for f in np.unique(fids)]
    return float(np.mean(values)) if values else float("nan")


def _dataset_summary(frame: pd.DataFrame, endpoints: pd.DataFrame, window: int) -> dict:
    valid = endpoints.loc[endpoints["window_valid"]] if len(endpoints) else endpoints
    return {
        "rows": int(len(frame)),
        "flights": int(frame["flight_id"].nunique()),
        "endpoints": int(len(endpoints)),
        "valid_endpoints": int(len(valid)),
        "window": window,
        "target_min": float(valid["target"].min()) if len(valid) else None,
        "target_max": float(valid["target"].max()) if len(valid) else None,
    }


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("torch is required for non-dry-run benchmark") from exc


def _train_one(x_train, h_train, v_train, y_train, x_val, h_val, v_val, y_val,
               *, kind: str, hidden_dim: int, num_layers: int, seed: int,
               loss_mode: str, epochs: int = 120, batch_size: int = 128,
               history: list[dict] | None = None, val_apex_mask=None):
    _require_torch()
    import torch
    from apogee_models import (
        GRUApogeeModel, SnapshotApogeeModel, project_apogee,
        raw_apogee, raw_apogee_loss, score_state_dict,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if kind == "gru":
        model = GRUApogeeModel(
            input_dim=len(ap.FEATURE_COLUMNS), hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(device)
    elif kind == "mlp":
        model = SnapshotApogeeModel(
            input_dim=len(ap.FEATURE_COLUMNS), hidden_dim=hidden_dim,
        ).to(device)
    else:
        raise ValueError(f"unknown model kind: {kind}")

    def forward(xb, hb, vb):
        if kind == "gru":
            r = model(xb)[0]
        else:
            r = model(xb[:, -1, :])
        h_raw = raw_apogee(hb, vb, r)
        h_proj, _, _ = project_apogee(hb, vb, r)
        return r, h_raw, h_proj

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    train_x = torch.tensor(x_train, dtype=torch.float32, device=device)
    train_h = torch.tensor(h_train, dtype=torch.float32, device=device)
    train_v = torch.tensor(v_train, dtype=torch.float32, device=device)
    train_y = torch.tensor(y_train, dtype=torch.float32, device=device)
    val_x = torch.tensor(x_val, dtype=torch.float32, device=device)
    val_h = torch.tensor(h_val, dtype=torch.float32, device=device)
    val_v = torch.tensor(v_val, dtype=torch.float32, device=device)
    val_y = torch.tensor(y_val, dtype=torch.float32, device=device)
    if history is not None:
        apex_mask = torch.as_tensor(val_apex_mask, dtype=torch.bool, device=device)
        if apex_mask.shape != val_h.shape or not apex_mask.any():
            raise ValueError("epoch tracing requires a nonempty last-1s validation mask")
    best = float("inf")
    best_state = None
    patience = 0
    for epoch in range(1, epochs + 1):
        if history is not None:
            train_loss_sum = train_h.new_zeros(())
            lr = optimizer.param_groups[0]["lr"]
        model.train()
        order = torch.randperm(len(train_x), device=device)
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            optimizer.zero_grad()
            r, h_raw, h_proj = forward(train_x[idx], train_h[idx], train_v[idx])
            target_h = (train_h[idx] + train_v[idx] ** 2 / (2 * 9.81)) * train_y[idx]
            if loss_mode == "raw":
                loss = raw_apogee_loss(r, train_y[idx], h_raw, target_h)
            else:
                # Same r term and height weight; only the height term uses
                # projection.  This isolates projection from objective changes.
                loss = torch.mean((r - train_y[idx]) ** 2) + 0.1 * torch.mean(
                    (h_proj / 100.0 - target_h / 100.0) ** 2
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if history is not None:
                train_loss_sum += loss.detach() * len(idx)
        model.eval()
        with torch.no_grad():
            r, h_raw, _ = forward(val_x, val_h, val_v)
            target_h = (val_h + val_v ** 2 / (2 * 9.81)) * val_y
            val_loss = float(raw_apogee_loss(r, val_y, h_raw, target_h).item())
        scheduler.step(val_loss)
        if val_loss < best:
            best = val_loss
            best_state = score_state_dict(model.state_dict())
            patience = 0
        else:
            patience += 1
        if history is None:
            if patience >= 12:
                break
        else:
            history.append({
                "epoch": epoch,
                "train_loss": float(train_loss_sum.item()) / len(train_x),
                "validation_loss": val_loss,
                "learning_rate": lr,
                "apex_last_1s_mae_m": float((h_raw[apex_mask] - target_h[apex_mask]).abs().mean().item()),
                "epochs_since_best": patience,
            })
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def _target_height(h: np.ndarray, v: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pure NumPy target used by the evaluator; returns ndarray."""
    h = np.asarray(h, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return (h + v**2 / (2 * 9.81)) * y


def _evaluate(model, kind, x, h, v, y):
    _require_torch()
    import torch
    from apogee_models import project_apogee, raw_apogee
    device = next(model.parameters()).device
    with torch.no_grad():
        xt = torch.tensor(x, dtype=torch.float32, device=device)
        ht = torch.tensor(h, dtype=torch.float32, device=device)
        vt = torch.tensor(v, dtype=torch.float32, device=device)
        r = model(xt)[0] if kind == "gru" else model(xt[:, -1, :])
        h_raw = raw_apogee(ht, vt, r)
        h_proj, _, below = project_apogee(ht, vt, r)
    target = _target_height(h, v, y)
    return {
        "r_raw": r.cpu().numpy(),
        "h_raw": h_raw.cpu().numpy(),
        "h_projected": h_proj.cpu().numpy(),
        "below_current_altitude": below.cpu().numpy(),
        "target": target,
    }


def _scale_array(scaler, array: np.ndarray) -> np.ndarray:
    return scaler.transform(
        array.reshape(-1, array.shape[-1])
    ).reshape(array.shape).astype(np.float32)


def _save_run_bundle(
    run_dir: Path,
    *,
    model,
    scaler,
    config: Mapping,
) -> None:
    import joblib
    import torch
    from apogee_models import score_state_dict

    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(score_state_dict(model.state_dict()), run_dir / "model.pth")
    joblib.dump(scaler, run_dir / "scaler.pkl")
    ap.write_json(run_dir / "run.json", config)


def _run_learning_curve(args) -> int:
    _require_torch()
    data_path = Path(args.data)
    frame = _read_data(data_path)
    flight_ids = np.sort(frame["flight_id"].unique())
    split = _load_split(Path(args.split_manifest) if args.split_manifest else None, flight_ids)
    features, endpoints = _benchmark_inputs(frame, args)
    selected_endpoints = _select_endpoints(frame, endpoints, args.endpoint_phase)
    state_metadata = _state_metadata(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for window in args.windows:
        val_x, val_h, val_v, val_y, val_f = _samples_for_window(
            features, selected_endpoints, split["validation"], window
        )
        for seed in args.seeds:
            shuffled = np.random.RandomState(seed).permutation(split["train"])
            for n_train in args.curve_sizes:
                selected = sorted(shuffled[:n_train].tolist())
                train_x, train_h, train_v, train_y, _ = _samples_for_window(
                    features, selected_endpoints, selected, window
                )
                scaler = _fit_endpoint_scaler(features, endpoints, selected)
                model = _train_one(
                    _scale_array(scaler, train_x), train_h, train_v, train_y,
                    _scale_array(scaler, val_x), val_h, val_v, val_y,
                    kind="gru", hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                    seed=seed, loss_mode="raw", epochs=args.epochs,
                )
                pred = _evaluate(
                    model, "gru", _scale_array(scaler, val_x), val_h, val_v, val_y
                )
                raw_err = pred["h_raw"] - pred["target"]
                row = {
                    "window": window, "seed": seed, "train_flights": n_train,
                    "n_train_windows": int(len(train_x)), "n_val_windows": int(len(val_x)),
                    "val_flight_mae_raw": _flight_mae(raw_err, val_f),
                    "val_window_mae_raw": float(np.mean(np.abs(raw_err))),
                }
                rows.append(row)
                run_dir = output_dir / "learning_curve" / f"w{window}_n{n_train}_seed{seed}"
                _save_run_bundle(run_dir, model=model, scaler=scaler, config={
                    **row, "model": "gru", "features": list(ap.FEATURE_COLUMNS),
                    **state_metadata, "loss": "raw",
                    "acceleration_encoding": "inertial", "scaler_encoding": "inertial",
                    "dt": 0.02, "stride": 10, "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers, "epochs": args.epochs,
                    "data": str(data_path), "data_sha256": ap.sha256_file(data_path),
                    "split": split,
                })
                print(json.dumps(row, ensure_ascii=False))
    pd.DataFrame(rows).to_csv(output_dir / "learning_curve.csv", index=False)
    return 0


def _run_benchmark(args) -> int:
    data_path = Path(args.data)
    frame = _read_data(data_path)
    flight_ids = np.sort(frame["flight_id"].unique())
    split = _load_split(Path(args.split_manifest) if args.split_manifest else None, flight_ids)
    features, endpoints = _benchmark_inputs(frame, args)
    selected_endpoints = _select_endpoints(frame, endpoints, args.endpoint_phase)
    print(f"data={data_path} rows={len(frame)} flights={len(flight_ids)}")
    print(f"split={ {k: len(v) for k, v in split.items()} }")
    for window in args.windows:
        summary = _dataset_summary(features, selected_endpoints, window)
        print(f"window={window} valid_endpoints={summary['valid_endpoints']}/{summary['endpoints']} "
              f"target={summary['target_min']}..{summary['target_max']}")
    if args.dry_run:
        print(f"features={list(ap.FEATURE_COLUMNS)}")
        print(f"state_source={args.state_source} endpoint_phase={args.endpoint_phase} "
              f"loss={args.loss} output_dir={args.output_dir}")
        print("dry-run: no torch import, no files written")
        return 0

    _require_torch()
    state_metadata = _state_metadata(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ap.write_json(output_dir / "split_manifest.json", split)
    # Hold normalization fixed when comparing endpoint phases; train flights only.
    scaler = _fit_endpoint_scaler(features, endpoints, split["train"])
    all_rows = []
    for window in args.windows:
        train_x, train_h, train_v, train_y, _ = _samples_for_window(
            features, selected_endpoints, split["train"], window)
        val_x, val_h, val_v, val_y, _ = _samples_for_window(
            features, selected_endpoints, split["validation"], window)
        test_x, test_h, test_v, test_y, test_f = _samples_for_window(
            features, selected_endpoints, split["test"], window)
        if not len(train_x) or not len(val_x) or not len(test_x):
            raise RuntimeError(f"empty train, validation, or test samples for window {window}")
        train_x, val_x, test_x = (
            _scale_array(scaler, train_x), _scale_array(scaler, val_x), _scale_array(scaler, test_x)
        )
        for kind in args.models:
            for seed in args.seeds:
                model = _train_one(
                    train_x, train_h, train_v, train_y, val_x, val_h, val_v, val_y,
                    kind=kind, hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                    seed=seed, loss_mode=args.loss, epochs=args.epochs,
                )
                pred = _evaluate(model, kind, test_x, test_h, test_v, test_y)
                denom = test_h + test_v ** 2 / (2 * 9.81)
                baseline = denom - denom * test_y
                floor_err = test_h - denom * test_y
                raw_err = pred["h_raw"] - pred["target"]
                proj_err = pred["h_projected"] - pred["target"]
                row = {
                    "window": window, "model": kind, "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers, "seed": seed, "loss": args.loss,
                    "n_test": int(len(test_y)),
                    "flight_mae_raw": _flight_mae(raw_err, test_f),
                    "flight_mae_projected": _flight_mae(proj_err, test_f),
                    "flight_mae_constant_r": _flight_mae(baseline, test_f),
                    "flight_mae_floor_only": _flight_mae(floor_err, test_f),
                    "window_mae_raw": float(np.mean(np.abs(raw_err))),
                    "below_current_altitude_rate": float(np.mean(pred["below_current_altitude"])),
                }
                all_rows.append(row)
                run_dir = output_dir / f"w{window}_{kind}_h{args.hidden_dim}_l{args.num_layers}_seed{seed}_{args.loss}"
                _save_run_bundle(run_dir, model=model, scaler=scaler, config={
                    **row, "features": list(ap.FEATURE_COLUMNS),
                    **state_metadata, "loss": args.loss,
                    "dt": 0.02, "stride": 10, "epochs": args.epochs,
                    "data": str(data_path), "data_sha256": ap.sha256_file(data_path),
                    "split": split,
                })
                print(json.dumps(row, ensure_ascii=False))
    pd.DataFrame(all_rows).to_csv(output_dir / "benchmark.csv", index=False)
    ap.write_json(output_dir / "run_config.json", {
        "features": list(ap.FEATURE_COLUMNS),
        **state_metadata,
        "acceleration_encoding": "inertial",
        "scaler_encoding": "inertial",
        "loss": args.loss,
        "windows": args.windows,
        "seeds": args.seeds,
        "models": args.models,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "epochs": args.epochs,
        "dt": 0.02,
        "stride": 10,
        "data": str(data_path),
        "data_sha256": ap.sha256_file(data_path),
        "n_common_endpoints": int(selected_endpoints["window_valid"].sum()) if len(selected_endpoints) else 0,
        "split": split,
        "output_dir": str(output_dir),
    })
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/simulated/all_trajectories.csv.gz")
    parser.add_argument("--split-manifest")
    parser.add_argument("--output-dir", help="default: data/experiments/apogee-v2[-m7-ukf][-coasting]")
    parser.add_argument("--windows", type=int, nargs="+", default=[25, 50, 100])
    parser.add_argument("--models", nargs="+", choices=["gru", "mlp"], default=["gru", "mlp"])
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--loss", choices=["raw", "projected"], default="raw")
    parser.add_argument("--state-source", choices=["simulation_clean", "estimated", "m7_ukf"],
                        default="simulation_clean")
    parser.add_argument("--ukf-replay", type=Path, help="host ukf_replay executable for m7_ukf")
    parser.add_argument("--endpoint-phase", choices=["all", "coasting"], default="all",
                        help="filter prediction instants for train/validation/test, not history")
    parser.add_argument("--learning-curve", action="store_true")
    parser.add_argument("--curve-sizes", type=int, nargs="+", default=[30, 60, 120])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.state_source == "m7_ukf":
        if args.ukf_replay is None or not args.ukf_replay.is_file():
            parser.error("m7_ukf requires --ukf-replay pointing to the host executable")
    elif args.ukf_replay is not None:
        parser.error("--ukf-replay requires --state-source m7_ukf")
    if args.output_dir is None:
        suffix = "-m7-ukf" if args.state_source == "m7_ukf" else ""
        if args.endpoint_phase == "coasting":
            suffix += "-coasting"
        args.output_dir = f"data/experiments/apogee-v2{suffix}"
    if args.learning_curve and not args.dry_run:
        raise SystemExit(_run_learning_curve(args))
    raise SystemExit(_run_benchmark(args))


if __name__ == "__main__":
    main()
