#!/usr/bin/env python
"""Benchmark common endpoints and history lengths without touching artifacts.

``--dry-run`` is NumPy/pandas-only.  Actual training imports torch lazily and
writes only under --output-dir.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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


def _samples_for_window(
    sampled: pd.DataFrame,
    endpoints: pd.DataFrame,
    flight_ids: list[int],
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs, hs, vs, ys, fids = [], [], [], [], []
    for flight_id in flight_ids:
        flight = sampled[sampled["flight_id"] == flight_id].reset_index(drop=True)
        ep = endpoints[endpoints["flight_id"] == flight_id]
        ep = ep[(ep["endpoint_index"] >= window - 1) & (ep["endpoint_index"] < len(flight))]
        # Shared endpoints are fixed by max_window=100; shorter windows only
        # change the suffix.  Exclude a common endpoint if any suffix is invalid.
        keep = []
        for index in ep["endpoint_index"].to_numpy(dtype=int):
            if flight["valid"].iloc[index - window + 1:index + 1].all():
                keep.append(True)
            else:
                keep.append(False)
        ep = ep[np.asarray(keep, dtype=bool)]
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
    return {
        "rows": int(len(frame)),
        "flights": int(frame["flight_id"].nunique()),
        "endpoints": int(len(endpoints)),
        "window": window,
        "target_min": float(endpoints["target"].min()) if len(endpoints) else None,
        "target_max": float(endpoints["target"].max()) if len(endpoints) else None,
    }


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("torch is required for non-dry-run benchmark") from exc


def _train_one(x_train, h_train, v_train, y_train, x_val, h_val, v_val, y_val,
               *, kind: str, hidden_dim: int, num_layers: int, seed: int,
               loss_mode: str, epochs: int = 120, batch_size: int = 128):
    _require_torch()
    import torch
    from apogee_models import GRUApogeeModel, SnapshotApogeeModel, raw_apogee, raw_apogee_loss, score_state_dict

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
        return r, raw_apogee(hb, vb, r)

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
    best = float("inf")
    best_state = None
    patience = 0
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(train_x), device=device)
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            optimizer.zero_grad()
            r, h_pred = forward(train_x[idx], train_h[idx], train_v[idx])
            if loss_mode == "raw":
                loss = raw_apogee_loss(r, train_y[idx], h_pred, (train_h[idx] + train_v[idx] ** 2 / (2 * 9.81)) * train_y[idx])
            else:
                loss = torch.mean((h_pred - (train_h[idx] + train_v[idx] ** 2 / (2 * 9.81)) * train_y[idx]) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            r, h_pred = forward(val_x, val_h, val_v)
            target_h = (val_h + val_v ** 2 / (2 * 9.81)) * val_y
            val_loss = float(torch.mean((h_pred - target_h) ** 2).item())
        scheduler.step(val_loss)
        if val_loss < best:
            best = val_loss
            best_state = score_state_dict(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= 12:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


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
    denom = h + v ** 2 / (2 * 9.81)
    target = denom * y
    return {
        "r_raw": r.cpu().numpy(),
        "h_raw": h_raw.cpu().numpy(),
        "h_projected": h_proj.cpu().numpy(),
        "below_current_altitude": below.cpu().numpy(),
        "target": target.cpu().numpy(),
    }


def _run_learning_curve(args) -> int:
    _require_torch()
    from sklearn.preprocessing import StandardScaler

    data_path = Path(args.data)
    frame = _read_data(data_path)
    flight_ids = np.sort(frame["flight_id"].unique())
    split = _load_split(Path(args.split_manifest) if args.split_manifest else None, flight_ids)
    sampled, endpoints = _resample_and_endpoints(frame)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for window in args.windows:
        val_x, val_h, val_v, val_y, val_f = _samples_for_window(
            sampled, endpoints, split["validation"], window
        )
        for seed in args.seeds:
            shuffled = np.random.RandomState(seed).permutation(split["train"])
            for n_train in args.curve_sizes:
                selected = sorted(shuffled[:n_train].tolist())
                train_x, train_h, train_v, train_y, _ = _samples_for_window(
                    sampled, endpoints, selected, window
                )
                scaler = StandardScaler().fit(train_x.reshape(-1, train_x.shape[-1]))
                def scale(a):
                    return scaler.transform(a.reshape(-1, a.shape[-1])).reshape(a.shape).astype(np.float32)
                model = _train_one(
                    scale(train_x), train_h, train_v, train_y,
                    scale(val_x), val_h, val_v, val_y,
                    kind="gru", hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                    seed=seed, loss_mode="raw", epochs=args.epochs,
                )
                pred = _evaluate(model, "gru", scale(val_x), val_h, val_v, val_y)
                raw_err = pred["h_raw"] - pred["target"]
                rows.append({
                    "window": window, "seed": seed, "train_flights": n_train,
                    "n_train_windows": int(len(train_x)), "n_val_windows": int(len(val_x)),
                    "val_flight_mae_raw": _flight_mae(raw_err, val_f),
                    "val_window_mae_raw": float(np.mean(np.abs(raw_err))),
                })
                print(json.dumps(rows[-1], ensure_ascii=False))
    pd.DataFrame(rows).to_csv(output_dir / "learning_curve.csv", index=False)
    return 0


def _run_benchmark(args) -> int:
    data_path = Path(args.data)
    frame = _read_data(data_path)
    flight_ids = np.sort(frame["flight_id"].unique())
    split = _load_split(Path(args.split_manifest) if args.split_manifest else None, flight_ids)
    sampled, endpoints = _resample_and_endpoints(frame)
    if args.state_source == "estimated":
        state = ap.causal_state_estimate(sampled)
        endpoints = endpoints.merge(
            state, how="left",
            left_on=["flight_id", "endpoint_time"], right_on=["flight_id", "time"],
        )
        if endpoints["h_est"].isna().any():
            raise RuntimeError("estimated state missing at an endpoint")
        endpoints["h"] = endpoints["h_est"]
        endpoints["v"] = endpoints["v_est"]
        endpoints["target"] = endpoints["apogee_rel"] / (
            endpoints["h"] + endpoints["v"] ** 2 / (2 * ap.SIM_G)
        )
    print(f"data={data_path} rows={len(frame)} flights={len(flight_ids)}")
    print(f"split={ {k: len(v) for k, v in split.items()} }")
    for window in args.windows:
        summary = _dataset_summary(sampled, endpoints, window)
        print(f"window={window} endpoints={summary['endpoints']} target={summary['target_min']}..{summary['target_max']}")
    if args.dry_run:
        print(f"features={list(ap.FEATURE_COLUMNS)}")
        print(f"state_source={args.state_source} loss={args.loss} output_dir={args.output_dir}")
        print("dry-run: no torch import, no files written")
        return 0

    _require_torch()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ap.write_json(output_dir / "split_manifest.json", split)
    all_rows = []
    for window in args.windows:
        train_x, train_h, train_v, train_y, _ = _samples_for_window(sampled, endpoints, split["train"], window)
        val_x, val_h, val_v, val_y, _ = _samples_for_window(sampled, endpoints, split["validation"], window)
        test_x, test_h, test_v, test_y, test_f = _samples_for_window(sampled, endpoints, split["test"], window)
        if not len(train_x) or not len(test_x):
            raise RuntimeError(f"empty samples for window {window}")
        # StandardScaler is fit on train endpoints only; this is a local
        # benchmark scaler and is saved with the run, not in data/simulated.
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(train_x.reshape(-1, train_x.shape[-1]))
        def scale(a):
            return scaler.transform(a.reshape(-1, a.shape[-1])).reshape(a.shape).astype(np.float32)
        train_x, val_x, test_x = scale(train_x), scale(val_x), scale(test_x)
        for kind in args.models:
            for seed in args.seeds:
                model = _train_one(
                    train_x, train_h, train_v, train_y, val_x, val_h, val_v, val_y,
                    kind=kind, hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                    seed=seed, loss_mode=args.loss,
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
                print(json.dumps(row, ensure_ascii=False))
    pd.DataFrame(all_rows).to_csv(output_dir / "benchmark.csv", index=False)
    ap.write_json(output_dir / "run_config.json", {
        "features": list(ap.FEATURE_COLUMNS),
        "state_source": args.state_source,
        "loss": args.loss,
        "windows": args.windows,
        "seeds": args.seeds,
        "models": args.models,
        "split": split,
        "data": str(data_path),
        "output_dir": str(output_dir),
    })
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/simulated/all_trajectories.csv.gz")
    parser.add_argument("--split-manifest")
    parser.add_argument("--output-dir", default="data/experiments/apogee-v2")
    parser.add_argument("--windows", type=int, nargs="+", default=[25, 50, 100])
    parser.add_argument("--models", nargs="+", choices=["gru", "mlp"], default=["gru", "mlp"])
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--loss", choices=["raw", "projected"], default="raw")
    parser.add_argument("--state-source", choices=["simulation_clean", "estimated"], default="simulation_clean")
    parser.add_argument("--learning-curve", action="store_true")
    parser.add_argument("--curve-sizes", type=int, nargs="+", default=[30, 60, 120])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.learning_curve and not args.dry_run:
        raise SystemExit(_run_learning_curve(args))
    raise SystemExit(_run_benchmark(args))


if __name__ == "__main__":
    main()
