#!/usr/bin/env python
"""Experiments A (input state) and B (physics residual) for the M7 GRU.

Plans: data/experiments/apogee-v2-m7-inputs/PLAN.md and
data/experiments/apogee-v2-m7-residual/PLAN.md.  Every variant uses the same
M7 UKF replay, split, coast endpoints, window, model size and training loop
as ``benchmark_apogee.py --state-source m7_ukf --endpoint-phase coasting``;
only the input columns and the output head change.

    A0 ukf            current six features, head (h_ukf + v_ukf^2/2g) * r
    A1 smoothed       baro, v_s, accel, tilt, gyro, q(h_s, v_s); head on h_s, v_s
    A2 ukf+smoothed   current six + h_s, v_s;                    head on h_s, v_s
    B0 physics        H_phys from h_s, v_s and online k (no learning)
    B1 residual       A1 + coast flag, k, H_phys - h_s;   H = H_phys + 20 (r - 1)

h_s, v_s come from ``apogee_physics.kinematic_smoother`` (1.5 s causal least
squares of barometer altitude with the measured acceleration).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import apogee_pipeline as ap  # noqa: E402
import apogee_physics as phy  # noqa: E402
import benchmark_apogee as bench  # noqa: E402

WINDOW = 100
RESIDUAL_SCALE_M = 20.0  # Delta = 20 (r - 1), |Delta| < 10 m with r = 1 + 0.5 tanh
BANDS = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 10.0)
UKF_FEATURES = list(ap.FEATURE_COLUMNS)
SMOOTH_FEATURES = ["altitude_noisy_rel", "v_s", "acceleration_inertial",
                   "tilt_angle", "gyro_roll_noisy", "q_s"]


@dataclass(frozen=True)
class Variant:
    name: str
    experiment: str
    features: tuple[str, ...]
    head_h: str
    head_v: str
    residual: bool = False


VARIANTS = {
    "A0_ukf": Variant("A0_ukf", "inputs", tuple(UKF_FEATURES), "altitude_rel", "velocity_z"),
    "A1_smoothed": Variant("A1_smoothed", "inputs", tuple(SMOOTH_FEATURES), "h_s", "v_s"),
    "A2_ukf_smoothed": Variant("A2_ukf_smoothed", "inputs", tuple(UKF_FEATURES + ["h_s", "v_s"]), "h_s", "v_s"),
    "B1_residual": Variant("B1_residual", "residual",
                           tuple(SMOOTH_FEATURES + ["coast_flag", "k_feature", "phys_climb"]),
                           "h_s", "v_s", residual=True),
}


def add_derived_columns(grid: pd.DataFrame, config: phy.CoastConfig | None = None) -> pd.DataFrame:
    """Smoothed state, its dynamic pressure and the physics baseline columns.

    Before coast is confirmed the physics columns are undefined; the NN
    features fill them with 0 and expose ``coast_flag`` instead, while
    ``h_phys`` stays NaN so it can never be scored there.
    """
    out = grid.copy()
    smooth = phy.kinematic_smoother(out, baro_col="altitude_noisy_rel")
    out["h_s"] = smooth["h_smooth"].to_numpy()
    out["v_s"] = smooth["v_smooth"].to_numpy()
    out["q_s"] = 0.5 * 1.225 * np.exp(-0.00012 * out["h_s"].abs()) * out["v_s"] ** 2
    physics = phy.online_ballistic_apogee(
        out.assign(altitude_rel=out["h_s"], velocity_z=out["v_s"]), config=config)
    out["h_phys"] = physics["apogee_ballistic"].to_numpy()
    out["coast_flag"] = physics["coast"].astype(float).to_numpy()
    out["k_feature"] = physics["k_est"].fillna(0.0).to_numpy()
    out["phys_climb"] = (out["h_phys"] - out["h_s"]).fillna(0.0)
    return out


def residual_height(h_phys, r):
    """Firmware-compatible residual head: same 1 + 0.5 tanh output as Kernel::ratio()."""
    return h_phys + RESIDUAL_SCALE_M * (r - 1.0)


def common_endpoints(grid: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    """Coast endpoints valid for every variant (same set for A and B)."""
    needed = sorted({c for v in VARIANTS.values() for c in v.features} | {"h_s", "v_s"})
    rows = []
    for flight_id, ep in selected[selected["window_valid"]].groupby("flight_id", sort=False):
        flight = grid[grid["flight_id"] == flight_id].reset_index(drop=True)
        values = flight[needed].to_numpy(dtype=float)
        h_phys = flight["h_phys"].to_numpy(dtype=float)
        for _, row in ep.iterrows():
            e = int(row["endpoint_index"])
            if e < WINDOW - 1 or not np.isfinite(h_phys[e]):
                continue
            if not np.isfinite(values[e - WINDOW + 1:e + 1]).all():
                continue
            rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def samples(grid: pd.DataFrame, endpoints: pd.DataFrame, flight_ids, variant: Variant):
    xs, hs, vs, hp, ytrue, meta = [], [], [], [], [], []
    for flight_id in flight_ids:
        ep = endpoints[endpoints["flight_id"] == flight_id]
        if ep.empty:
            continue
        flight = grid[grid["flight_id"] == flight_id].reset_index(drop=True)
        idx = ep["endpoint_index"].to_numpy(dtype=int)
        xs.append(ap.slice_history(flight[list(variant.features)].to_numpy(dtype=np.float32), idx, WINDOW))
        hs.append(flight[variant.head_h].to_numpy(dtype=np.float32)[idx])
        vs.append(flight[variant.head_v].to_numpy(dtype=np.float32)[idx])
        hp.append(flight["h_phys"].to_numpy(dtype=np.float32)[idx])
        ytrue.append(ep["apogee_rel"].to_numpy(dtype=np.float32))
        meta.append(ep[["flight_id", "endpoint_time"]])
    return (np.concatenate(xs), np.concatenate(hs), np.concatenate(vs), np.concatenate(hp),
            np.concatenate(ytrue), pd.concat(meta, ignore_index=True))


def fit_scaler(grid: pd.DataFrame, all_endpoints: pd.DataFrame, train_ids, features):
    """Train flights only, every valid endpoint (same rule as benchmark_apogee)."""
    from sklearn.preprocessing import StandardScaler

    rows = []
    for flight_id in train_ids:
        ep = all_endpoints[(all_endpoints["flight_id"] == flight_id) & all_endpoints["window_valid"]]
        if ep.empty:
            continue
        flight = grid[grid["flight_id"] == flight_id].reset_index(drop=True)
        values = flight.iloc[ep["endpoint_index"].to_numpy(dtype=int)][list(features)].to_numpy(dtype=float)
        rows.append(values[np.isfinite(values).all(axis=1)])
    return StandardScaler().fit(np.concatenate(rows))


def train(variant: Variant, train_data, val_data, *, seed: int, epochs: int, hidden: int = 32, layers: int = 1):
    import torch
    from apogee_models import GRUApogeeModel, raw_apogee_loss, score_state_dict

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GRUApogeeModel(input_dim=len(variant.features), hidden_dim=hidden, num_layers=layers)
    t = [torch.as_tensor(a, dtype=torch.float32) for a in train_data[:5]]
    v = [torch.as_tensor(a, dtype=torch.float32) for a in val_data[:5]]

    def loss_fn(x, h, vz, hphys, y):
        r = model(x)[0]
        if variant.residual:
            return torch.mean(((residual_height(hphys, r) - y) / 10.0) ** 2)
        denom = h + vz ** 2 / (2 * ap.SIM_G)
        return raw_apogee_loss(r, y / denom, denom * r, y)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    best, best_state, patience, history = float("inf"), None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(t[0]))
        for start in range(0, len(order), 128):
            idx = order[start:start + 128]
            optimizer.zero_grad()
            loss = loss_fn(*(a[idx] for a in t))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(*v).item())
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "validation_loss": val_loss})
        if val_loss < best:
            best, best_state, patience = val_loss, score_state_dict(model.state_dict()), 0
        else:
            patience += 1
            if patience >= 12:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def predict(model, variant: Variant, data):
    import torch

    x, h, v, hphys, _ = (torch.as_tensor(a, dtype=torch.float32) for a in data[:5])
    with torch.no_grad():
        r = model(x)[0]
        height = residual_height(hphys, r) if variant.residual else (h + v ** 2 / (2 * ap.SIM_G)) * r
    return height.numpy().astype(float), r.numpy().astype(float)


def regions(pred: pd.DataFrame) -> pd.DataFrame:
    """Same regions as apogee-v2-m7-capacity plus time-to-apogee bands."""
    pred = pred.reset_index(drop=True)
    first = pred.groupby("flight_id")["endpoint_time"].transform("min")
    last = pred.groupby("flight_id")["endpoint_time"].transform("max")
    masks = {
        "coast_all": np.ones(len(pred), dtype=bool),
        "first_1s": (pred["endpoint_time"] - first).to_numpy() < 1.0 - 1e-9,
        "last_1s": (pred["time_to_apogee_s"] <= 1.0).to_numpy(),
        "last_per_flight": (pred["endpoint_time"] == last).to_numpy(),
    }
    for lo, hi in zip(BANDS[:-1], BANDS[1:]):
        masks[f"tta_{lo:g}-{hi:g}s"] = ((pred["time_to_apogee_s"] > lo) & (pred["time_to_apogee_s"] <= hi)).to_numpy()
    rows = []
    for (model, seed), group in pred.groupby(["model", "seed"], sort=False):
        for region, mask in masks.items():
            sel = group[mask[group.index.to_numpy()]]
            if sel.empty:
                continue
            err = sel["prediction"] - sel["apogee_true"]
            rows.append({"model": model, "seed": seed, "region": region, "n": len(sel),
                         "mae": float(err.abs().mean()), "bias": float(err.mean()),
                         "p95": float(err.abs().quantile(0.95)), "max": float(err.abs().max()),
                         "over_5m": int((err > 5).sum())})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment", choices=("inputs", "residual", "all"), default="all")
    parser.add_argument("--data", type=Path, default=ROOT / "data/simulated/all_trajectories.csv.gz")
    parser.add_argument("--split", type=Path, default=ROOT / "data/experiments/apogee-v2/split_manifest.json")
    parser.add_argument("--ukf-replay", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--epochs", type=int, default=120)
    args = parser.parse_args(argv)

    frame = bench._read_data(args.data)
    split = bench._load_split(args.split, np.sort(frame["flight_id"].unique()))
    state_args = SimpleNamespace(state_source="m7_ukf", ukf_replay=args.ukf_replay,
                                 endpoint_phase="coasting")
    grid, endpoints = bench._benchmark_inputs(frame, state_args)
    selected = bench._select_endpoints(frame, endpoints, "coasting")
    grid = add_derived_columns(grid)
    common = common_endpoints(grid, selected)
    apex_t = frame.loc[frame.groupby("flight_id")["altitude_rel"].idxmax()].set_index("flight_id")["time"]
    common["time_to_apogee_s"] = common["flight_id"].map(apex_t) - common["endpoint_time"]
    print(f"coast endpoints: selected {int(selected['window_valid'].sum())}, common {len(common)} "
          f"(test {int(common['flight_id'].isin(split['test']).sum())})")

    names = [n for n, v in VARIANTS.items() if args.experiment in ("all", v.experiment)]
    test_rows = []
    test_meta = common[common["flight_id"].isin(split["test"])][["flight_id", "endpoint_time", "time_to_apogee_s"]]
    physics = test_meta.merge(
        grid[["flight_id", "time", "h_phys"]].rename(columns={"time": "endpoint_time"}),
        on=["flight_id", "endpoint_time"], how="left", validate="one_to_one")
    for seed in args.seeds:
        test_rows.append(physics.assign(model="B0_physics", seed=seed, prediction=physics["h_phys"], r=np.nan))

    run_meta = {}
    for name in names:
        variant = VARIANTS[name]
        scaler = fit_scaler(grid, endpoints, split["train"], variant.features)
        data = {}
        for part in ("train", "validation", "test"):
            x, h, v, hp, y, meta = samples(grid, common, split[part], variant)
            x = scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape).astype(np.float32)
            data[part] = (x, h, v, hp, y, meta)
        for seed in args.seeds:
            model, history = train(variant, data["train"], data["validation"], seed=seed, epochs=args.epochs)
            height, r = predict(model, variant, data["test"])
            meta = data["test"][5].merge(test_meta, on=["flight_id", "endpoint_time"], how="left")
            test_rows.append(meta.assign(model=name, seed=seed, prediction=height, r=r))
            out_dir = ROOT / f"data/experiments/apogee-v2-m7-{variant.experiment}" / f"{name}_seed{seed}"
            import joblib
            import torch
            from apogee_models import score_state_dict
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(score_state_dict(model.state_dict()), out_dir / "model.pth")
            joblib.dump(scaler, out_dir / "scaler.pkl")
            pd.DataFrame(history).to_csv(out_dir / "epoch_history.csv", index=False)
            ap.write_json(out_dir / "run.json", {
                "variant": name, "features": list(variant.features), "head_h": variant.head_h,
                "head_v": variant.head_v, "residual": variant.residual,
                "residual_scale_m": RESIDUAL_SCALE_M if variant.residual else None,
                "seed": seed, "window": WINDOW, "hidden_dim": 32, "num_layers": 1, "epochs": args.epochs,
                "best_epoch": int(np.argmin([h["validation_loss"] for h in history]) + 1),
                "n_train": int(len(data["train"][0])), "n_validation": int(len(data["validation"][0])),
                "n_test": int(len(data["test"][0])), "state_source": "m7_ukf",
                "ukf_replay_sha256": ap.sha256_file(args.ukf_replay),
                "data_sha256": ap.sha256_file(args.data), "smoother_window_s": 1.5,
                "coast_config": phy.CoastConfig().__dict__,
            })
            run_meta[f"{name}_seed{seed}"] = int(np.argmin([h["validation_loss"] for h in history]) + 1)
            print(f"{name} seed {seed}: best epoch {run_meta[f'{name}_seed{seed}']}", flush=True)

    pred = pd.concat(test_rows, ignore_index=True)
    truth = common[["flight_id", "endpoint_time", "apogee_rel"]].rename(columns={"apogee_rel": "apogee_true"})
    pred = pred.merge(truth, on=["flight_id", "endpoint_time"], how="left", validate="many_to_one")
    for experiment in sorted({VARIANTS[n].experiment for n in names}):
        out = ROOT / f"data/experiments/apogee-v2-m7-{experiment}"
        keep = {"B0_physics"} | {n for n in names if VARIANTS[n].experiment == experiment}
        part = pred[pred["model"].isin(keep)]
        part.to_csv(out / "predictions.csv", index=False)
        regions(part).to_csv(out / "summary.csv", index=False)
        ap.write_json(out / "run_config.json", {
            "variants": sorted(keep), "seeds": args.seeds, "epochs": args.epochs, "window": WINDOW,
            "split": args.split.resolve().relative_to(ROOT).as_posix(),
            "endpoint_rule": "flight_phase coasting & 100-step window valid & physics defined & all features finite",
            "n_common_endpoints": int(len(common)),
            "best_epochs": {k: v for k, v in run_meta.items() if k.rsplit("_seed", 1)[0] in keep},
            "data_sha256": ap.sha256_file(args.data), "ukf_replay_sha256": ap.sha256_file(args.ukf_replay),
        })
    # Cross-experiment table: everything predicted on the common endpoints so far.
    stored = [pd.read_csv(ROOT / f"data/experiments/apogee-v2-m7-{e}/predictions.csv")
              for e in ("inputs", "residual") if (ROOT / f"data/experiments/apogee-v2-m7-{e}/predictions.csv").exists()]
    combined = pd.concat(stored, ignore_index=True).drop_duplicates(["model", "seed", "flight_id", "endpoint_time"])
    summary = regions(combined.reset_index(drop=True))
    summary.to_csv(ROOT / "data/experiments/apogee-v2-m7-residual/comparison.csv", index=False)
    view = summary[summary["region"].isin(["coast_all", "first_1s", "last_1s", "last_per_flight"])]
    with pd.option_context("display.width", 200):
        print(view.pivot_table(index=["model", "seed"], columns="region", values="mae").round(3).to_string())
        print(view[view["region"] == "coast_all"][["model", "seed", "mae", "bias", "p95", "over_5m"]]
              .round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
