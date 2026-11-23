"""Contract checks for the M7 input / physics-residual experiment (no torch, no UKF replay)."""
import numpy as np
import pandas as pd

import experiment_m7_inputs_residual as ex


def _grid(n=400, dt=0.02, burn_s=1.5):
    t = np.arange(n) * dt
    a = np.where(t < burn_s, 30.0, np.nan)
    h, v = np.zeros(n), np.zeros(n)
    for i in range(1, n):
        acc = 30.0 if t[i] < burn_s else -9.81 - 0.003 * max(v[i - 1], 0) ** 2
        a[i] = acc
        v[i] = v[i - 1] + acc * dt
        h[i] = h[i - 1] + v[i] * dt
    a[0] = 30.0
    return pd.DataFrame({
        "flight_id": 0, "time": t, "valid": True,
        "altitude_noisy_rel": h, "altitude_rel": h, "velocity_z": v,
        "acceleration_inertial": a, "tilt_angle": -3.0, "gyro_roll_noisy": 0.0,
        "dynamic_pressure": 0.6 * v ** 2, "apogee_rel": h.max(),
    })


def test_residual_head_matches_firmware_ratio_range():
    assert ex.residual_height(100.0, 1.0) == 100.0
    assert np.isclose(ex.residual_height(100.0, 1.5), 110.0)
    assert np.isclose(ex.residual_height(100.0, 0.5), 90.0)


def test_physics_columns_are_undefined_before_coast_and_filled_for_features():
    grid = ex.add_derived_columns(_grid())
    burn = grid["acceleration_inertial"] > 0
    assert grid.loc[burn, "h_phys"].isna().all()
    assert (grid.loc[burn, ["coast_flag", "k_feature", "phys_climb"]] == 0).all().all()
    coast = grid["coast_flag"] == 1
    assert coast.any() and grid.loc[coast, "h_phys"].notna().all()
    # Score once the 1.5 s smoother window no longer spans the 40 m/s^2 burnout step,
    # where the generator's Euler step and the smoother's trapezoid rule disagree.
    clean = coast & (grid["v_s"] > 5) & (grid["time"] > 1.5 + 1.5)
    err = (grid.loc[clean, "h_phys"] - grid["altitude_rel"].max()).abs()
    assert clean.any() and err.max() < 0.5, err.max()


def test_common_endpoints_require_physics_and_full_window():
    grid = ex.add_derived_columns(_grid())
    idx = np.arange(ex.WINDOW - 1, len(grid), 10)
    selected = pd.DataFrame({"flight_id": 0, "endpoint_index": idx,
                             "endpoint_time": grid["time"].to_numpy()[idx], "window_valid": True,
                             "apogee_rel": grid["apogee_rel"].iloc[0]})
    common = ex.common_endpoints(grid, selected)
    assert len(common) and len(common) < len(selected)
    assert np.isfinite(grid["h_phys"].to_numpy()[common["endpoint_index"].astype(int)]).all()


def test_every_variant_feature_exists_after_derivation():
    grid = ex.add_derived_columns(_grid())
    for variant in ex.VARIANTS.values():
        missing = [c for c in variant.features + (variant.head_h, variant.head_v) if c not in grid]
        assert not missing, (variant.name, missing)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
