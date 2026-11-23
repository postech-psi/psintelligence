"""Contract tests for apogee_physics.py.

Runs without pytest (``python test_apogee_physics.py``) and is also
pytest-collectable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import apogee_physics as phy

G = 9.81


def _coast_flight(k: float = 0.003, v0: float = 50.0, h0: float = 60.0,
                  burn_s: float = 1.0, dt: float = 0.002, bias: float = 0.0):
    """Burn at +20 m/s^2 for burn_s, then exact quadratic-drag coast."""
    t, h, v, a = [0.0], [0.0], [0.0], [20.0]
    while t[-1] < burn_s:
        t.append(t[-1] + dt); v.append(v[-1] + 20.0 * dt); h.append(h[-1] + v[-1] * dt); a.append(20.0)
    # Re-anchor to the requested coast start state for an analytic apex.
    h[-1], v[-1] = h0, v0
    while v[-1] > 0:
        acc = -G - k * v[-1] ** 2
        t.append(t[-1] + dt)
        v.append(v[-1] + acc * dt)
        h.append(h[-1] + v[-1] * dt)
        a.append(acc)
    frame = pd.DataFrame({"flight_id": 0, "time": t, "altitude_rel": h,
                          "velocity_z": v, "acceleration_inertial": np.asarray(a) + bias})
    return frame.iloc[::10].reset_index(drop=True)  # 50 Hz like the dataset


def test_zero_drag_reduces_to_energy_height():
    h, v = np.array([10.0, 50.0]), np.array([30.0, 5.0])
    out = phy.ballistic_apogee(h, v, np.zeros(2))
    assert np.allclose(out, h + v**2 / (2 * G))


def test_drag_lowers_apex_monotonically_and_past_apex_returns_h():
    h, v = np.full(3, 100.0), np.full(3, 40.0)
    out = phy.ballistic_apogee(h, v, np.array([0.0, 0.001, 0.005]))
    assert out[0] > out[1] > out[2] > 100.0
    assert phy.ballistic_apogee(5.0, -3.0, 0.002) == 5.0


def test_closed_form_matches_numerical_integration():
    k, h0, v0 = 0.003, 60.0, 50.0
    h, v, dt = h0, v0, 1e-5
    while v > 0:
        v += (-G - k * v * v) * dt
        h += v * dt
    assert abs(phy.ballistic_apogee(h0, v0, k) - h) < 0.01


def test_negative_drag_is_rejected():
    try:
        phy.ballistic_apogee(1.0, 1.0, -1e-3)
    except phy.PhysicsBaselineError:
        return
    raise AssertionError("negative k accepted")


def test_fit_recovers_drag_constant_and_ignores_slow_samples():
    v = np.linspace(40, 1, 200)
    a = -G - 0.004 * v**2
    a[v < 5] = 50.0  # garbage below v_min must not matter
    assert abs(phy.fit_drag_constant(a, v) - 0.004) < 1e-12
    assert np.isnan(phy.fit_drag_constant(a[v < 5], v[v < 5]))


def test_online_baseline_is_nan_during_burn_and_accurate_in_coast():
    frame = _coast_flight()
    true_apex = frame["altitude_rel"].max()
    out = phy.online_ballistic_apogee(frame)
    burn = frame["acceleration_inertial"] > 0
    assert out.loc[burn, "apogee_ballistic"].isna().all()
    assert not out.loc[burn, "coast"].any()
    coast = out["apogee_ballistic"].notna()
    assert coast.sum() > 50
    err = (out.loc[coast, "apogee_ballistic"] - true_apex).abs()
    assert err.max() < 0.5, err.max()
    assert np.allclose(out.loc[coast, "k_est"], 0.003, rtol=0.02)


def test_online_baseline_is_causal():
    frame = _coast_flight()
    full = phy.online_ballistic_apogee(frame)
    cut = len(frame) // 2
    head = phy.online_ballistic_apogee(frame.iloc[:cut])
    pd.testing.assert_frame_equal(full.iloc[:cut], head)


def test_settle_delays_first_output():
    frame = _coast_flight()
    early = phy.online_ballistic_apogee(frame, config=phy.CoastConfig(settle_s=0.0))
    late = phy.online_ballistic_apogee(frame, config=phy.CoastConfig(settle_s=0.4))
    t_early = early.loc[early["coast"], "time"].min()
    t_late = late.loc[late["coast"], "time"].min()
    # Settle counts from onset; the default 5-sample confirm already elapsed 4 steps.
    dt = float(frame["time"].diff().median())
    assert abs((t_late - t_early) - (0.4 - 4 * dt)) < 0.03


def test_per_flight_state_does_not_leak():
    a, b = _coast_flight(k=0.002), _coast_flight(k=0.006)
    b["flight_id"] = 1
    both = pd.concat([a, b], ignore_index=True)
    out = phy.online_ballistic_apogee(both)
    k1 = out.loc[(out.flight_id == 0) & out.coast, "k_est"].median()
    k2 = out.loc[(out.flight_id == 1) & out.coast, "k_est"].median()
    assert abs(k1 - 0.002) < 1e-4 and abs(k2 - 0.006) < 2e-4


def test_reference_extrapolates_a_log_cut_before_apogee():
    frame = _coast_flight()
    true_apex = frame["altitude_rel"].max()
    cut = frame[frame["velocity_z"] > 8.0]  # log stops while still climbing
    ref = phy.apogee_reference(cut["time"], cut["altitude_rel"], cut["velocity_z"],
                               cut["acceleration_inertial"])
    assert not ref.apogee_observed
    assert ref.observed_max < true_apex - 2.0
    assert abs(ref.best_estimate - true_apex) < 0.3


def test_reference_uses_observed_max_when_apex_is_logged():
    frame = _coast_flight()
    ref = phy.apogee_reference(frame["time"], frame["altitude_rel"], frame["velocity_z"],
                               frame["acceleration_inertial"])
    assert ref.apogee_observed
    assert ref.best_estimate == frame["altitude_rel"].max()


def test_smoother_recovers_velocity_from_noisy_baro_and_is_causal():
    frame = _coast_flight()
    rng = np.random.default_rng(0)
    frame["altitude_noisy_rel"] = frame["altitude_rel"] + rng.normal(0, 1.5, len(frame))
    out = phy.kinematic_smoother(frame, window_s=1.5)
    # The synthetic flight re-anchors h/v at burnout (t=1 s); score windows after it.
    late = frame["time"] > 1.0 + 1.6
    v_err = (out.loc[late, "v_smooth"] - frame.loc[late, "velocity_z"]).abs()
    h_err = (out.loc[late, "h_smooth"] - frame.loc[late, "altitude_rel"]).abs()
    assert v_err.mean() < 0.6, v_err.mean()
    assert h_err.mean() < 0.6, h_err.mean()
    cut = len(frame) // 2
    head = phy.kinematic_smoother(frame.iloc[:cut], window_s=1.5)
    pd.testing.assert_frame_equal(out.iloc[:cut], head)


def test_smoother_is_exact_without_noise():
    frame = _coast_flight()
    frame["altitude_noisy_rel"] = frame["altitude_rel"]
    out = phy.kinematic_smoother(frame, window_s=0.5)
    coast = frame["time"] > 1.0 + 0.6
    assert (out.loc[coast, "v_smooth"] - frame.loc[coast, "velocity_z"]).abs().max() < 0.05


def test_invalid_config_is_rejected():
    for bad in (phy.CoastConfig(confirm_samples=0), phy.CoastConfig(settle_s=-1.0),
                phy.CoastConfig(fit_window_s=0.0)):
        try:
            bad.validate()
        except phy.PhysicsBaselineError:
            continue
        raise AssertionError(f"accepted {bad}")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
