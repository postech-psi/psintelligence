"""Contract tests for apogee_pipeline.py.

Runs without pytest (``python test_apogee_pipeline.py``) and is also
pytest-collectable.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import apogee_pipeline as ap


def _sim_frame(n: int = 140) -> pd.DataFrame:
    t = np.arange(n) * 0.02
    altitude = 407.0 + 0.5 * 9.81 * t**2
    return pd.DataFrame({
        "flight_id": 0,
        "time": t,
        "altitude": altitude,
        "altitude_noisy": altitude + 0.1,
        "velocity_z": 9.81 * t,
        "acceleration_noisy": 9.81 + 0.2,
        "tilt_angle": -2.0,
        "gyro_roll_noisy": 0.0,
    })


def test_sim_acceleration_removes_gravity_without_mutating_source():
    frame = _sim_frame()
    out = ap.make_sim_features(frame)
    assert np.allclose(out["acceleration_inertial"], 0.2, atol=1e-12)
    assert np.allclose(frame["acceleration_noisy"], 9.81 + 0.2, atol=1e-12)
    assert out["altitude_rel"].iloc[0] == 0.0


def test_real_acceleration_does_not_subtract_gravity_twice():
    frame = pd.DataFrame({
        "flight_id": 0,
        "time": [0.0, 0.02, 0.04],
        "altitude": [1.0, 1.1, 1.3],
        "altitude_noisy": [1.0, 1.1, 1.3],
        "velocity_z": [0.0, 0.1, 0.2],
        "accel_g": [0.0, 0.5, -0.5],
    })
    out = ap.make_real_features(frame)
    assert np.allclose(out["acceleration_inertial"], [0.0, 4.903325, -4.903325], atol=1e-12)
    assert np.allclose(out["altitude_rel"], [0.0, 0.1, 0.3], atol=1e-12)


def test_relative_labels_are_finite_at_window_endpoints_only():
    frame = ap.make_sim_features(_sim_frame())
    frame = ap.add_relative_labels(frame)
    assert np.isnan(frame["energy_ratio_rel"].iloc[0])
    endpoint = int(np.argmax(frame["altitude_rel"]))
    assert np.isfinite(frame["energy_ratio_rel"].iloc[endpoint])
    assert 0.0 < frame["energy_ratio_rel"].iloc[endpoint] <= 1.5


def test_causal_resample_never_reads_future_observations():
    frame = ap.add_relative_labels(ap.make_sim_features(_sim_frame(30)))
    original = ap.resample_causal(frame, dt=0.02, max_age_s=0.04)
    changed = frame.copy()
    changed.loc[21:, "velocity_z"] = 999.0
    changed_resampled = ap.resample_causal(changed, dt=0.02, max_age_s=0.04)
    np.testing.assert_array_equal(
        original.loc[original["time"] <= 0.40, "velocity_z"].to_numpy(),
        changed_resampled.loc[changed_resampled["time"] <= 0.40, "velocity_z"].to_numpy(),
    )
    assert not np.array_equal(original["velocity_z"], changed_resampled["velocity_z"])


def test_resample_marks_stale_grid_rows_invalid():
    frame = ap.add_relative_labels(ap.make_sim_features(_sim_frame(10)))
    frame.loc[5:, "time"] += 0.5  # 0.5 s gap after row 4
    out = ap.resample_causal(frame, dt=0.02, max_age_s=0.04)
    stale = out[(out["time"] >= 0.14) & (out["time"] < 0.60)]
    assert not stale.empty and not stale["valid"].any()


def test_endpoint_and_history_lengths_only_change_the_past():
    x = np.arange(140 * 6, dtype=float).reshape(140, 6)
    endpoints = np.array([99, 109, 119, 129, 139])
    x100 = ap.slice_history(x, endpoints, 100)
    x50 = ap.slice_history(x, endpoints, 50)
    x25 = ap.slice_history(x, endpoints, 25)
    np.testing.assert_array_equal(x50, x100[:, -50:, :])
    np.testing.assert_array_equal(x25, x100[:, -25:, :])


def test_endpoint_manifest_uses_common_max_window_grid():
    frame = ap.make_sim_features(_sim_frame())
    frame = ap.add_relative_labels(frame)
    sampled = ap.resample_causal(frame)
    manifest = ap.make_endpoint_manifest(sampled)
    assert manifest["endpoint_index"].iloc[0] == 99
    assert manifest["endpoint_index"].diff().dropna().eq(10).all()
    assert manifest["endpoint_time"].is_monotonic_increasing


def test_split_is_disjoint_and_covers_all_flights():
    manifest = ap.build_split_manifest(range(200))
    ap.validate_split_manifest(manifest, range(200))
    assert {k: len(v) for k, v in manifest.items()} == {
        "train": 120, "validation": 30, "calibration": 25, "test": 25,
    }


def test_baseline_manifest_rejects_missing_schema_key():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data = root / "data.bin"
        scaler = root / "scaler.bin"
        model = root / "model.bin"
        for path, payload in ((data, b"d"), (scaler, b"s"), (model, b"m")):
            path.write_bytes(payload)
        split = ap.build_split_manifest(range(200))
        manifest = ap.build_baseline_manifest(
            commit="abc", data_path=data, scaler_path=scaler, model_path=model,
            split=split, dt=0.02, window_size=100, stride=10, seed=42,
        )
        ap.validate_baseline_manifest(manifest)
        broken = dict(manifest)
        broken.pop("environment")
        try:
            ap.validate_baseline_manifest(broken)
        except ap.PipelineError:
            pass
        else:
            raise AssertionError("missing manifest key must fail")


def test_invalid_feature_rows_are_rejected():
    frame = ap.make_sim_features(_sim_frame())
    frame.loc[10, "velocity_z"] = np.nan
    try:
        ap.validate_features(frame)
    except ap.PipelineError:
        pass
    else:
        raise AssertionError("NaN feature must fail")


def test_legacy_scaler_matches_shifted_inertial_scaler():
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    legacy = np.column_stack([
        rng.normal(80, 30, 100),
        rng.normal(20, 10, 100),
        rng.normal(10, 20, 100),  # specific force
        rng.normal(-20, 10, 100),
        rng.normal(0, 0.2, 100),
        rng.normal(500, 300, 100),
    ])
    old_scaler = StandardScaler().fit(legacy)
    inertial = legacy.copy()
    inertial[:, 2] -= ap.SIM_G
    new_scaler = StandardScaler().fit(inertial)
    adapted = ap.adapt_legacy_scaler_to_inertial(old_scaler)
    np.testing.assert_allclose(
        adapted.transform(inertial), new_scaler.transform(inertial), atol=1e-12
    )


def test_causal_state_estimate_is_finite_and_causal():
    frame = ap.add_relative_labels(ap.make_sim_features(_sim_frame(30)))
    state = ap.causal_state_estimate(frame)
    assert len(state) == len(frame)
    assert np.isfinite(state[["h_est", "v_est"]].to_numpy()).all()
    changed = frame.copy()
    changed.loc[21:, "acceleration_inertial"] = 100.0
    changed_state = ap.causal_state_estimate(changed)
    np.testing.assert_array_equal(
        state.loc[state["time"] <= 0.40, "h_est"].to_numpy(),
        changed_state.loc[changed_state["time"] <= 0.40, "h_est"].to_numpy(),
    )


def test_baseline_artifact_hash_validation_detects_mutation():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data, scaler, model, last = (root / n for n in ("d.bin", "s.bin", "m.bin", "l.bin"))
        for p, b in ((data, b"d"), (scaler, b"s"), (model, b"m"), (last, b"l")):
            p.write_bytes(b)
        split = ap.build_split_manifest(range(200))
        manifest = ap.build_baseline_manifest(
            commit="abc", data_path=data, scaler_path=scaler, model_path=model,
            last_model_path=last, split=split, dt=0.02, window_size=100, stride=10, seed=42,
        )
        ap.validate_artifact_hashes(manifest, root=root)
        model.write_bytes(b"changed")
        try:
            ap.validate_artifact_hashes(manifest, root=root)
        except ap.PipelineError:
            pass
        else:
            raise AssertionError("mutated model must fail hash validation")


def test_scaler_cannot_be_adapted_twice():
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(1)
    x = rng.normal(size=(30, 6))
    scaler = StandardScaler().fit(x)
    once = ap.adapt_legacy_scaler_to_inertial(scaler)
    assert getattr(once, "_psintelligence_accel_encoding") == "inertial"
    try:
        ap.adapt_legacy_scaler_to_inertial(once)
    except ap.PipelineError:
        pass
    else:
        raise AssertionError("double adaptation must fail")


def test_encoding_resolution_prefers_explicit_metadata():
    assert ap.resolve_acceleration_encoding({"acceleration_encoding": "inertial"}) == "inertial"
    assert ap.resolve_acceleration_encoding({"artifact_acceleration_encoding": "legacy_specific_force"}) == "legacy_specific_force"
    assert ap.resolve_acceleration_encoding({"feature_cols": ["acceleration_inertial"]}) == "legacy_specific_force"


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    print(f"apogee_pipeline 테스트 {len(tests)}건 실행")
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print("\n전체 통과 ✅")
