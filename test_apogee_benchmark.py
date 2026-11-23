"""Non-torch tests for the benchmark runner's data path."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

import apogee_pipeline as ap
from test_apogee_pipeline import _sim_frame

_spec = importlib.util.spec_from_file_location(
    "benchmark_apogee", Path(__file__).with_name("scripts") / "benchmark_apogee.py"
)
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


def _fixture():
    frames = []
    for flight_id in (0, 1):
        frame = _sim_frame(140)
        frame["flight_id"] = flight_id
        frames.append(frame)
    frame = ap.make_sim_features(pd.concat(frames, ignore_index=True))
    frame = ap.add_relative_labels(frame)
    sampled = ap.resample_causal(frame)
    endpoints = ap.make_endpoint_manifest(sampled)
    return sampled, endpoints


def test_short_windows_share_common_endpoint_suffixes():
    sampled, endpoints = _fixture()
    x25, h25, v25, y25, f25 = bench._samples_for_window(sampled, endpoints, [0, 1], 25)
    x100, h100, v100, y100, f100 = bench._samples_for_window(sampled, endpoints, [0, 1], 100)
    assert len(x25) == len(x100)
    np.testing.assert_array_equal(x25, x100[:, -25:, :])
    np.testing.assert_array_equal(h25, h100)
    np.testing.assert_array_equal(y25, y100)
    np.testing.assert_array_equal(f25, f100)


def test_flight_mean_does_not_weight_long_flights_more():
    errors = np.array([100.0, 100.0, 1.0])
    fids = np.array([0, 0, 1])
    assert bench._flight_mae(errors, fids) == 50.5


def test_endpoint_manifest_keeps_common_grid_for_all_windows():
    _, endpoints = _fixture()
    assert endpoints["endpoint_index"].iloc[0] == 99
    assert len(endpoints) == len(endpoints.drop_duplicates(["flight_id", "endpoint_index"]))


def test_target_height_is_numpy():
    h = np.array([100.0, 120.0])
    v = np.array([10.0, 5.0])
    y = np.array([1.0, 0.9])
    target = bench._target_height(h, v, y)
    assert isinstance(target, np.ndarray)
    np.testing.assert_allclose(target, (h + v**2 / (2 * 9.81)) * y)


def test_invalid_common_history_excludes_endpoint_for_all_lengths():
    sampled, _ = _fixture()
    for fid in (0, 1):
        idx = sampled.index[sampled["flight_id"] == fid][50]
        sampled.loc[idx, "valid"] = False
    endpoints = ap.make_endpoint_manifest(sampled)
    counts = []
    for window in (25, 50, 100):
        x, *_ = bench._samples_for_window(sampled, endpoints, [0, 1], window)
        counts.append(len(x))
    assert counts == [0, 0, 0]


def _m7_fixture(n=230):
    frame = ap.add_relative_labels(ap.make_sim_features(_sim_frame(n)))
    step = np.arange(n)
    replay = pd.DataFrame({
        "time_us": step * 20_000,
        "baro_alt": 0.25 * step,
        "kf_alt": 0.5 * step,
        "kf_vz": 10.0 + 0.1 * step,
        "valid": 1,
    })
    return frame, replay


def test_m7_head_and_label_use_replayed_state_but_feature_keeps_barometer():
    frame, replay = _m7_fixture()
    features, endpoints = bench._grid_with_m7_state(frame, replay)
    x, h, v, r, _ = bench._samples_for_window(features, endpoints, [0], 100)
    idx = int(endpoints.iloc[0]["endpoint_index"])
    baro, altitude, velocity = replay.iloc[idx][["baro_alt", "kf_alt", "kf_vz"]]
    q = 0.5 * 1.225 * np.exp(-0.00012 * abs(altitude)) * velocity ** 2
    np.testing.assert_allclose(x[0, -1, [0, 1, 5]], [baro, velocity, q])
    np.testing.assert_allclose([h[0], v[0]], [altitude, velocity])
    np.testing.assert_allclose(bench._target_height(h, v, r), frame["apogee_rel"].iloc[0])


def test_m7_invalid_state_breaks_history_without_compressing_time():
    frame, replay = _m7_fixture()
    replay.loc[60, "valid"] = 0  # a held finite value is still invalid
    features, endpoints = bench._grid_with_m7_state(frame, replay)
    assert not features.iloc[60]["valid"]
    valid = endpoints.loc[endpoints["window_valid"]]
    assert valid.iloc[0]["endpoint_index"] == 169
    _, h, _, _, _ = bench._samples_for_window(features, endpoints, [0], 25)
    assert h[0] == replay.iloc[169]["kf_alt"]


def test_coasting_filter_preserves_powered_history_and_does_not_look_ahead():
    frame = ap.add_relative_labels(ap.make_sim_features(_sim_frame(150)))
    frame["flight_phase"] = np.where(frame["time"] < 2.0, "powered_ascent", "coasting")
    # The event observation at 2.195 s cannot retroactively qualify 2.18 s.
    frame.loc[100:109, "flight_phase"] = "powered_ascent"
    frame.loc[110, "time"] = 2.195
    features, endpoints = bench._resample_and_endpoints(frame)
    selected = bench._select_endpoints(frame, endpoints, "coasting")
    assert selected["endpoint_index"].tolist() == [119, 129, 139, 149]
    x, *_ = bench._samples_for_window(features, selected, [0], 100)
    np.testing.assert_allclose(x[0, 0], features.loc[20, list(ap.FEATURE_COLUMNS)].to_numpy(dtype=float))


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    print(f"apogee_benchmark 테스트 {len(tests)}건 실행")
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print("\n전체 통과 ✅")
