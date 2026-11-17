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


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    print(f"apogee_benchmark 테스트 {len(tests)}건 실행")
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print("\n전체 통과 ✅")
