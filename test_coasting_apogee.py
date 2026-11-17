"""NumPy/pandas contracts; never import or execute torch."""
import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import apogee_pipeline as ap
import coasting_apogee as coast


def fixture_frame():
    # D at the coasting endpoint is 10 + 20**2/(2g).
    frames, metadata = [], []
    for fid, height in enumerate([25.0, 31.0, 5.0]):
        frames.append(pd.DataFrame({
            "flight_id": fid, "time": [0.0, 1.0],
            "altitude": [407.0, 417.0], "altitude_noisy": [407.5, 417.5],
            "velocity_z": [0.0, 20.0], "acceleration_noisy": [9.81, -0.2],
            "tilt_angle": [-2.0, -2.0], "gyro_roll_noisy": [0.0, 0.1],
            "flight_phase": ["powered_ascent", "coasting"],
            "energy_ratio": [123.0, 456.0],
        }))
        metadata.append({"flight_id": fid, "apogee_altitude": 407.0 + height})
    return pd.concat(frames, ignore_index=True), pd.DataFrame(metadata)


def test_shared_features_metadata_agl_label_and_no_mutation(monkeypatch):
    raw, meta = fixture_frame()
    before = raw.copy(deep=True)
    called = []
    original = ap.make_sim_features

    def spy(frame):
        called.append(True)
        return original(frame)

    monkeypatch.setattr(ap, "make_sim_features", spy)
    out = coast.make_coasting_frame(raw, meta)
    assert called == [True]
    pd.testing.assert_frame_equal(raw, before)
    np.testing.assert_array_equal(out.energy_ratio, before.energy_ratio)
    np.testing.assert_allclose(out.apogee_agl, [25, 25, 31, 31, 5, 5])
    np.testing.assert_allclose(out.acceleration_inertial, raw.acceleration_noisy - ap.SIM_G)
    np.testing.assert_allclose(out.r_target.iloc[1::2], np.array([25, 31, 5]) / out.denom.iloc[1::2])
    assert out.input_valid.tolist() == [False, True, False, True, False, True]
    assert out.strict_consistent.iloc[1::2].tolist() == [True, False, False]
    assert out.label_status.iloc[1::2].tolist() == ["consistent", "above_ballistic", "below_current"]


def test_tiny_upper_violation_is_not_clipped():
    raw, meta = fixture_frame()
    denom = 10.0 + 20.0**2 / (2 * ap.SIM_G)
    meta.loc[0, "apogee_altitude"] = 407.0 + denom + 1e-8
    out = coast.make_coasting_frame(raw, meta)
    assert out.loc[1, "r_target"] > 1
    assert out.loc[1, "candidate"]
    assert not out.loc[1, "strict_consistent"]
    assert out.loc[1, "label_status"] == "above_ballistic"


def test_phase_is_required_not_inferred_from_time():
    raw, meta = fixture_frame()
    with pytest.raises(ap.PipelineError, match="flight_phase"):
        coast.make_coasting_frame(raw.drop(columns="flight_phase"), meta)
    raw.loc[1, "flight_phase"] = None
    out = coast.make_coasting_frame(raw, meta)
    assert out.loc[1, "input_rejection"] == "missing_phase"


def test_missing_nonfinite_labels_features_and_metadata_are_audited():
    raw, meta = fixture_frame()
    meta.loc[0, "apogee_altitude"] = np.nan
    meta.loc[1, "apogee_altitude"] = np.inf
    raw.loc[5, "gyro_roll_noisy"] = np.inf
    out = coast.make_coasting_frame(raw, meta)
    assert out.loc[1, "label_status"] == "missing_label"
    assert out.loc[3, "label_status"] == "nonfinite_label"
    assert out.loc[5, "input_rejection"] == "nonfinite_feature"
    assert not out.candidate.any()
    out = coast.make_coasting_frame(raw, meta.iloc[1:])
    assert out.loc[1, "label_status"] == "missing_metadata"
    with pytest.raises(ap.PipelineError, match="duplicate"):
        coast.make_coasting_frame(raw, pd.concat([meta, meta.iloc[:1]]))


@pytest.mark.parametrize("column,value,reason", [
    ("altitude", np.nan, "missing_state"),
    ("altitude", np.inf, "nonfinite_state"),
    ("velocity_z", np.nan, "missing_state"),
    ("velocity_z", np.inf, "nonfinite_state"),
    ("velocity_z", 3.0, "not_ascending"),
    ("altitude", 406.0, "negative_altitude"),
    ("gyro_roll_noisy", np.nan, "missing_feature"),
])
def test_input_rejections(column, value, reason):
    raw, meta = fixture_frame()
    raw.loc[1, column] = value
    out = coast.make_coasting_frame(raw, meta)
    assert out.loc[1, "input_rejection"] == reason
    assert not out.loc[1, "candidate"]


def test_missing_launch_pad_rejects_whole_flight():
    raw, meta = fixture_frame()
    raw.loc[0, "altitude"] = np.nan
    out = coast.make_coasting_frame(raw, meta)
    assert not out.loc[out.flight_id == 0, "input_valid"].any()


def test_split_is_shared_and_test_does_not_select_on_answer():
    raw, meta = fixture_frame()
    frames, metas = [], []
    for fid in range(200):
        # Every flight has strict and non-strict coasting rows.
        part = raw[raw.flight_id == 0].copy()
        part = pd.concat([part, part.iloc[[1]].assign(time=2.0, altitude=427.0)], ignore_index=True)
        part["flight_id"] = fid
        frames.append(part)
        metas.append({"flight_id": fid, "apogee_altitude": 438.0})
    out = coast.make_coasting_frame(pd.concat(frames, ignore_index=True), pd.DataFrame(metas))
    split = ap.build_split_manifest(range(200))
    parts, audit = coast.partition_coasting(out, split)
    assert {k: len(v) for k, v in split.items()} == {
        "train": 120, "validation": 30, "calibration": 25, "test": 25,
    }
    assert parts["train"].strict_consistent.all()
    assert parts["validation"].strict_consistent.all()
    assert len(parts["test"]) == 50
    assert len(parts["test_strict"]) == 25
    assert len(parts["calibration"]) == 50
    test_audit = audit.set_index("split").loc["test"]
    assert test_audit["candidate_rows"] == 50
    assert test_audit["above_ballistic"] == 25
    assert test_audit["selected_rows"] == 50
    assert audit.set_index("split").loc["train", "excluded_inconsistent"] == 120
    with pytest.raises(ap.PipelineError):
        coast.partition_coasting(out, {k: [0] for k in split})


def test_reference_head_grid_and_label_inverse():
    h, v, logit = np.meshgrid([0.0, 1e-8, 10.0, 1e3, 1e8],
                              [3.000001, 20.0, 100.0, 1e4],
                              [-1e300, -1000.0, -5.0, 0.0, 5.0, 1000.0, 1e300])
    r, height = coast.constrained_r_head(logit, h, v)
    denom = h + v**2 / (2 * ap.SIM_G)
    assert np.isfinite(r).all() and np.isfinite(height).all()
    assert np.all(r >= h / denom) and np.all(r <= 1)
    assert np.all(height >= h - 1e-7)
    assert np.all(height <= denom + 1e-7)
    np.testing.assert_allclose(height, denom * r)
    raw, meta = fixture_frame()
    out = coast.make_coasting_frame(raw, meta).query("candidate")
    np.testing.assert_allclose(coast.reconstruct_apogee(out.r_target, out.altitude_rel, out.velocity_z), out.apogee_agl)
    # The old h + r*G inverse is a different target and is not accepted.
    wrong = out.altitude_rel + out.r_target * out.gain
    assert not np.allclose(wrong, out.apogee_agl)


@pytest.mark.parametrize("logit,h,v", [
    (np.nan, 10, 20), (np.inf, 10, 20), (0, np.nan, 20),
    (0, -1, 20), (0, 10, np.inf), (0, 10, 3), (0, 10, -20),
    (0, 0, 0), (0, 1e308, 1e308),
])
def test_reference_head_refuses_invalid_domain(logit, h, v):
    with pytest.raises(ap.PipelineError):
        coast.constrained_r_head(logit, h, v)


@pytest.mark.parametrize("gravity", [0, -1, np.nan, np.inf])
def test_reference_refuses_invalid_gravity(gravity):
    with pytest.raises(ap.PipelineError):
        coast.constrained_r_head(0, 10, 20, gravity=gravity)


def test_metrics_count_full_population_and_flight_weighting():
    raw, meta = fixture_frame()
    out = coast.make_coasting_frame(raw, meta).query("candidate").copy()
    r = out.r_target.to_numpy() + np.array([0.1, -0.1, 0.0])
    metrics = coast.score_predictions(out, r)
    assert metrics["rows"] == 3 and metrics["flights"] == 3
    assert metrics["r_mae"] == pytest.approx(0.2 / 3)
    assert metrics["strict_rows"] == 1
    with pytest.raises(ap.PipelineError):
        coast.score_predictions(out, [np.nan] * 3)
    with pytest.raises(ap.PipelineError):
        coast.score_predictions(out, np.zeros((3, 1)))
    assert coast.score_predictions(out.iloc[:0], np.empty(0))["rows"] == 0


def test_real_gate_refuses_unknown_coast_and_does_not_read_file(tmp_path):
    missing = tmp_path / "not-present.txt"
    out, reason = coast.load_real_coasting(missing, launch_time_s=None, coast_start_s=None)
    assert out is None and "REAL_COAST_START_S" in reason
    out, reason = coast.load_real_coasting(missing, launch_time_s=None, coast_start_s=3.5)
    assert out is None and "REAL_LAUNCH_TIME_S" in reason
    with pytest.raises(ap.PipelineError):
        coast.load_real_coasting(missing, launch_time_s=10, coast_start_s=np.inf)


def test_real_raw_parser_agl_inertial_and_gate(tmp_path):
    path = tmp_path / "real.txt"
    lines = []
    for t, h, v in [(9.0, 100.0, 0.0), (10.0, 100.0, 0.0), (12.0, 115.0, 20.0), (14.0, 140.0, 4.0)]:
        lines.append(f"Time: {t}s Euler: (0, -2, 0) Gyro: (0, 0, 180) Accel: (0, 0, -1) kal: (alt : {h}, velo : {v})")
    path.write_text("\n".join(lines), encoding="utf-8")
    out, reason = coast.load_real_coasting(path, launch_time_s=10.0, coast_start_s=1.5)
    assert out is not None
    assert reason == "configured coast-only diagnostic; no independent apogee reference"
    np.testing.assert_allclose(out.time, [2.0, 4.0])
    np.testing.assert_allclose(out.altitude_rel, [15.0, 40.0])
    np.testing.assert_allclose(out.acceleration_inertial, -ap.LOG_G)
    np.testing.assert_allclose(out.gyro_roll_noisy, np.pi)
    assert out.attrs["observed_max_agl"] == 40.0
    assert "apogee_agl" not in out


def test_label_can_be_represented_by_head_without_redefining_target():
    raw, meta = fixture_frame()
    row = coast.make_coasting_frame(raw, meta).query("strict_consistent").iloc[0]
    floor = row.altitude_rel / row.denom
    # This is only inversion of the internal activation, not a new label.
    probability = (row.r_target - floor) / (1 - floor)
    logit = np.log(probability / (1 - probability))
    ratio, height = coast.constrained_r_head(logit, row.altitude_rel, row.velocity_z)
    assert ratio == pytest.approx(row.r_target)
    assert height == pytest.approx(row.apogee_agl)


def test_nonfinite_launch_pad_is_not_mislabeled_as_missing():
    raw, meta = fixture_frame()
    raw.loc[0, "altitude"] = np.inf
    with np.errstate(invalid="ignore"):
        out = coast.make_coasting_frame(raw, meta)
    assert out.loc[out.flight_id == 0, "input_rejection"].eq("nonfinite_state").all()


def test_default_notebook_executes_without_torch_or_artifacts(monkeypatch, capsys):
    import builtins
    import sys

    nb = json.loads(Path("step3-ANN&PINN.ipynb").read_text(encoding="utf-8"))
    namespace: dict[str, Any] = {"__name__": "coasting_default_contract"}
    original_import = builtins.__import__
    had_torch = "torch" in sys.modules
    output_dir = coast.EXPERIMENT_DIR
    had_artifacts = output_dir.exists()

    def forbid_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("torch import attempted during NumPy-only validation")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_torch)
    for index, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), f"Step3 cell {index}", "exec"), namespace)
    assert ("torch" in sys.modules) == had_torch
    assert output_dir.exists() == had_artifacts
    assert namespace["trained"] is False
    assert namespace["learned_metrics"] == {}
    assert namespace["real_df"] is None
    parts, audit = namespace["parts"], namespace["audit"].set_index("split")
    assert len(parts["test"]) == audit.loc["test", "candidate_rows"]
    assert len(parts["test_strict"]) == audit.loc["test", "strict_rows"]
    assert parts["train"].strict_consistent.all()
    assert parts["validation"].strict_consistent.all()
    train = parts["train"][list(ap.FEATURE_COLUMNS)].to_numpy()
    np.testing.assert_allclose(namespace["feature_scaler"].mean_, train.mean(axis=0))
    assert "REAL_COAST_START_S" in capsys.readouterr().out


def test_notebook_source_contracts_and_cleared_outputs():
    nb = json.loads(Path("step3-ANN&PINN.ipynb").read_text(encoding="utf-8"))
    assert len(nb["cells"]) == 33
    assert len({cell["id"] for cell in nb["cells"]}) == 33
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    ast.parse(code)
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
    for removed in ["torch.maximum", "torch.clamp", "use_coast", "mono_loss", "range_loss", "coast_loss", "true_apogee", "0.7 *", "pinn_no_coast"]:
        assert removed not in code
    assert "copy.deepcopy(model.state_dict())" in code
    assert "model.load_state_dict(best_state)" in code
    assert "REAL_COAST_START_S = None" in code
    assert "data/experiments/step3-coasting" in code
    assert "RUN_TRAINING = False" in code
    assert "r_floor + (1.0 - r_floor) * torch.sigmoid(logit)" in code
    assert "feature_scaler.fit(train_df" in code
    assert 'parts["test"]' in code and 'parts["test_strict"]' in code
