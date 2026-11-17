"""Input-domain guard tests. Run: python test_inference_guard.py"""
import pandas as pd

from inference_guard import check_feature_domain, format_domain_violations


FEATURES = ["altitude_noisy", "velocity_z"]
MIN = {"altitude_noisy": 402.0, "velocity_z": -0.1}
MAX = {"altitude_noisy": 667.5, "velocity_z": 77.7}


def test_in_range_has_no_violation():
    frame = pd.DataFrame({"altitude_noisy": [402.0, 500.0], "velocity_z": [0.0, 50.0]})
    assert check_feature_domain(frame, FEATURES, MIN, MAX) == []


def test_out_of_range_reports_observed_and_allowed_ranges():
    frame = pd.DataFrame({"altitude_noisy": [1.007, 186.464], "velocity_z": [0.2, 50.5]})
    violations = check_feature_domain(frame, FEATURES, MIN, MAX)
    assert len(violations) == 1
    violation = violations[0]
    assert violation.feature == "altitude_noisy"
    assert violation.outside_rows == 2
    assert violation.observed_min == 1.007
    assert violation.observed_max == 186.464
    assert "OUT OF TRAINING RANGE" in format_domain_violations(violations)


def test_mismatched_range_metadata_is_rejected():
    frame = pd.DataFrame({"altitude_noisy": [500.0], "velocity_z": [10.0]})
    try:
        check_feature_domain(frame, FEATURES, {"altitude_noisy": 402.0}, MAX)
    except ValueError:
        return
    raise AssertionError("mismatched range metadata must raise ValueError")


if __name__ == "__main__":
    tests = [
        test_in_range_has_no_violation,
        test_out_of_range_reports_observed_and_allowed_ranges,
        test_mismatched_range_metadata_is_rejected,
    ]
    print(f"Domain guard 테스트 {len(tests)}건 실행")
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print("\n전체 통과 ✅")
