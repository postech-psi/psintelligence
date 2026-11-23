"""Contract checks for same-clock window comparison."""
import numpy as np
import pandas as pd

from benchmark_window_replay import _physical_predictions, _with_history_validity


def test_gap_recovers_without_shifting_endpoints():
    grid = pd.DataFrame({'flight_id': np.repeat([7, 8], 200), 'valid': True})
    grid.loc[75, 'valid'] = False
    endpoints = pd.DataFrame({'flight_id': [7, 7, 7, 8, 8, 8],
                              'endpoint_index': [99, 129, 179] * 2,
                              'endpoint_time': [1.98, 2.58, 3.58] * 2})
    for window, expected in [(50, [False, True, True, True, True, True]),
                             (100, [False, False, True, True, True, True]),
                             (150, [False, False, False, False, False, True])]:
        manifest = _with_history_validity(grid, endpoints, window)
        pd.testing.assert_frame_equal(manifest.drop(columns='window_valid'), endpoints)
        assert manifest['window_valid'].tolist() == expected


def test_only_upward_motion_adds_remaining_height():
    heights = np.array([40., 60., 80.])
    estimates = _physical_predictions(heights, np.array([-9.81, 0., 9.81]))
    np.testing.assert_array_equal(estimates['ukf_altitude'], heights)
    np.testing.assert_allclose(estimates['positive_v_ballistic'], [40., 60., 84.905])


if __name__ == '__main__':
    test_gap_recovers_without_shifting_endpoints()
    test_only_upward_motion_adds_remaining_height()
    print('window replay: gap recovery, same endpoints, and positive-only baseline passed')
