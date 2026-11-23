"""Causal acquisition/arrival checks for the host sensor experiment."""
import numpy as np
import pandas as pd

from benchmark_baro_replay import Scenario, synthesize_events


def _flight():
    t = np.arange(9) * 0.02
    return pd.DataFrame(dict(flight_id=1, time=t, altitude_rel=100 * t,
                             velocity_z=100.0, acceleration_inertial=100 * t,
                             tilt_angle=0.0, gyro_roll_noisy=0.0, apogee_rel=200.0))


def test_barometer_is_observed_once_at_delayed_arrival():
    frame = _flight()
    events = synthesize_events(frame, Scenario(0.0, 10_000, poll_jitter=False), 42)
    imu, baro = events[events.new_imu], events[events.baro_observation.notna()]
    np.testing.assert_allclose(imu.time, frame.time)
    assert imu.baro_observation.isna().all()
    np.testing.assert_allclose(baro.time, frame.time.iloc[:-1] + 0.01)
    np.testing.assert_allclose(baro.baro_observation, frame.altitude_rel.iloc[:-1])
    np.testing.assert_allclose(baro.acceleration_inertial, frame.acceleration_inertial.iloc[:-1])


def test_dropped_barometer_does_not_turn_held_altitude_into_measurements():
    events = synthesize_events(_flight(), Scenario(0.5, 20_000, drop_probability=1.0), 42)
    assert events.baro_observation.isna().all()
    np.testing.assert_allclose(events.time, _flight().time)


if __name__ == '__main__':
    test_barometer_is_observed_once_at_delayed_arrival()
    test_dropped_barometer_does_not_turn_held_altitude_into_measurements()
    print('barometer replay: causal arrival and missing observation checks passed')
