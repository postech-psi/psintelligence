"""Run from the PSIntelligence project root after training both w100 bundles."""
from pathlib import Path
import json
import re
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path('scripts').resolve()))
import benchmark_window_replay as w
from apogee_models import raw_apogee

baro, bench = w.baro, w.bench
ap = bench.ap
OUT = Path(__file__).resolve().parent
OLD = OUT.with_name('apogee-v2-m7-windows')
EXE = Path('../../Avionics/flight-computer/.pio/tools/ukf_replay.exe')
DATA = Path('data/simulated/all_trajectories.csv.gz')
META = Path('data/simulated/flight_metadata.csv.gz')
LOG = Path('data/112.txt')
WINDOW = 100


def load_models():
    models = {}
    shared = None
    for seed in w.SEEDS:
        old = OLD / f'w{WINDOW}_seed{seed}'
        new = OUT / old.name
        a, b = (json.loads((p / 'run.json').read_text(encoding='utf-8')) for p in (old, new))
        assert {k: v for k, v in a.items() if k != 'ukf_replay_sha256'} == {
            k: v for k, v in b.items() if k != 'ukf_replay_sha256'}
        assert a['ukf_replay_sha256'] != b['ukf_replay_sha256']
        for key, path in [('ukf_replay', EXE), ('data', DATA), ('metadata', META)]:
            assert b[key + '_sha256'] == ap.sha256_file(path), path
        assert b['seed'] == seed and b['window'] == WINDOW
        config = {k: v for k, v in b.items() if k != 'seed'}
        assert shared is None or config == shared
        shared = config
        models['old_gru', seed] = baro._load(old)
        models['retrained_gru', seed] = baro._load(new)
    return models, shared


def predict(models, x, h, v):
    ht, vt = (torch.as_tensor(a, dtype=torch.float32) for a in (h, v))
    with torch.inference_mode():
        for (name, seed), (model, scaler) in models.items():
            r = model(torch.from_numpy(bench._scale_array(scaler, x)))[0]
            yield name, seed, raw_apogee(ht, vt, r).numpy()


def simulation(models, config):
    frame = bench._read_data(DATA)
    metadata = pd.read_csv(META).set_index('flight_id')[['apogee_time', 'burnout_time']]
    previous = pd.read_csv(OLD / 'coverage.csv').query('window == @WINDOW')
    historical = pd.read_csv(OLD / 'matched_predictions.csv').query('model == "gru_100"')
    historical = historical[['split', 'scenario', 'seed', 'flight_id', 'endpoint_time', 'prediction']]
    historical['endpoint_tick'] = np.rint(historical.pop('endpoint_time') / config['dt']).astype(int)
    historical = historical.rename(columns={'prediction': 'historical_prediction'})
    prediction_rows, coverage_rows = [], []
    for scenario, profile in {'original': None, **baro.SCENARIOS}.items():
        grid, _ = baro._replayed(frame, EXE, profile, config['profile_seeds'].get(scenario))
        endpoints = ap.make_endpoint_manifest(grid, max_window=min(config['windows']), stride=config['stride'])
        endpoints = bench._select_endpoints(frame, endpoints, config['endpoint_phase'])
        endpoints = endpoints.loc[np.isfinite(endpoints.target) &
                                  (endpoints.endpoint_time <= endpoints.flight_id.map(metadata.apogee_time))]
        endpoints = w._with_history_validity(grid, endpoints, WINDOW)
        for part in ('validation', 'test'):
            x, h, v, _, rows = baro._samples(grid, endpoints, config['split'][part], metadata, WINDOW)
            rows = rows.assign(split=part, scenario=scenario, ukf_alt=h, ukf_vz=v)
            for region, mask in w._regions(rows).items():
                if region != 'last_available':
                    coverage_rows.append(dict(split=part, scenario=scenario, region=region,
                                              new_windows=int(mask.sum()),
                                              new_flights=rows.loc[mask, 'flight_id'].nunique()))
            for name, estimates in w._physical_predictions(h, v).items():
                prediction_rows.append(rows.assign(model=name, seed=0, prediction=estimates))
            for name, seed, estimates in predict(models, x, h, v):
                prediction_rows.append(rows.assign(model=name, seed=seed, prediction=estimates))
    coverage = previous.merge(pd.DataFrame(coverage_rows), on=['split', 'scenario', 'region'],
                              validate='one_to_one').rename(columns={'windows': 'old_windows', 'flights': 'old_flights'})
    predictions = pd.concat(prediction_rows, ignore_index=True)
    assert np.isfinite(predictions.prediction).all()
    predictions['endpoint_tick'] = np.rint(predictions.endpoint_time / config['dt']).astype(int)
    old_matched = predictions.query('model == "retrained_gru"').merge(
        historical, on=['split', 'scenario', 'seed', 'flight_id', 'endpoint_tick'], validate='one_to_one')
    intersection = pd.concat([old_matched, old_matched.assign(model='old_gru_old_ukf',
                                                             prediction=old_matched.historical_prediction)])

    def summarize(batch):
        keys = ['split', 'scenario', 'model', 'seed']
        stats = []
        for group, rows in batch.groupby(keys):
            rows = rows.sort_values(w.KEYS)
            for region, mask in w._regions(rows).items():
                part = rows.loc[mask]
                err = part.prediction - part.apogee_rel
                if len(err):
                    stats.append(dict(zip(keys, group), region=region, n=len(err), flights=part.flight_id.nunique(),
                                      mae=err.abs().mean(), p95=err.abs().quantile(.95), bias=err.mean()))
        return pd.DataFrame(stats)

    predictions.to_csv(OUT / 'simulation_predictions.csv', index=False)
    coverage.to_csv(OUT / 'simulation_coverage.csv', index=False)
    summary = summarize(predictions)
    summary.to_csv(OUT / 'simulation_metrics.csv', index=False)
    summarize(intersection).to_csv(OUT / 'simulation_historical_intersection.csv', index=False)
    print('Held-out simulation, apex last 1s (MAE, seed mean):')
    print(summary.query('split == "test" & region == "apex_last_1s"').pivot_table(
        index='scenario', columns='model', values='mae').round(3).to_string())
    print(coverage.query('split == "test" & region == "apex_last_1s"').to_string(index=False))


LOG_RE = re.compile(
    r'^State:\s*(?P<state>\w+), Time:\s*(?P<time>[\d.]+)s.*?'
    r'Alt:\s*(?P<baro_alt>[-\d.]+)m\s*, P:\s*(?P<pressure>[\d.]+)hPa.*?'
    r'Euler:\s*\([^,]+,\s*(?P<pitch>[-\d.]+),[^)]+\).*?'
    r'Gyro:\s*\([^,]+,\s*[^,]+,\s*(?P<gyro_roll>[-\d.]+)\).*?'
    r'Accel:\s*\([^,]+,\s*[^,]+,\s*(?P<accel_g>[-\d.]+)\).*?'
    r'kal:\s*\(alt\s*:\s*(?P<logged_h>[-\d.naif]+),\s*velo\s*:\s*(?P<logged_v>[-\d.naif]+),'
    r'\s*apo\s*:\s*(?P<logged_quadratic>[-\d.naif]+)')


def real_log(models, config):
    records = [LOG_RE.search(line).groupdict() for line in LOG.read_text(encoding='utf-8').splitlines() if line.strip()]
    assert records
    raw = pd.DataFrame(records)
    for col in raw.columns.difference(['state']):
        raw[col] = raw[col].astype(float)
    clock_us = np.rint(raw['time'].to_numpy() * 1_000_000).astype(np.int64)
    assert np.all(np.diff(clock_us) > 0)
    raw['t_log_s'] = raw['time']
    raw['time'] = (clock_us - clock_us[0]) / 1_000_000
    launch = int(np.flatnonzero(raw['accel_g'].to_numpy() > .5)[0])
    launch_t = raw.at[launch, 'time']
    launch_state_t = raw.loc[raw.state.eq('LAUNCH'), 'time'].iloc[0]
    zero = raw.iloc[:launch].baro_alt.median()
    raw['baro_rel'] = raw.baro_alt - zero
    raw['flight_id'] = 0
    raw['acceleration_inertial'] = raw.accel_g * ap.LOG_G
    # Same pitch and gyro axes as the existing real-log inference demo.
    raw['tilt_angle'] = -raw.pitch.abs()
    raw['gyro_roll_noisy'] = np.deg2rad(raw.gyro_roll)
    raw['apogee_rel'] = np.nan  # the log does not contain a target
    log_end = float(raw.time.iloc[-1])
    hi = float(raw.baro_rel.max())
    status, traces = {}, []
    for mode, offer in {'every_logged_row': raw.baro_rel.notna(),
                        'pressure_change_proxy': raw.pressure.ne(raw.pressure.shift())}.items():
        replay = bench._run_ukf_replay(raw, raw.baro_rel.where(offer), EXE)
        grid, _ = bench._grid_with_m7_state(raw, replay)
        aligned = np.searchsorted(raw.time.to_numpy(), grid.observation_time.to_numpy(), side='right') - 1
        idx = np.arange(len(grid))
        eligible = grid.valid.rolling(WINDOW, min_periods=WINDOW).sum().eq(WINDOW)
        h, v = grid.altitude_rel.to_numpy(), grid.velocity_z.to_numpy()
        positive_denom = h + v**2 / (2 * ap.SIM_G) > 1e-9
        report_clock = (idx >= min(config['windows']) - 1) & ((idx - min(config['windows']) + 1) % config['stride'] == 0)
        windows = np.flatnonzero(eligible & positive_denom & report_clock)
        assert (windows >= WINDOW - 1).all()
        for i in windows:
            assert grid.valid.iloc[i - WINDOW + 1:i + 1].all()
        trace = grid.assign(mode=mode, report_clock=report_clock, window_valid=eligible,
                            usable_prediction=eligible & positive_denom & report_clock,
                            t_since_accel_trigger=grid.time - launch_t, t_to_log_end=log_end - grid.time,
                            baro_logged_rel=raw.baro_rel.to_numpy()[aligned],
                            logged_v=raw.logged_v.to_numpy()[aligned],
                            logged_quadratic_rel=raw.logged_quadratic.to_numpy()[aligned] - zero)
        if len(windows):
            x = ap.slice_history(grid[list(ap.FEATURE_COLUMNS)].to_numpy(dtype=np.float32), windows, WINDOW)
            for name, seed, estimates in predict(models, x, h[windows], v[windows]):
                assert np.isfinite(estimates).all()
                trace.loc[windows, f'{name}_{seed}'] = estimates
        flight = trace.loc[trace.time >= launch_t]
        last = flight.loc[flight.t_to_log_end <= 1.0]
        predictions = {}
        for seed in w.SEEDS:
            for name in ('old_gru', 'retrained_gru'):
                col = f'{name}_{seed}'
                tail = last.loc[last.usable_prediction]
                if len(tail):
                    a = tail[col].to_numpy()
                    predictions[col] = dict(last=float(a[-1]), min=float(a.min()), max=float(a.max()),
                                            below_logged_max=int((a < hi).sum()),
                                            below_current_ukf=int((a < tail.altitude_rel.to_numpy()).sum()))
        status[mode] = dict(offered_bmp_rows=int(offer.sum()), valid_ukf_rows=int(replay.valid.sum()),
                            flight_predicted=int(flight.usable_prediction.sum()),
                            flight_scheduled=int(flight.report_clock.sum()),
                            last_1s_predicted=int(last.usable_prediction.sum()),
                            last_1s_scheduled=int(last.report_clock.sum()), predictions=predictions,
                            last_prediction_s_before_end=(float(last.loc[last.usable_prediction, 't_to_log_end'].iloc[-1])
                                                          if last.usable_prediction.any() else None),
                            final_kf_alt=float(replay.kf_alt.iloc[-1]),
                            final_kf_vz=float(replay.kf_vz.iloc[-1]) if replay.valid.iloc[-1] else None)
        traces.append(trace)
        pd.concat([raw, replay.add_prefix('replay_'), offer.rename('offered_bmp')], axis=1).to_csv(
            OUT / f'112_{mode}_observations.csv', index=False)
    summary = dict(log_sha256=ap.sha256_file(LOG), rows=len(raw), pad_zero_m=float(zero),
                   first_time_s=float(raw.t_log_s.iloc[0]), last_time_s=float(raw.t_log_s.iloc[-1]),
                   accel_trigger_time_s=float(raw.t_log_s.iloc[launch]),
                   launch_state_time_s=float(raw.t_log_s.iloc[0] + launch_state_t),
                   last_observed_baro_rel_m=float(raw.baro_rel.iloc[-1]), max_observed_baro_rel_m=hi,
                   last_logged_kf_velocity_mps=float(raw.logged_v.iloc[-1]),
                   apogee_observed=False, modes=status)
    ap.write_json(OUT / '112_summary.json', summary)
    pd.concat(traces, ignore_index=True).to_csv(OUT / '112_validation_trace.csv', index=False)
    print('112.txt (AGL; no apogee label):\n' + json.dumps(summary, indent=2))


if __name__ == '__main__':
    torch.set_num_threads(2)
    models, config = load_models()
    simulation(models, config)
    real_log(models, config)
