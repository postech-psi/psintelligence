#!/usr/bin/env python
"""Host-only UKF/GRU barometer sensitivity experiment. Run from the project root."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import benchmark_apogee as bench


@dataclass(frozen=True)
class Scenario:
    sigma_m: float
    latency_us: int
    poll_jitter: bool = True
    drop_probability: float = 0.0


# Steady pad statistics from 112.txt; moving-flight conditions are assumptions.
PAD_SIGMA_M = 0.0612
RHO_20MS = 0.83
OBSERVATION_US = 20_000
POLL_US = 10_000
TRAIN_SCENARIOS = {
    'low': Scenario(0.12, 10_000),
    'middle': Scenario(0.35, 20_000),
    'high': Scenario(0.75, 30_000),
}
SCENARIOS = {**TRAIN_SCENARIOS, 'stress': Scenario(1.5, 40_000, drop_probability=0.015)}


def synthesize_events(frame: pd.DataFrame, scenario: Scenario, seed: int, *,
                      burst_start_probability: float = 0.0, burst_samples: int = 2) -> pd.DataFrame:
    """Delays only BMP data; estimator sees one observation at its arrival, or none."""
    if not 0 <= burst_start_probability <= 1 or burst_samples < 2:
        raise ValueError('burst probability must be in [0, 1] and burst length at least two')
    if scenario.latency_us < (POLL_US if scenario.poll_jitter else 0):
        raise ValueError('poll jitter would make observations arrive before acquisition')
    events = []
    for flight_id, group in frame.groupby('flight_id', sort=True):
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(flight_id)]))
        g = group.sort_values('time')
        time = np.rint(g['time'].to_numpy() * 1_000_000).astype(np.int64)
        if np.any(np.diff(time) != OBSERVATION_US):
            raise ValueError('scenario requires the Step 1.5 20 ms source grid')
        n = len(time)
        white = rng.standard_normal(n)
        ar = np.empty(n)
        ar[0] = white[0]
        innovation = np.sqrt(1 - RHO_20MS**2)
        for i in range(1, n):
            ar[i] = RHO_20MS * ar[i-1] + innovation * white[i]
        moving = np.maximum.accumulate(np.abs(g['velocity_z'].to_numpy()) > 1.0)
        altitude = g['altitude_rel'].to_numpy() + ar * np.where(moving, scenario.sigma_m, PAD_SIGMA_M)
        arrival = time + scenario.latency_us
        if scenario.poll_jitter:
            # Bounded polling phase: 10/30 or 20/20 ms, not sensor rate claims.
            arrival[1::2] += rng.integers(-1, 2, size=n // 2) * POLL_US
        keep = (arrival <= time[-1]) & (rng.random(n) >= scenario.drop_probability)
        if burst_start_probability:
            starts = np.flatnonzero(moving & (rng.random(n) < burst_start_probability))
            for start in starts:
                keep[start:start + burst_samples] = False

        # Hold IMU features at BMP-only event times, but don't send the held value
        # as a fresh IMU sample. Preserve unchanged BMP values as new if scheduled.
        timeline = np.union1d(time, arrival[keep])
        i = np.searchsorted(time, timeline, side='right') - 1
        merged = g.iloc[i][['flight_id', 'acceleration_inertial', 'tilt_angle',
                            'gyro_roll_noisy', 'apogee_rel']].reset_index(drop=True)
        merged['time'] = timeline / 1_000_000
        merged['baro_observation'] = np.nan
        merged.loc[np.searchsorted(timeline, arrival[keep]), 'baro_observation'] = altitude[keep]
        merged['new_imu'] = timeline == time[i]
        events.append(merged)
    return pd.concat(events, ignore_index=True)


def _replayed(frame, replay_exe, scenario, seed, **burst):
    if scenario is None:
        return bench._benchmark_inputs(frame, SimpleNamespace(state_source='m7_ukf', ukf_replay=replay_exe))
    events = synthesize_events(frame, scenario, seed, **burst)
    replay_input = events.assign(acceleration_inertial=events['acceleration_inertial'].where(events['new_imu']))
    replay = bench._run_ukf_replay(replay_input, events['baro_observation'], replay_exe)
    return bench._grid_with_m7_state(events, replay)


def _samples(grid, ep, ids, metadata, window=100):
    x, h, v, r, flight_ids = bench._samples_for_window(grid, ep, ids, window)
    rows = pd.concat([ep.loc[(ep.flight_id == i) & ep.window_valid] for i in ids], ignore_index=True)
    np.testing.assert_array_equal(flight_ids, rows.flight_id.to_numpy())
    rows = rows[['flight_id', 'endpoint_time', 'apogee_rel']].join(metadata, on='flight_id', validate='many_to_one')
    rows['remaining_s'] = rows.apogee_time - rows.endpoint_time
    rows['coast_s'] = rows.endpoint_time - rows.burnout_time
    return x, h, v, r, rows


def _compare(models, data, ids, metadata, ideal_rows, split):
    predictions = []
    coverage = []
    for name, (grid, endpoints) in data.items():
        x, h, v, r, rows = _samples(grid, endpoints, ids, metadata)
        coast_key = ['flight_id', 'endpoint_time']
        first = ideal_rows.drop_duplicates('flight_id')[coast_key]
        coverage.append(dict(split=split, scenario=name, windows=len(rows), ideal_windows=len(ideal_rows),
                             flights=rows.flight_id.nunique(),
                             first_windows=len(rows.merge(first, on=coast_key))))
        for model_name, seed, model, scaler in models:
            result = bench._evaluate(model, 'gru', bench._scale_array(scaler, x), h, v, r)
            prediction = result['h_raw']
            if not np.isfinite(prediction).all():
                raise RuntimeError('non-finite estimate')
            predictions.append(rows.assign(split=split, scenario=name, model=model_name, seed=seed,
                                           prediction=prediction, ukf_alt=h, ukf_vz=v))
    return predictions, coverage


def _summarize(predictions):
    result = []
    keys = ['split', 'scenario', 'model', 'seed']
    for key, g in predictions.groupby(keys):
        g = g.sort_values(['flight_id', 'endpoint_time'])
        regions = {'coast': g, 'coast_first_1s': g[g.coast_s < 1],
                   'apex_last_1s': g[g.remaining_s <= 1],
                   'first_available': g.drop_duplicates('flight_id'),
                   'last_available': g.drop_duplicates('flight_id', keep='last')}
        for name, sample in regions.items():
            if sample.empty:
                continue
            err = sample.prediction - sample.apogee_rel
            a = err.abs()
            result.append(dict(zip(keys, key), region=name, n=len(a), flights=sample.flight_id.nunique(),
                               mae=a.mean(), bias=err.mean(), p95=a.quantile(.95), max_error=a.max(),
                               over_5m=(a > 5).sum()))
    return pd.DataFrame(result)


def _load(bundle):
    import joblib
    import torch
    from apogee_models import GRUApogeeModel

    model = GRUApogeeModel(hidden_dim=32, num_layers=1)
    model.load_state_dict(torch.load(bundle / 'model.pth', map_location='cpu', weights_only=True))
    return model.eval(), joblib.load(bundle / 'scaler.pkl')


def main():
    import json
    from sklearn.preprocessing import StandardScaler

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ukf-replay', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--baseline-42', type=Path, required=True)
    parser.add_argument('--baseline-43', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data/simulated/all_trajectories.csv.gz'))
    parser.add_argument('--metadata', type=Path, default=Path('data/simulated/flight_metadata.csv.gz'))
    parser.add_argument('--split', type=Path, default=Path('data/experiments/apogee-v2/split_manifest.json'))
    parser.add_argument('--epochs', type=int, default=120)
    args = parser.parse_args()
    frame = bench._read_data(args.data)
    metadata = pd.read_csv(args.metadata).set_index('flight_id')[['apogee_time', 'burnout_time']]
    split = bench._load_split(args.split, frame.flight_id.unique())
    data_hash = bench.ap.sha256_file(args.data)
    replay_hash = bench.ap.sha256_file(args.ukf_replay)
    # Use one realization per profile and flight; train, validation and test IDs
    # remain disjoint. The fourth profile never participates in the fit.
    profile_seeds = {name: 11200 + i for i, name in enumerate(SCENARIOS)}
    config = dict(data_sha256=data_hash, metadata_sha256=bench.ap.sha256_file(args.metadata),
                  ukf_replay_sha256=replay_hash, split=split, output_head='r_raw',
                  features=list(bench.ap.FEATURE_COLUMNS), window=100, dt=.02, stride=10,
                  hidden_dim=32, num_layers=1, epochs=args.epochs, endpoint_phase='coasting',
                  pad_sigma_m=PAD_SIGMA_M, rho_20ms=RHO_20MS,
                  scenarios={name: asdict(s) for name, s in SCENARIOS.items()},
                  profile_seeds=profile_seeds, train_scenarios=list(TRAIN_SCENARIOS))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = []
    for seed, path in [(42, args.baseline_42), (43, args.baseline_43)]:
        cfg = json.loads((path / 'run.json').read_text(encoding='utf-8'))
        for key, expected in dict(seed=seed, split=split, data_sha256=data_hash,
                                  ukf_replay_sha256=replay_hash, window=100, hidden_dim=32,
                                  num_layers=1, endpoint_phase='coasting', loss='raw', epochs=args.epochs).items():
            if cfg[key] != expected:
                raise ValueError(f'baseline {path}: incompatible {key}')
        base.append(('original', seed, *_load(path)))
    data = {}
    for name, scenario in {'original': None, **SCENARIOS}.items():
        grid, endpoints = _replayed(frame, args.ukf_replay, scenario, profile_seeds.get(name))
        e = bench._select_endpoints(frame, endpoints, 'coasting')
        data[name] = grid, e.loc[e.endpoint_time <= e.flight_id.map(metadata.apogee_time)]

    samples = {part: [_samples(*data[name], split[part], metadata) for name in TRAIN_SCENARIOS]
               for part in ('train', 'validation')}
    scaler = StandardScaler().fit(np.concatenate([s[0][:, -1, :] for s in samples['train']]))
    combined = {part: [np.concatenate([s[i] for s in series]) for i in range(4)]
                for part, series in samples.items()}
    for part in combined:
        combined[part][0] = bench._scale_array(scaler, combined[part][0])
        print(part, len(combined[part][0]), 'windows', flush=True)
    models = list(base)
    for seed in (42, 43):
        model = bench._train_one(*combined['train'], *combined['validation'], kind='gru',
                                 hidden_dim=32, num_layers=1, seed=seed, loss_mode='raw', epochs=args.epochs)
        bundle = args.output_dir / f'mixed_seed{seed}'
        bench._save_run_bundle(bundle, model=model, scaler=scaler, config={**config, 'seed': seed})
        models.append(('mixed', seed, *_load(bundle)))
        print('saved and reloaded', bundle, flush=True)
    rows, coverage = [], []
    for part in ('validation', 'test'):
        ideal = _samples(*data['original'], split[part], metadata)[-1]
        p, c = _compare(models, data, split[part], metadata, ideal, part)
        rows += p
        coverage += c
    predictions = pd.concat(rows, ignore_index=True)
    predictions.to_csv(args.output_dir / 'predictions.csv', index=False)
    summary = _summarize(predictions)
    summary.to_csv(args.output_dir / 'comparison.csv', index=False)
    pd.DataFrame(coverage).to_csv(args.output_dir / 'coverage.csv', index=False)
    bench.ap.write_json(args.output_dir / 'run_config.json', config)
    print(summary.groupby(['split','scenario','model','region']).mae.mean().round(3).unstack().to_string())


if __name__ == '__main__':
    main()
