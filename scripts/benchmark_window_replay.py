#!/usr/bin/env python
"""Compare GRU history lengths on the same timed barometer replays."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

import benchmark_baro_replay as baro

bench = baro.bench
WINDOWS = (50, 100, 150)
SEEDS = (42, 43)
STRIDE = 10
KEYS = ['flight_id', 'endpoint_time']

# These profiles are synthesized only for the training flight IDs.
TRAIN_AUGMENTATIONS = {
    'noisy_burst_2': (baro.Scenario(0.95, 40_000), 11300, 0.009, 2),
    'noisy_burst_3': (baro.Scenario(1.25, 50_000), 11301, 0.006, 3),
}


def _with_history_validity(grid, endpoints, window):
    """Preserve the common prediction clock even when a history has a gap."""
    result = endpoints.copy()
    flags = []
    for flight_id, group in grid.groupby('flight_id', sort=False):
        available = group['valid'].rolling(window, min_periods=window).sum().eq(window).to_numpy()
        mask = result['flight_id'].eq(flight_id)
        indices = result.loc[mask, 'endpoint_index'].to_numpy(dtype=int)
        flags.append(pd.Series(available[indices], index=result.index[mask]))
    result['window_valid'] = pd.concat(flags).reindex(result.index)
    return result


def _physical_predictions(h, v):
    return {'ukf_altitude': h, 'positive_v_ballistic': h + np.maximum(v, 0.0) ** 2 / (2 * bench.ap.SIM_G)}


def _train(data, split, metadata, args, config):
    from sklearn.preprocessing import StandardScaler

    # Every model trains and early-stops on the same endpoints and scaler.
    # Shorter windows receive the causal suffix of the longest valid history.
    combined = {}
    for part in ('train', 'validation'):
        names = tuple(baro.TRAIN_SCENARIOS)
        if part == 'train' and args.augment_baro:
            names += tuple(TRAIN_AUGMENTATIONS)
        series = [baro._samples(data[name][0], data[name][1][max(WINDOWS)], split[part], metadata,
                                max(WINDOWS)) for name in names]
        combined[part] = tuple(np.concatenate(parts) for parts in zip(*(s[:4] for s in series)))
        if args.augment_baro:
            print(f'{part} windows: ' + ', '.join(f'{name}={len(s[0])}' for name, s in zip(names, series)),
                  flush=True)
        if part == 'validation' and args.trace_convergence:
            val_apex_mask = np.concatenate([s[-1]['remaining_s'].between(0, 1).to_numpy() for s in series])
    scaler = StandardScaler().fit(combined['train'][0][:, -1, :])
    for window in (args.train_window,) if args.train_window else WINDOWS:
        train, val = [(bench._scale_array(scaler, x[:, -window:, :]), h, v, r)
                      for x, h, v, r in combined.values()]
        print(f'window={window}: train={len(train[0])}, validation={len(val[0])}', flush=True)
        for seed in config['seeds']:
            history = [] if args.trace_convergence else None
            model = bench._train_one(*train, *val, kind='gru', hidden_dim=32, num_layers=1,
                                     seed=seed, loss_mode='raw', epochs=args.epochs,
                                     history=history, val_apex_mask=val_apex_mask if args.trace_convergence else None)
            bundle = args.output_dir / f'w{window}_seed{seed}'
            bench._save_run_bundle(bundle, model=model, scaler=scaler,
                                   config={**config, 'seed': seed, 'window': window})
            if history is not None:
                pd.DataFrame(history).to_csv(bundle / 'epoch_history.csv', index=False)
            print(f'saved {bundle.name}', flush=True)


def _regions(rows):
    return {'coast': rows['remaining_s'].ge(0),
            'coast_first_1s': rows['coast_s'].lt(1),
            'apex_0p5_to_2s': rows['remaining_s'].between(.5, 2),
            'apex_last_1s': rows['remaining_s'].le(1),
            'last_available': ~rows['flight_id'].duplicated(keep='last')}


def _evaluate(data, split, metadata, args, config):
    models = {}
    for window in WINDOWS:
        for seed in SEEDS:
            path = args.output_dir / f'w{window}_seed{seed}'
            saved = json.loads((path / 'run.json').read_text(encoding='utf-8'))
            if saved != {**config, 'seed': seed, 'window': window}:
                raise ValueError(f'incompatible model settings: {path}')
            models[window, seed] = baro._load(path)

    prediction_rows, coverage_rows = [], []
    for part in ('validation', 'test'):
        ids = split[part]
        _, ref = data['original']
        ref_rows = baro._samples(data['original'][0], ref[min(WINDOWS)], ids, metadata, min(WINDOWS))[-1]
        for scenario, (grid, by_window) in data.items():
            common = by_window[max(WINDOWS)]
            for window, endpoints in by_window.items():
                # Each model keeps its own validity for availability. Compare
                # against all potential endpoints, not just ones it could fill.
                rows = baro._samples(grid, endpoints, ids, metadata, window)[-1]
                for region, mask in _regions(rows).items():
                    if region == 'last_available':
                        continue
                    ref_mask = _regions(ref_rows)[region]
                    coverage_rows.append(dict(split=part, scenario=scenario, window=window, region=region,
                                              windows=int(mask.sum()), flights=rows.loc[mask, 'flight_id'].nunique(),
                                              reference_windows=int(ref_mask.sum()),
                                              reference_flights=ref_rows.loc[ref_mask, 'flight_id'].nunique()))

            x, h, v, r, rows = baro._samples(grid, common, ids, metadata, max(WINDOWS))
            rows = rows.assign(split=part, scenario=scenario)
            for name, estimate in _physical_predictions(h, v).items():
                prediction_rows.append(rows.assign(model=name, seed=0, prediction=estimate))
            for (window, seed), (model, scaler) in models.items():
                inputs = bench._scale_array(scaler, x[:, -window:, :])
                result = bench._evaluate(model, 'gru', inputs, h, v, r)
                if not np.isfinite(result['h_raw']).all():
                    raise RuntimeError('non-finite estimate on valid common window')
                prediction_rows.append(rows.assign(model=f'gru_{window}', seed=seed, prediction=result['h_raw']))

    predictions = pd.concat(prediction_rows, ignore_index=True)
    report = []
    keys = ['split', 'scenario', 'model', 'seed']
    for key, rows in predictions.groupby(keys):
        rows = rows.sort_values(KEYS)
        for region, mask in _regions(rows).items():
            errors = (rows.loc[mask, 'prediction'] - rows.loc[mask, 'apogee_rel']).abs()
            if not errors.empty:
                report.append(dict(zip(keys, key), region=region, n=len(errors), mae=errors.mean(),
                                   p95=errors.quantile(.95), max_error=errors.max(), over_5m=int((errors > 5).sum())))
    summary = pd.DataFrame(report)
    predictions.to_csv(args.output_dir / 'matched_predictions.csv', index=False)
    summary.to_csv(args.output_dir / 'matched_comparison.csv', index=False)
    pd.DataFrame(coverage_rows).to_csv(args.output_dir / 'coverage.csv', index=False)
    bench.ap.write_json(args.output_dir / 'run_config.json', config)
    print(summary.groupby(['split', 'scenario', 'model', 'region']).mae.mean().round(3).unstack().to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ukf-replay', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data/simulated/all_trajectories.csv.gz'))
    parser.add_argument('--metadata', type=Path, default=Path('data/simulated/flight_metadata.csv.gz'))
    parser.add_argument('--split', type=Path, default=Path('data/experiments/apogee-v2/split_manifest.json'))
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--trace-convergence', action='store_true',
                        help='record per-epoch losses, learning rate and last-1s MAE; continue past early stopping')
    parser.add_argument('--augment-baro', action='store_true',
                        help='add high noise, delay and burst losses only to training flights')
    parser.add_argument('--seeds', nargs='+', type=int,
                        help='override the default training seeds (requires --train-window)')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--train-window', type=int, choices=WINDOWS, help='train one window length without evaluating')
    mode.add_argument('--evaluate-only', action='store_true', help='reload the six saved bundles and evaluate')
    args = parser.parse_args()
    if (args.trace_convergence or args.augment_baro or args.seeds is not None) and not args.train_window:
        parser.error('--trace-convergence, --augment-baro and --seeds require --train-window')
    seeds = args.seeds if args.seeds is not None else list(SEEDS)
    if len(set(seeds)) != len(seeds) or any(not 0 <= seed < 2**32 for seed in seeds):
        parser.error('--seeds must contain distinct unsigned 32-bit integers')
    frame = bench._read_data(args.data)
    metadata = pd.read_csv(args.metadata).set_index('flight_id')[['apogee_time', 'burnout_time']]
    split = bench._load_split(args.split, frame.flight_id.unique())
    profile_seeds = {name: 11200 + i for i, name in enumerate(baro.SCENARIOS)}
    config = dict(data_sha256=bench.ap.sha256_file(args.data), metadata_sha256=bench.ap.sha256_file(args.metadata),
                  ukf_replay_sha256=bench.ap.sha256_file(args.ukf_replay), split=split,
                  windows=list(WINDOWS), seeds=seeds, features=list(bench.ap.FEATURE_COLUMNS),
                  hidden_dim=32, num_layers=1, epochs=args.epochs, stride=STRIDE, dt=.02,
                  endpoint_phase='coasting', common_window=max(WINDOWS),
                  pad_sigma_m=baro.PAD_SIGMA_M, rho_20ms=baro.RHO_20MS,
                  profile_seeds=profile_seeds, train_scenarios=list(baro.TRAIN_SCENARIOS),
                  scenarios={name: asdict(s) for name, s in baro.SCENARIOS.items()})
    if args.trace_convergence:
        config['trace_convergence'] = True
        config['val_apex_horizon_s'] = 1.0
    if args.augment_baro:
        config['train_only_augmentations'] = {
            name: dict(scenario=asdict(scenario), seed=seed, burst_start_probability=probability, burst_samples=length)
            for name, (scenario, seed, probability, length) in TRAIN_AUGMENTATIONS.items()
        }
    scenarios = baro.TRAIN_SCENARIOS if args.train_window else {'original': None, **baro.SCENARIOS}
    data = {}
    def add_replay(name, source, scenario, seed, **burst):
        grid, _ = baro._replayed(source, args.ukf_replay, scenario, seed, **burst)
        # Index/phase match the existing 100-step/200-ms run. Start at 49
        # rather than 0 to avoid materializing warmup-only endpoints.
        endpoints = bench.ap.make_endpoint_manifest(grid, max_window=min(WINDOWS), stride=STRIDE)
        selected = bench._select_endpoints(source, endpoints, 'coasting')
        selected = selected.loc[np.isfinite(selected.target) &
                                (selected.endpoint_time <= selected.flight_id.map(metadata.apogee_time))]
        data[name] = grid, {w: _with_history_validity(grid, selected, w) for w in WINDOWS}

    for name, scenario in scenarios.items():
        add_replay(name, frame, scenario, profile_seeds.get(name))
    if args.augment_baro:
        train_frame = frame.loc[frame.flight_id.isin(split['train'])]
        for name, (scenario, seed, probability, length) in TRAIN_AUGMENTATIONS.items():
            add_replay(name, train_frame, scenario, seed,
                       burst_start_probability=probability, burst_samples=length)
    if not args.evaluate_only:
        _train(data, split, metadata, args, config)
    if not args.train_window:
        _evaluate(data, split, metadata, args, config)


if __name__ == '__main__':
    main()
