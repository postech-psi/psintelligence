"""Compare augmented and unaugmented 240-epoch runs. Run from the project root."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path('scripts').resolve()))
import benchmark_window_replay as w

baro, bench = w.baro, w.bench
ap = bench.ap
OUT = Path(__file__).resolve().parent
BASELINE = OUT.with_name('apogee-v2-m7-ukf-convergence')
LOG_REPLAY = OUT.with_name('apogee-v2-m7-ukf-gap')
EXE = Path('../../Avionics/flight-computer/.pio/tools/ukf_replay.exe')
DATA = Path('data/simulated/all_trajectories.csv.gz')
META = Path('data/simulated/flight_metadata.csv.gz')
LOG = Path('data/112.txt')

# This diagnostic profile is not used for training or checkpoint selection.
BURST_SEED = 11400
BURST_PROFILE = baro.Scenario(1.5, 60_000)
BURST = dict(burst_start_probability=.007, burst_samples=4)


def load_models():
    models, runs = {}, []
    config = None
    for seed in w.SEEDS:
        name = f'w100_seed{seed}'
        aug, base = OUT / name, BASELINE / name
        a, b = (json.loads((p / 'run.json').read_text(encoding='utf-8')) for p in (aug, base))
        augmentation = a.pop('train_only_augmentations')
        assert a == b and a['seed'] == seed and a['window'] == 100 and a['epochs'] == 240
        assert augmentation.keys().isdisjoint(a['scenarios'])
        for key, path in [('ukf_replay', EXE), ('data', DATA), ('metadata', META)]:
            assert a[key + '_sha256'] == ap.sha256_file(path), path
        shared = {k: v for k, v in a.items() if k != 'seed'}
        assert config is None or config == shared
        config = shared
        for label, path in [('baseline', base), ('augmented', aug)]:
            model, scaler = baro._load(path)
            curve = pd.read_csv(path / 'epoch_history.csv')
            assert curve.epoch.tolist() == list(range(1, 241))
            best = curve.loc[curve.validation_loss.idxmin()]
            runs.append(dict(model=label, seed=seed, train_windows=int(scaler.n_samples_seen_),
                             best_epoch=int(best.epoch), validation_loss=best.validation_loss,
                             validation_last_1s_mae=best.apex_last_1s_mae_m))
            models[label, seed] = model, scaler
    return models, config, pd.DataFrame(runs)


def estimates(models, x, h, v):
    ht, vt = (torch.as_tensor(a, dtype=torch.float32) for a in (h, v))
    with torch.inference_mode():
        for (name, seed), (model, scaler) in models.items():
            ratio = model(torch.from_numpy(bench._scale_array(scaler, x)))[0]
            result = (ht + vt.square() / (2 * ap.SIM_G)) * ratio
            assert torch.isfinite(result).all()
            yield name, seed, result.numpy()


def simulate(models, config):
    frame = bench._read_data(DATA)
    meta = pd.read_csv(META).set_index('flight_id')[['apogee_time', 'burnout_time']]
    report, predictions = [], []
    for scenario, profile in {'original': None, **baro.SCENARIOS, 'burst_holdout': BURST_PROFILE}.items():
        if scenario == 'burst_holdout':
            ids = config['split']['validation'] + config['split']['test']
            source = frame.loc[frame.flight_id.isin(ids)]
            grid, _ = baro._replayed(source, EXE, profile, BURST_SEED, **BURST)
        else:
            source = frame
            grid, _ = baro._replayed(source, EXE, profile, config['profile_seeds'].get(scenario))
        endpoints = ap.make_endpoint_manifest(grid, max_window=min(config['windows']), stride=config['stride'])
        endpoints = bench._select_endpoints(source, endpoints, config['endpoint_phase'])
        endpoints = endpoints.loc[np.isfinite(endpoints.target) &
                                  (endpoints.endpoint_time <= endpoints.flight_id.map(meta.apogee_time))]
        endpoints = w._with_history_validity(grid, endpoints, config['window'])
        for part in ('validation', 'test'):
            x, h, v, _, rows = baro._samples(grid, endpoints, config['split'][part], meta, config['window'])
            for name, seed, estimate in estimates(models, x, h, v):
                pred = rows.assign(split=part, scenario=scenario, model=name, seed=seed, prediction=estimate)
                predictions.append(pred)
                for region, mask in w._regions(pred).items():
                    if region == 'last_available':
                        continue
                    err = pred.loc[mask, 'prediction'] - pred.loc[mask, 'apogee_rel']
                    if len(err):
                        report.append(dict(split=part, scenario=scenario, model=name, seed=seed, region=region,
                                           n=len(err), flights=pred.loc[mask, 'flight_id'].nunique(),
                                           mae=err.abs().mean(), p95=err.abs().quantile(.95), bias=err.mean()))
    report = pd.DataFrame(report)
    previous = pd.read_csv(BASELINE / 'checkpoint_comparison.csv').query('model == "extended"')
    matched = previous.merge(report.query('model == "baseline" & scenario != "burst_holdout"'),
                             on=['split', 'scenario', 'seed', 'region'], validate='one_to_one')
    assert len(matched) == len(previous)
    np.testing.assert_array_equal(matched.n_x, matched.n_y)
    np.testing.assert_allclose(matched.mae_x, matched.mae_y, atol=5e-5)
    report.to_csv(OUT / 'simulation_comparison.csv', index=False)
    pd.concat(predictions).to_csv(OUT / 'simulation_predictions.csv', index=False)
    ap.write_json(OUT / 'burst_holdout.json', dict(seed=BURST_SEED, scenario=w.asdict(BURST_PROFILE), **BURST))
    print('Test last-1s and coast metrics (seed means):')
    print(report.query('split == "test" & region in ["coast", "apex_last_1s"]').pivot_table(
        index=['scenario', 'region'], columns='model', values=['mae', 'p95', 'n', 'flights']).round(3).to_string())
    print('Per-seed test last-1s MAE:')
    print(report.query('split == "test" & region == "apex_last_1s"').pivot(
        index=['scenario', 'seed'], columns='model', values='mae').round(3).to_string())


def validate_log(models, config):
    # Reuse the exact latest-UKF grid; the observation assumptions and mask stay fixed.
    info = json.loads((LOG_REPLAY / '112_summary.json').read_text(encoding='utf-8'))
    upstream = json.loads((LOG_REPLAY / 'w100_seed42' / 'run.json').read_text(encoding='utf-8'))
    assert info['log_sha256'] == ap.sha256_file(LOG) and not info['apogee_observed']
    assert upstream['ukf_replay_sha256'] == config['ukf_replay_sha256']
    reports, traces = [], []
    for mode, trace in pd.read_csv(LOG_REPLAY / '112_validation_trace.csv').groupby('mode', sort=False):
        trace = trace.reset_index(drop=True)
        idx = np.flatnonzero(trace.usable_prediction)
        x = ap.slice_history(trace[list(ap.FEATURE_COLUMNS)].to_numpy(dtype=np.float32), idx, config['window'])
        assert all(trace.valid.iloc[i - config['window'] + 1:i + 1].all() for i in idx)
        predictions = trace.iloc[idx].copy()
        flight_count = int(predictions.t_since_accel_trigger.ge(0).sum())
        assert flight_count == info['modes'][mode]['flight_predicted']
        for name, seed, estimate in estimates(models, x, predictions.altitude_rel.to_numpy(),
                                               predictions.velocity_z.to_numpy()):
            predictions[f'{name}_{seed}'] = estimate
            tail = predictions.loc[predictions.t_to_log_end <= 1.0, f'{name}_{seed}']
            reports.append(dict(mode=mode, model=name, seed=seed, flight_predictions=flight_count,
                                last_1s_predictions=len(tail), final_m=float(tail.iloc[-1]),
                                last_1s_min_m=float(tail.min()), last_1s_max_m=float(tail.max()),
                                last_1s_below_log_max=int((tail < info['max_observed_baro_rel_m']).sum()),
                                final_seconds_before_log_end=float(predictions.t_to_log_end.iloc[-1])))
        traces.append(predictions[['mode', 't_since_accel_trigger', 't_to_log_end', 'altitude_rel',
                                   'velocity_z', *[f'{name}_{seed}' for name, seed in models]]])
    pd.concat(traces).to_csv(OUT / '112_predictions.csv', index=False)
    result = pd.DataFrame(reports)
    result.to_csv(OUT / '112_comparison.csv', index=False)
    print('112.txt regression (no apogee label):\n' + result.round(3).to_string(index=False))


if __name__ == '__main__':
    torch.set_num_threads(2)
    models, config, runs = load_models()
    runs.to_csv(OUT / 'training_comparison.csv', index=False)
    print(runs.to_string(index=False))
    simulate(models, config)
    validate_log(models, config)
