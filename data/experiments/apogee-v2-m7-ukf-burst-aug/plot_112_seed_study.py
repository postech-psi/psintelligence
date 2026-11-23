"""Generate the 112.txt checkpoint variability plot from the PSIntelligence root."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from compare_augmentation import EXE, LOG, LOG_REPLAY, OUT, ap, baro, estimates

BATCHES = {
    OUT: (42, 43),
    OUT / 'seed-study-batch-a': (504723, 873529, 698543, 608065),
    OUT / 'seed-study-batch-b': (650042, 313474, 733519, 487659),
}
SEEDS = tuple(seed for batch in BATCHES.values() for seed in batch)
PRIMARY = 'every_logged_row'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def load_models():
    models, history = {}, []
    baseline = read_json(OUT / 'w100_seed42' / 'run.json')
    assert baseline['ukf_replay_sha256'] == ap.sha256_file(EXE)
    assert baseline['window'] == 100 and baseline['epochs'] == 240
    shared = {k: v for k, v in baseline.items() if k not in ('seed', 'seeds')}
    for directory, seeds in BATCHES.items():
        for seed in seeds:
            bundle = directory / f'w100_seed{seed}'
            config = read_json(bundle / 'run.json')
            assert config.pop('seed') == seed and config.pop('seeds') == list(seeds)
            assert shared == config
            model, scaler = baro._load(bundle)
            models['seed', seed] = model, scaler
            curve = pd.read_csv(bundle / 'epoch_history.csv')
            assert curve.epoch.tolist() == list(range(1, 241))
            best = curve.loc[curve.validation_loss.idxmin()]
            assert scaler.n_samples_seen_ == 7937
            history.append(dict(seed=seed, best_epoch=int(best.epoch), val_loss=float(best.validation_loss),
                                val_apex_last_1s_mae_m=float(best.apex_last_1s_mae_m)))
    assert len(models) == 10
    return models, shared, pd.DataFrame(history)


def evaluate_log(models, config):
    summary = read_json(LOG_REPLAY / '112_summary.json')
    assert summary['log_sha256'] == ap.sha256_file(LOG) and summary['apogee_observed'] is False
    old = pd.read_csv(OUT / '112_comparison.csv').query('model == "augmented"')
    rows, metrics = [], []
    for mode, trace in pd.read_csv(LOG_REPLAY / '112_validation_trace.csv').groupby('mode', sort=False):
        trace = trace.reset_index(drop=True)
        indices = np.flatnonzero(trace.usable_prediction & trace.t_since_accel_trigger.ge(0))
        assert len(indices) == summary['modes'][mode]['flight_predicted']
        valid = trace.valid.to_numpy()
        window = config['window']
        assert all(i >= window - 1 and valid[i - window + 1:i + 1].all() for i in indices)
        x = ap.slice_history(trace[list(ap.FEATURE_COLUMNS)].to_numpy(dtype=np.float32), indices, window)
        rows_for_plot = trace.loc[trace.report_clock & trace.t_since_accel_trigger.ge(0),
                                  ['mode', 't_since_accel_trigger', 't_to_log_end', 'usable_prediction',
                                   'altitude_rel', 'velocity_z']].copy()
        elapsed = trace.loc[indices, 't_since_accel_trigger'].to_numpy()
        end = trace.loc[indices, 't_to_log_end'].to_numpy()
        recent = end <= 1.0
        assert recent.sum() == summary['modes'][mode]['last_1s_predicted']
        for _, seed, result in estimates(models, x, trace.altitude_rel.to_numpy()[indices],
                                         trace.velocity_z.to_numpy()[indices]):
            rows_for_plot.loc[indices, f'seed_{seed}'] = result
            if seed in (42, 43):
                ref = old.loc[old['mode'].eq(mode) & old.seed.eq(seed), 'final_m'].item()
                np.testing.assert_allclose(result[-1], ref, rtol=0, atol=5e-5)
            metrics.append(dict(mode=mode, seed=seed, n=len(result), final_m=float(result[-1]),
                                final_log_s=summary['accel_trigger_time_s'] + elapsed[-1],
                                last_1s_min_m=float(result[recent].min()), last_1s_max_m=float(result[recent].max())))
        rows.append(rows_for_plot)
    return summary, pd.concat(rows, ignore_index=True), pd.DataFrame(metrics)


def plot(summary, trace, path):
    observed = pd.read_csv(LOG_REPLAY / '112_every_logged_row_observations.csv')
    launch = summary['accel_trigger_time_s']
    observed = observed.loc[observed.t_log_s.ge(launch)]
    tx = observed.t_log_s.to_numpy() - launch
    height = observed.logged_h.to_numpy() - summary['pad_zero_m']
    gaps = np.flatnonzero(np.diff(tx) > .035)
    assert len(gaps) == 1
    gap_start = tx[gaps[0] + 1]
    assert not observed.replay_valid.to_numpy()[gaps[0] + 1:].any()
    trace = trace.loc[trace['mode'].eq(PRIMARY)]
    tt = trace.t_since_accel_trigger.to_numpy()
    y = trace[[f'seed_{seed}' for seed in SEEDS]].to_numpy()
    mask = trace.usable_prediction.to_numpy()
    assert np.array_equal(np.isfinite(y).all(axis=1), mask)
    assert np.isnan(y[~mask]).all()
    assert tt[np.flatnonzero(mask)[-1]] < gap_start
    lo, hi, median = (np.full(len(tt), np.nan) for _ in range(3))
    lo[mask], hi[mask], median[mask] = y[mask].min(axis=1), y[mask].max(axis=1), np.median(y[mask], axis=1)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, (main, zoom) = plt.subplots(1, 2, figsize=(12, 5), width_ratios=[1.65, 1], layout='constrained')
    for ax in (main, zoom):
        ax.axvspan(gap_start, tx[-1], color='#64748b', alpha=.18, lw=0, label='Replay velocity invalid')
        ax.plot(tx, height, color='#234963', lw=2.35, label='Onboard UKF altitude (logged)')
        ax.fill_between(tt, lo, hi, color='#dd8530', alpha=.21, label='Min–max across 10 seeds')
        for series in y.T:
            ax.plot(tt, series, color='#9c4c14', lw=.75, alpha=.28)
        ax.plot(tt, median, color='#9b4000', lw=1.85, marker='o', ms=2.7, label='GRU median')
        ax.grid(axis='y', color='#dbe3ea', linewidth=.7)
        ax.set_axisbelow(True)
        ax.set_xlabel('Time since acceleration trigger (s)')
    main.set(xlim=(0, tx[-1]+.13), ylim=(-5, max(215, float(np.nanmax(hi))+12)), ylabel='Altitude above pad (m)')
    main.set_title('Full flight log', loc='left', fontweight='bold')
    main.legend(loc='upper left', frameon=False, fontsize=9)
    tail = mask & (tt >= tx[-1] - 1.05)
    zoom.set(xlim=(tx[-1]-1.05, tx[-1]+.08),
             ylim=(min(height[tx >= tx[-1]-1.07].min(), lo[tail].min())-1.2, hi[tail].max()+2.3))
    zoom.set_title('Last second of the log', loc='left', fontweight='bold')
    last = np.flatnonzero(mask)[-1]
    zoom.annotate(f'Final GRU: {lo[last]:.2f}–{hi[last]:.2f} m', (tt[last], median[last]),
                  xytext=(.04,.92), textcoords='axes fraction', va='top', color='#843800',
                  arrowprops=dict(arrowstyle='-', color='#843800', lw=.85))
    fig.suptitle('112.txt | Apogee estimates from 10 augmented GRU training seeds', fontweight='bold')
    fig.savefig(path, dpi=160, facecolor='white')


if __name__ == '__main__':
    torch.set_num_threads(2)
    models, config, history = load_models()
    summary, trace, metrics = evaluate_log(models, config)
    metrics = metrics.merge(history, on='seed', validate='many_to_one')
    trace.to_csv(OUT / '112_10seed_trace.csv', index=False)
    metrics.to_csv(OUT / '112_10seed_summary.csv', index=False)
    plot(summary, trace, OUT / '112_10seed_apogee_vs_ukf.png')
    stats = metrics.groupby('mode').final_m.agg(['min', 'median', 'max', 'std'])
    print(stats.round(4).to_string())
    print(metrics.pivot(index='seed', columns='mode', values='final_m').round(4).to_string())
