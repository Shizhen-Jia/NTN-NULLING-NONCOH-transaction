"""Re-score archived physical INR at a separate publication threshold."""
from pathlib import Path
import numpy as np

from listening_experiments.storage import atomic_json, atomic_npz
from .analysis import write_csv
from .runner import records


def summarize_protection(directory, threshold_db=-6.0):
    """Keep simulation settings and legacy metrics fixed; resample deployments."""
    directory = Path(directory)
    config, loaded = records(directory)
    if not np.isfinite(threshold_db):
        raise ValueError("The protection threshold must be finite")
    rng = np.random.default_rng(config.validation_seed + 100)
    draws = rng.integers(0, config.num_scenes,
                        size=(config.bootstrap_draws, config.num_scenes))

    def aggregate(db):
        threshold = 10 ** (db / 10)
        final, baseline = [], []
        for scene in range(config.num_scenes):
            group = [r for r in loaded if int(r['macro_id']) == scene]
            final.append(np.mean([(r['inr_linear'] > threshold).mean(axis=-1)
                                  for r in group], axis=0))
            baseline.append(np.mean([np.mean(r['raw_inr_linear'] > threshold)
                                     for r in group]))
        final, baseline = np.stack(final), np.asarray(baseline)
        stats = {}
        for name, values in [('q_out', final), ('q_mrt', baseline)]:
            lo, hi = np.quantile(values[draws].mean(axis=1), [.025, .975], axis=0)
            for suffix, value in zip(['', '_lower', '_upper'],
                                     [values.mean(axis=0), lo, hi]):
                stats[name + suffix] = np.broadcast_to(value, final.shape[1:]).copy()
        return stats

    reference = aggregate(config.threshold_db)
    with np.load(directory / 'analysis_arrays.npz') as previous:
        for name, value in reference.items():
            np.testing.assert_allclose(value, previous[name], rtol=1e-12, atol=1e-14)
    result = aggregate(threshold_db)
    if threshold_db < config.threshold_db:
        for name in result:
            assert np.all(result[name] >= reference[name] - 1e-14)
    magnitude = f'{abs(threshold_db):g}'.replace('.', 'p')
    tag = ('minus' if threshold_db < 0 else 'plus') + magnitude + 'db'
    stem = 'protection_' + tag
    result.update(p_values=np.asarray(config.p_values), windows=np.asarray(config.windows),
                  lambdas=np.asarray(config.lambdas), threshold_db=np.asarray(threshold_db))
    atomic_npz(directory / f'{stem}_arrays.npz', **result)
    rows = []
    for pi, p in enumerate(config.p_values):
        for li, L in enumerate(config.windows):
            for wi, weight in enumerate(config.lambdas):
                row = dict(p=p, L=L, T=L*config.snapshots_per_window,
                           lambda_value=weight, threshold_db=threshold_db)
                for name in ['q_out', 'q_out_lower', 'q_out_upper',
                             'q_mrt', 'q_mrt_lower', 'q_mrt_upper']:
                    row[name] = float(result[name][pi, li, wi])
                rows.append(row)
    write_csv(directory / f'{stem}_summary.csv', rows)
    atomic_json(directory / f'{stem}_config.json', dict(
        threshold_db=threshold_db, threshold_linear=10**(threshold_db/10),
        noise_rise_db=float(10*np.log10(1+10**(threshold_db/10))),
        reference_threshold_db=config.threshold_db, reference_statistics_verified=True,
        scenes=config.num_scenes, repetitions=config.repeats,
        bootstrap_draws=config.bootstrap_draws, bootstrap_seed=config.validation_seed+100,
        aggregation='Equal linked UE within each record, then equal repetition and deployment',
        intervals='Pointwise 95% deployment-cluster percentile intervals; all budgets, p and lambda paired',
        scope='Final-beam physical INR; all fixed linked UEs, including silent and undetected users',
        source_run_id=str(loaded[0]['run_id']), figure_stem='activity_coverage_protection_ab_'+tag))
    return result
