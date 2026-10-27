"""Independent noise validation and observation invariants before a full sweep."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np

from listening_experiments.calibrate_rank import noise_excess_samples
from listening_experiments.analysis import _wilson
from listening_experiments.observations import window_observations, pooled_observations
from listening_experiments.storage import atomic_json, atomic_npz, fingerprint


def check_observations(case, observation):
    """Compare streamed prefixes to explicit X X^H / T and inspect independent sector noise."""
    from copy import copy
    short = copy(observation)
    short.listening_windows = (1, 2)
    blocks = [window_observations(case.channel, short, 0, window)[0][0] for window in (0, 1)]
    for windows, covariance, activity in pooled_observations(case.channel, short, 0):
        x = np.concatenate(blocks[:windows], axis=-1)
        expected = x @ x.swapaxes(-1, -2).conj() / x.shape[-1]
        np.testing.assert_allclose(covariance[0], expected, rtol=1e-12, atol=1e-25)
        assert np.all(activity == windows)
    # Reproduce the generator's independent symbol and noise streams.
    from listening_experiments.observations import complex_normal
    seed = [short.seed, case.channel.macro_id, 0, 0]
    users = case.channel.h_ntn_ul.shape[0]
    symbols = complex_normal(np.random.default_rng(np.random.SeedSequence(seed+[1])),
                             (users, short.snapshots_per_window))
    h = case.channel.h_ntn_ul[:, 0, :, :].conj().transpose(1, 2, 0)
    residual = blocks[0] - h @ symbols
    for b in range(len(h)):
        expected = complex_normal(np.random.default_rng(np.random.SeedSequence(seed+[2,b])),
                                  (h.shape[1], short.snapshots_per_window))*np.sqrt(short.sensing_noise_normalized)
        np.testing.assert_allclose(residual[b], expected, rtol=1e-10, atol=1e-20)
    assert not np.array_equal(residual[0], residual[1])
    return dict(prefix_covariance_agrees=True, activity_always_one=True,
                shared_user_symbols=True, independent_sector_noise=True)


def _noise_job(arguments):
    positions, music, budget, trials, seed = arguments
    excess = noise_excess_samples(positions, music, snapshots=budget, trials=trials,
                                  seed=np.random.SeedSequence([seed, 990001, budget]))
    margin = music.rank_margin_for_snapshots(budget)
    false = int(np.sum(excess > margin))
    lower, upper = _wilson(false, trials)
    row = dict(T=budget, trials=trials, false_source_detections=false, rate=false/trials,
               wilson_lower=lower, wilson_upper=upper, rank_margin=margin,
               maximum_noise_excess=float(excess.max()))
    return budget, excess, row


def validate_noise(directory, case, config, music):
    """Check the frozen rule on new noise trials; do not tune it to experiment outcomes."""
    import json
    path = directory / "noise_validation.json"
    identity = fingerprint(dict(positions=case.channel.array_positions_dl, music=music,
                                snapshots=config.snapshots, trials=config.noise_trials, seed=config.seed,
                                limit=config.noise_wilson_limit))
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["identity"] != identity or not saved["passed"]:
            raise ValueError("Noise validation checkpoint does not match this configuration")
        return saved
    jobs = [(case.channel.array_positions_dl, music, t, config.noise_trials, config.seed)
            for t in config.snapshots]
    rows, arrays = [], {}
    with ProcessPoolExecutor(max_workers=min(config.workers, len(jobs)), mp_context=mp.get_context("spawn")) as pool:
        for future in as_completed([pool.submit(_noise_job, job) for job in jobs]):
            budget, excess, row = future.result()
            rows.append(row)
            arrays[f"excess_T{budget}"] = excess
            print(f"Noise T={budget}: {row['false_source_detections']}/{config.noise_trials}, "
                  f"95% upper={row['wilson_upper']:.4f}", flush=True)
    rows.sort(key=lambda row: row["T"])
    passed = all(row["wilson_upper"] <= config.noise_wilson_limit for row in rows)
    report = dict(identity=identity, passed=passed, rows=rows, limit=config.noise_wilson_limit,
        seed_namespace=990001,
        rule=f"{music.rank_noise_margin:g} * sqrt({music.rank_noise_reference_snapshots} / T)",
        scope="Per-sector source-count false alarms under iid complex Gaussian noise; not network-wide",
        peak_false_alarm_bound="Accepted peaks require K_hat > 0, so source-count false alarms bound peak false alarms")
    atomic_npz(directory / "noise_validation_samples.npz", **arrays)
    atomic_json(path, report)
    if not passed:
        raise RuntimeError("Frozen rank rule failed held-out noise validation; inspect report before running")
    return report
