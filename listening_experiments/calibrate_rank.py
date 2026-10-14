"""Empirical calibration using Gaussian noise snapshots, never analytic R or DL risk.

Run from the project root with ``python -m listening_experiments.calibrate_rank``.
The threshold is selected on a training stream at T=128, then frozen before
independent validation at every paper listening budget. All rates are per BS,
per budget; finite-trial confidence intervals are not network guarantees.
"""
import argparse
from dataclasses import asdict, replace
import math
from pathlib import Path
import numpy as np

from multipath_support import planar_subarrays, spatially_smooth_covariance
from .config import MusicConfig, positive_int
from .observations import complex_normal
from .cache import load_cached_channels
from .storage import atomic_json, atomic_npz, fingerprint, code_signature
from .analysis import _wilson


def noise_excess_samples(positions, config, *, snapshots, trials, seed):
    """Largest smoothed eigenvalue minus unit noise power in independent trials."""
    positive_int(snapshots, "snapshots")
    positive_int(trials, "trials")
    config.validate()
    m = len(positions)
    indices = (planar_subarrays(positions, config.subarray_rows, config.subarray_cols)[0]
               if config.spatial_smoothing else np.arange(m)[None, :])
    rng = np.random.default_rng(seed)
    excess = np.empty(trials)
    for trial in range(trials):
        y = complex_normal(rng, (m, snapshots))
        covariance = y @ y.conj().T / snapshots
        smooth = spatially_smooth_covariance(
            covariance, indices, forward_backward=config.spatial_smoothing and config.forward_backward)
        excess[trial] = max(float(np.linalg.eigvalsh(smooth)[-1]) - 1, 0)
    return excess


def calibrate(positions, output_root, *, config=MusicConfig(), reference=128,
              windows=(1, 2, 4, 8, 16, 24, 32, 48, 57, 64), training_trials=2000,
              validation_trials=1000, target_per_bs=0.001, training_seed=310258,
              validation_seed=2026101307):
    if config.source_estimation != "rank":
        raise ValueError("This calibration is for rank source estimation")
    if not 0 < target_per_bs < 1 or training_seed == validation_seed:
        raise ValueError("Use a valid target and distinct training/validation seeds")
    training = noise_excess_samples(positions, config, snapshots=reference,
                                   trials=training_trials, seed=training_seed)
    quantile = float(np.quantile(training, 1 - target_per_bs, method="higher"))
    # Conservative rounding and 0.025 buffer, specified before validation.
    margin = round(math.ceil((quantile + .025) / .05) * .05, 10)
    calibrated = replace(config, rank_noise_margin=margin, rank_noise_reference_snapshots=reference)
    print(f"Training: quantile={quantile:.6f}, frozen reference margin={margin:g}", flush=True)
    samples, rows = {"training_excess": training}, []
    for L in windows:
        positive_int(L, "window count")
        T = reference * L
        excess = noise_excess_samples(positions, calibrated, snapshots=T, trials=validation_trials,
                                      seed=np.random.SeedSequence([validation_seed, T]))
        effective = calibrated.rank_margin_for_snapshots(T)
        count = int(np.sum(excess > effective))
        lo, hi = _wilson(count, validation_trials)
        row = dict(L=L, T=T, trials=validation_trials, noise_margin=effective,
                   false_detections=count, empirical_rate=count / validation_trials,
                   wilson_lower=lo, wilson_upper=hi,
                   old_margin_false_detections=int(np.sum(excess > .001)),
                   excess_max=float(excess.max()))
        rows.append(row)
        samples[f"validation_T{T}"] = excess
        print(f"Validation L={L:2d}, T={T:4d}, margin={effective:.5f}: {count}/{validation_trials} false detections", flush=True)
    report = dict(music=asdict(calibrated), reference_snapshots=reference, training_trials=training_trials,
                  validation_trials_per_budget=validation_trials, target_per_bs=target_per_bs,
                  training_seed=training_seed, validation_seed=validation_seed, training_quantile=quantile,
                  array_positions=np.asarray(positions).tolist(), rows=rows,
                  scope="unit-variance iid complex Gaussian noise, specified array/smoothing and T; per BS, not network-wide",
                  method="sample covariances only; held-out validation; no INR/rate used for selection",
                  code=code_signature(["listening_experiments/calibrate_rank.py", "listening_experiments/config.py",
                                       "listening_experiments/observations.py", "multipath_support.py"]))
    directory = Path(output_root) / fingerprint(report)[:16]
    atomic_json(directory / "calibration.json", report)
    atomic_npz(directory / "noise_excess_samples.npz", **samples)
    print(f"Calibration report: {directory.resolve()}", flush=True)
    return calibrated, report, directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("result/fig10_fig11/rank_calibration"))
    parser.add_argument("--training-trials", type=int, default=2000)
    parser.add_argument("--validation-trials", type=int, default=1000)
    args = parser.parse_args()
    channel = load_cached_channels(args.cache, num_macros=1)[0]
    calibrate(channel.array_positions_dl * channel.f_ul / channel.f_dl, args.output,
              training_trials=args.training_trials, validation_trials=args.validation_trials)
