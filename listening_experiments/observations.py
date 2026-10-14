"""Shared user activity/waveforms, independent BS noise, and pooled sample covariance."""
import numpy as np

from .config import ExperimentConfig


def complex_normal(rng, shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def window_observations(channels, config: ExperimentConfig, episode, window):
    """Return all p cases for one physical window in the reference 'conj' convention.

    Y is divided by sqrt(P_UL), so active source symbols have unit variance and
    noise variance is N_BS/P_UL. Each user's CIR has already been summed coherently.
    Seeds are window-specific: extending the grid does not change existing data.
    """
    h = channels.h_ntn_ul[:, 0, :, :].conj().transpose(1, 2, 0)  # [BS, antenna, user]
    ntx, m, users = h.shape
    seed = [config.seed, int(channels.macro_id), int(episode), int(window)]
    activity_u = np.random.default_rng(np.random.SeedSequence(seed + [0])).random(users)
    activities = activity_u[None, :] < np.asarray(config.p_values)[:, None]
    symbols = complex_normal(np.random.default_rng(np.random.SeedSequence(seed + [1])),
                             (users, config.snapshots_per_window))
    noise = np.stack([
        complex_normal(np.random.default_rng(np.random.SeedSequence(seed + [2, b])),
                       (m, config.snapshots_per_window))
        for b in range(ntx)]) * np.sqrt(config.sensing_noise_normalized)
    data = np.stack([h @ (active[:, None] * symbols) + noise for active in activities])
    return data, activities


def pooled_observations(channels, config, episode):
    """Yield (L, covariance[p,BS,M,M], J[p,user]) only at requested budgets."""
    targets = set(config.listening_windows)
    m, ntx = channels.h_ntn_ul.shape[3], channels.h_ntn_ul.shape[2]
    cumulative = np.zeros((len(config.p_values), ntx, m, m), dtype=np.complex128)
    counts = np.zeros((len(config.p_values), channels.h_ntn_ul.shape[0]), dtype=np.int64)
    for window in range(max(targets)):
        data, active = window_observations(channels, config, episode, window)
        cumulative += data @ data.swapaxes(-1, -2).conj()
        counts += active
        L = window + 1
        if L in targets:
            covariance = cumulative / (L * config.snapshots_per_window)
            yield L, (covariance + covariance.swapaxes(-1, -2).conj()) / 2, counts.copy()
