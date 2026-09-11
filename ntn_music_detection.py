"""MUSIC-based narrowband detection/angle utilities for NTN/TN channels.

The blind detector used in the notebook follows the anonymous noncoherent model

    Rxx ~= sum_k g_k u_k u_k^H + sigma^2 I

and is referred to as ``Blind MDL-MUSIC Detection`` when the source count is
estimated with MDL.

This module is designed for the per-BS channel tensor:
    hi.shape == (num_ntn, num_ntn_ant, num_bs_ant)

Typical usage in your notebook:
    from ntn_music_detection import detect_music_from_hi, run_music_angle_pipeline
    out = detect_music_from_hi(hi=h_i, num_sources=None, threshold=3.0)
    res = run_music_angle_pipeline(h_all, tx_rows=8, tx_cols=8, nsect=3)
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
from scipy.ndimage import minimum_filter
from scipy.optimize import minimize, nnls

# Re-export the shared CIR helper for existing notebook/module imports.
from multipath_support import collapse_cir_to_narrowband


BLIND_MDL_MUSIC_DETECTION_NAME = "Blind MDL-MUSIC Detection"
BLIND_MDL_MUSIC_MODEL_SUMMARY = "Rxx ~= sum_k g_k u_k u_k^H + sigma^2 I"


def blind_music_detection_name(source_estimation: str = "mdl") -> str:
    """Return the display name for the configured blind MUSIC detector."""
    mode = str(source_estimation).strip().lower()
    if mode == "mdl":
        return BLIND_MDL_MUSIC_DETECTION_NAME
    return f"Blind {mode.upper()}-MUSIC Detection"


def _as_complex_array(x: np.ndarray) -> np.ndarray:
    """Convert input to a complex ndarray without modifying the original."""
    arr = np.asarray(x)
    if np.iscomplexobj(arr):
        return arr.astype(np.complex128, copy=False)
    return arr.astype(np.complex128) + 0j


def _to_numpy(x: Any) -> np.ndarray:
    """Convert tensor-like objects to NumPy arrays without changing values."""
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def extract_hi_for_tx(
    h_all: np.ndarray,
    tx_index: int,
    *,
    nonzero_only: bool = False,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract one TX tensor hi from h_all.

    Parameters
    ----------
    h_all : np.ndarray
        Shape (num_rx, num_rx_ant, num_tx, num_tx_ant).
    tx_index : int
        Flat TX index in [0, num_tx-1].
    nonzero_only : bool
        If True, keep only RX rows with any non-zero channel in hi.
    eps : float
        Threshold for non-zero check.

    Returns
    -------
    hi : np.ndarray
        Shape (num_rx_kept, num_rx_ant, num_tx_ant).
    rx_mask : np.ndarray
        Bool mask over original RX index, shape (num_rx,).
    """
    h = _as_complex_array(h_all)
    if h.ndim != 4:
        raise ValueError(
            "h_all must have shape (num_rx, num_rx_ant, num_tx, num_tx_ant)."
        )

    num_rx, _num_rx_ant, num_tx, _num_tx_ant = h.shape
    t = int(tx_index)
    if t < 0 or t >= num_tx:
        raise ValueError(f"tx_index={t} out of range for num_tx={num_tx}.")

    hi = h[:, :, t, :]
    if not nonzero_only:
        return hi, np.ones((num_rx,), dtype=bool)

    rx_mask = np.any(np.abs(hi) > float(eps), axis=(1, 2))
    return hi[rx_mask], rx_mask


def extract_tx_channel_matrix(
    h_all: np.ndarray,
    tx_index: int,
    *,
    rx_ant_index: int = 0,
    nonzero_only: bool = False,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract per-TX matrix H_t for one RX antenna.

    Returns H_t with shape (num_rx_kept, num_tx_ant), matching your
    per-sector MUSIC input requirement like (100, 64).
    """
    hi, rx_mask = extract_hi_for_tx(h_all, tx_index, nonzero_only=nonzero_only, eps=eps)
    num_rx_ant = hi.shape[1]
    r = int(rx_ant_index)
    if r < 0 or r >= num_rx_ant:
        raise ValueError(f"rx_ant_index={r} out of range for num_rx_ant={num_rx_ant}.")
    return hi[:, r, :], rx_mask


def _broadcast_powers(
    user_powers: Optional[np.ndarray],
    num_ntn: int,
    num_ntn_ant: int,
) -> np.ndarray:
    """Broadcast user powers to shape (num_ntn, num_ntn_ant)."""
    if user_powers is None:
        return np.ones((num_ntn, num_ntn_ant), dtype=np.float64)

    p = np.asarray(user_powers, dtype=np.float64)
    if p.ndim == 1:
        if p.shape[0] != num_ntn:
            raise ValueError(
                "user_powers has shape (num_ntn,), but num_ntn does not match."
            )
        return np.repeat(p[:, None], num_ntn_ant, axis=1)

    if p.ndim == 2 and p.shape == (num_ntn, num_ntn_ant):
        return p

    raise ValueError(
        "user_powers must be None, shape (num_ntn,), or shape (num_ntn, num_ntn_ant)."
    )


def _covariance_from_static_channels(
    hi: np.ndarray,
    user_powers_2d: np.ndarray,
    noise_var: float,
) -> np.ndarray:
    """Build covariance analytically for static narrowband channels.

    Model:
        x = sum_{u,r} sqrt(p_{u,r}) h_{u,r} s_{u,r} + w
        E[s s^H] = I, E[w w^H] = noise_var * I
    """
    num_ntn, num_ntn_ant, num_bs_ant = hi.shape
    rxx = np.zeros((num_bs_ant, num_bs_ant), dtype=np.complex128)

    for u in range(num_ntn):
        for r in range(num_ntn_ant):
            h = hi[u, r, :].reshape(-1, 1)
            rxx += user_powers_2d[u, r] * (h @ h.conj().T)

    if noise_var > 0.0:
        rxx += noise_var * np.eye(num_bs_ant, dtype=np.complex128)
    return rxx


def _sample_covariance_from_snapshots(
    hi: np.ndarray,
    user_powers_2d: np.ndarray,
    noise_var: float,
    num_snapshots: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic snapshots and return (Rxx, X).

    - symbols are circular Gaussian CN(0, 1).
    - This is standard for subspace estimation and is modulation-agnostic.
    """
    num_ntn, num_ntn_ant, num_bs_ant = hi.shape
    ur = num_ntn * num_ntn_ant

    # H shape: (M, UR), each column is one source channel vector to BS array.
    h_mat = hi.reshape(ur, num_bs_ant).T
    p_vec = user_powers_2d.reshape(ur)
    p_sqrt = np.sqrt(np.maximum(p_vec, 0.0))

    s = (
        rng.standard_normal((ur, num_snapshots))
        + 1j * rng.standard_normal((ur, num_snapshots))
    ) / np.sqrt(2.0)
    s *= p_sqrt[:, None]

    x_clean = h_mat @ s
    if noise_var > 0.0:
        w = (
            rng.standard_normal((num_bs_ant, num_snapshots))
            + 1j * rng.standard_normal((num_bs_ant, num_snapshots))
        ) * np.sqrt(noise_var / 2.0)
        x = x_clean + w
    else:
        x = x_clean

    rxx = (x @ x.conj().T) / float(num_snapshots)
    return rxx, x


def _estimate_num_sources_mdl(
    eigenvalues_desc: np.ndarray,
    num_snapshots: int,
    max_sources: Optional[int] = None,
) -> int:
    """Estimate source count with Wax-Kailath MDL."""
    eig = np.real(np.asarray(eigenvalues_desc, dtype=np.float64))
    if eig.size == 0 or not np.all(np.isfinite(eig)) or float(np.max(eig)) <= 0.0:
        return 0
    eig = np.maximum(eig / float(np.max(eig)), np.finfo(np.float64).eps)
    m = eig.shape[0]
    if max_sources is None:
        max_sources = m - 1
    max_sources = int(np.clip(max_sources, 0, m - 1))

    # MDL is meaningful with at least a few snapshots.
    n = int(max(num_snapshots, 2))

    mdl_vals = np.full(max_sources + 1, np.inf, dtype=np.float64)
    for k in range(max_sources + 1):
        noise_eigs = eig[k:]
        p = m - k
        if p <= 0:
            continue
        am = np.mean(noise_eigs)
        # Wax-Kailath MDL:
        # MDL(k) = -n*(m-k)*log(gm/am) + 0.5*k*(2m-k)*log(n)
        log_ratio = min(float(np.mean(np.log(noise_eigs)) - np.log(am)), 0.0)
        mdl_vals[k] = -n * p * log_ratio + 0.5 * k * (2 * m - k) * np.log(n)

    return int(np.argmin(mdl_vals))


def _estimate_num_sources_energy(
    eigenvalues_desc: np.ndarray,
    energy_ratio: float = 0.95,
) -> int:
    """Fallback source-count estimate by cumulative eigen-energy."""
    eig = np.real(np.asarray(eigenvalues_desc, dtype=np.float64))
    eig = np.maximum(eig, 0.0)
    total = float(np.sum(eig))
    if total <= 0.0:
        return 0
    csum = np.cumsum(eig) / total
    k = int(np.searchsorted(csum, energy_ratio) + 1)
    return int(np.clip(k, 0, eig.shape[0] - 1))


def _signal_eigenvalues_above_noise(
    eigenvalues_desc: np.ndarray,
    *,
    noise_power: float,
) -> np.ndarray:
    """Return descending covariance eigenvalues after removing a known noise floor."""
    eig = np.array(np.real(eigenvalues_desc), dtype=np.float64, copy=True).reshape(-1)
    eig[~np.isfinite(eig)] = 0.0
    return np.maximum(eig - float(max(noise_power, 0.0)), 0.0)


def _estimate_num_sources_rank(
    eigenvalues_desc: np.ndarray,
    *,
    noise_power: float = 0.0,
    relative_threshold: float = 1e-4,
    noise_margin: float = 1e-3,
) -> int:
    """Estimate the effective analytic covariance rank above noise and dynamic-range floors."""
    if float(relative_threshold) < 0.0 or float(noise_margin) < 0.0:
        raise ValueError("rank thresholds must be nonnegative.")

    sig = _signal_eigenvalues_above_noise(
        eigenvalues_desc,
        noise_power=noise_power,
    )
    if sig.size == 0 or float(sig[0]) <= 0.0:
        return 0

    floor = max(
        float(relative_threshold) * float(sig[0]),
        float(noise_margin) * float(max(noise_power, 0.0)),
        np.finfo(np.float64).eps * float(sig[0]),
    )
    return int(np.clip(np.count_nonzero(sig > floor), 0, sig.size - 1))


def _estimate_num_sources_eigengap(
    eigenvalues_desc: np.ndarray,
    *,
    noise_power: float = 0.0,
    relative_threshold: float = 1e-4,
    noise_margin: float = 1e-3,
) -> int:
    """Estimate K from the largest signal-eigenvalue drop above the rank floor."""
    sig = _signal_eigenvalues_above_noise(
        eigenvalues_desc,
        noise_power=noise_power,
    )
    if sig.size < 2 or float(sig[0]) <= 0.0:
        return 0

    floor = max(
        float(relative_threshold) * float(sig[0]),
        float(noise_margin) * float(max(noise_power, 0.0)),
        np.finfo(np.float64).eps * float(sig[0]),
    )
    rank_cap = int(np.clip(np.count_nonzero(sig > floor), 0, sig.size - 1))
    if rank_cap <= 1:
        return rank_cap

    ratios = sig[:rank_cap] / np.maximum(sig[1 : rank_cap + 1], floor)
    return int(np.argmax(ratios) + 1)


def _compute_user_scores(
    hi: np.ndarray,
    us: np.ndarray,
    en: np.ndarray,
    reduce_ntn_ant: Literal["max", "mean"],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute MUSIC-like user scores from signal/noise subspace projections.

    score = ||Us^H h||^2 / ||En^H h||^2 for normalized h.
    """
    num_ntn, num_ntn_ant, _ = hi.shape
    eps = 1e-12
    score_per_ant = np.zeros((num_ntn, num_ntn_ant), dtype=np.float64)

    for u in range(num_ntn):
        for r in range(num_ntn_ant):
            h = hi[u, r, :].reshape(-1, 1)
            norm_h = np.linalg.norm(h)
            if norm_h <= eps:
                score_per_ant[u, r] = 0.0
                continue
            h_n = h / norm_h

            sig_proj = np.linalg.norm(us.conj().T @ h_n) ** 2 if us.size else 0.0
            noi_proj = np.linalg.norm(en.conj().T @ h_n) ** 2 if en.size else eps
            score_per_ant[u, r] = float(sig_proj / max(noi_proj, eps))

    if reduce_ntn_ant == "max":
        score_user = np.max(score_per_ant, axis=1)
    elif reduce_ntn_ant == "mean":
        score_user = np.mean(score_per_ant, axis=1)
    else:
        raise ValueError("reduce_ntn_ant must be 'max' or 'mean'.")

    return score_per_ant, score_user


def detect_ntn_music_from_hi(
    hi: np.ndarray,
    *,
    num_sources: Optional[int] = None,
    threshold: Optional[float] = 1.0,
    user_powers: Optional[np.ndarray] = None,
    noise_var: float = 0.0,
    covariance_mode: Literal["analytic", "sample"] = "analytic",
    num_snapshots: int = 200,
    rng_seed: Optional[int] = None,
    source_estimation: Literal["mdl", "energy", "rank"] = "mdl",
    energy_ratio: float = 0.95,
    rank_relative_threshold: float = 1e-4,
    rank_noise_margin: float = 1e-3,
    reduce_ntn_ant: Literal["max", "mean"] = "max",
    compute_user_scores: bool = True,
    covariance_override: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Run narrowband MUSIC detection for one BS/sector channel tensor.

    Parameters
    ----------
    hi : np.ndarray
        Shape (num_ntn, num_ntn_ant, num_bs_ant), complex channel tensor.
    num_sources : int | None
        Signal subspace dimension K. If None, estimated automatically.
    threshold : float | None
        Detection threshold on user score.
        - If provided: detected_mask_user = score_user >= threshold
        - If None: top-K users are marked as detected.
    user_powers : np.ndarray | None
        Optional source powers:
        - shape (num_ntn,)
        - or shape (num_ntn, num_ntn_ant)
    noise_var : float
        Noise variance per BS antenna.
    covariance_mode : {"analytic", "sample"}
        "analytic": Rxx from static channels directly.
        "sample": synthetic snapshots are generated and sample covariance is used.
    num_snapshots : int
        Number of snapshots when covariance_mode == "sample".
    rng_seed : int | None
        Random seed for reproducibility in sample mode.
    source_estimation : {"mdl", "energy", "rank"}
        Automatic K-estimation rule when num_sources is None.
    energy_ratio : float
        Used only when source_estimation == "energy".
    rank_relative_threshold : float
        Signal-eigenvalue floor relative to the strongest signal eigenvalue.
    rank_noise_margin : float
        Additional floor relative to the known analytic noise power.
    reduce_ntn_ant : {"max", "mean"}
        How to aggregate multi-antenna NTN scores to user-level score.
    compute_user_scores : bool
        If False, only covariance/subspace estimation is returned. This is used
        by the blind anonymous detector, which treats `(u_k, g_k)` peaks as the
        primary outputs and computes any per-user mapping only for evaluation.

    covariance_override : ndarray | None
        Optional preprocessed covariance, in the same channel convention as hi.
        Used for spatial smoothing. No synthetic independent path signals are
        generated. hi is used only for dimensions and optional user diagnostics.

    Returns
    -------
    Dict[str, np.ndarray]
        Keys:
        - "detected_mask_user": bool array, shape (num_ntn,)
        - "detected_mask_per_ant": bool array, shape (num_ntn, num_ntn_ant)
        - "score_user": float array, shape (num_ntn,)
        - "score_per_ant": float array, shape (num_ntn, num_ntn_ant)
        - "num_sources_est": int scalar in ndarray
        - "threshold_used": float scalar in ndarray
        - "eigenvalues_desc": float array, shape (num_bs_ant,)
        - "covariance": complex array, shape (num_bs_ant, num_bs_ant)
    """
    hi_c = _as_complex_array(hi)
    if hi_c.ndim != 3:
        raise ValueError(
            "hi must be a 3D tensor with shape (num_ntn, num_ntn_ant, num_bs_ant)."
        )

    num_ntn, num_ntn_ant, num_bs_ant = hi_c.shape
    if num_bs_ant < 2:
        raise ValueError("MUSIC requires num_bs_ant >= 2.")
    if noise_var < 0.0:
        raise ValueError("noise_var must be >= 0.")
    if covariance_mode == "sample" and num_snapshots < 2:
        raise ValueError("num_snapshots must be >= 2 in sample mode.")

    p_2d = _broadcast_powers(user_powers, num_ntn=num_ntn, num_ntn_ant=num_ntn_ant)

    if covariance_mode not in ("analytic", "sample"):
        raise ValueError("covariance_mode must be 'analytic' or 'sample'.")
    if covariance_override is not None:
        rxx = np.asarray(covariance_override, dtype=np.complex128)
        if rxx.shape != (num_bs_ant, num_bs_ant) or not np.all(np.isfinite(rxx)):
            raise ValueError("covariance_override must be a finite M by M matrix.")
        scale = np.linalg.norm(rxx)
        if np.linalg.norm(rxx-rxx.conj().T) > 1e-8*scale:
            raise ValueError("covariance_override must be Hermitian.")
        rxx = (rxx+rxx.conj().T)/2
        if np.linalg.eigvalsh(rxx)[0] < -1e-8*scale:
            raise ValueError("covariance_override must be positive semidefinite.")
        n_for_mdl = int(num_snapshots) if covariance_mode == "sample" else max(num_ntn*num_ntn_ant, num_bs_ant+1)
    elif covariance_mode == "analytic":
        rxx = _covariance_from_static_channels(hi_c, p_2d, noise_var)
        n_for_mdl = max(num_ntn * num_ntn_ant, num_bs_ant + 1)
    elif covariance_mode == "sample":
        rng = np.random.default_rng(rng_seed)
        rxx, _x = _sample_covariance_from_snapshots(
            hi=hi_c,
            user_powers_2d=p_2d,
            noise_var=noise_var,
            num_snapshots=num_snapshots,
            rng=rng,
        )
        n_for_mdl = int(num_snapshots)
    else:
        raise ValueError("covariance_mode must be 'analytic' or 'sample'.")

    # Hermitian eigendecomposition.
    evals, evecs = np.linalg.eigh(rxx)
    idx = np.argsort(np.real(evals))[::-1]
    evals_desc = np.real(evals[idx])
    evecs_desc = evecs[:, idx]

    k_mdl = _estimate_num_sources_mdl(
        eigenvalues_desc=evals_desc,
        num_snapshots=n_for_mdl,
    )
    k_rank = _estimate_num_sources_rank(
        eigenvalues_desc=evals_desc,
        noise_power=noise_var,
        relative_threshold=rank_relative_threshold,
        noise_margin=rank_noise_margin,
    )
    k_eigengap = _estimate_num_sources_eigengap(
        eigenvalues_desc=evals_desc,
        noise_power=noise_var,
        relative_threshold=rank_relative_threshold,
        noise_margin=rank_noise_margin,
    )
    k_energy = _estimate_num_sources_energy(
        eigenvalues_desc=evals_desc,
        energy_ratio=energy_ratio,
    )

    if num_sources is None:
        if source_estimation == "mdl":
            k_est = k_mdl
        elif source_estimation == "energy":
            k_est = k_energy
        elif source_estimation == "rank":
            k_est = k_rank
        else:
            raise ValueError("source_estimation must be 'mdl', 'energy', or 'rank'.")
    else:
        k_est = int(num_sources)

    k_est = int(np.clip(k_est, 0, num_bs_ant - 1))
    us = evecs_desc[:, :k_est] if k_est > 0 else np.empty((num_bs_ant, 0), dtype=np.complex128)
    en = evecs_desc[:, k_est:] if k_est < num_bs_ant else np.empty((num_bs_ant, 0), dtype=np.complex128)

    if bool(compute_user_scores):
        score_per_ant, score_user = _compute_user_scores(
            hi=hi_c,
            us=us,
            en=en,
            reduce_ntn_ant=reduce_ntn_ant,
        )

        if threshold is None:
            # If no threshold is provided, mark top-K users as detected.
            k_users = int(np.clip(k_est, 0, num_ntn))
            detected = np.zeros(num_ntn, dtype=bool)
            if k_users > 0:
                top_idx = np.argsort(score_user)[::-1][:k_users]
                detected[top_idx] = True
            detected_per_ant = np.repeat(detected[:, None], num_ntn_ant, axis=1)
            threshold_used = np.nan
        else:
            threshold_used = float(threshold)
            detected = score_user >= threshold_used
            detected_per_ant = score_per_ant >= threshold_used
    else:
        score_per_ant = np.empty((0, 0), dtype=np.float64)
        score_user = np.empty((0,), dtype=np.float64)
        detected = np.zeros((0,), dtype=bool)
        detected_per_ant = np.zeros((0, 0), dtype=bool)
        threshold_used = np.nan if threshold is None else float(threshold)

    return {
        "detected_mask_user": detected,
        "detected_mask_per_ant": detected_per_ant,
        "score_user": score_user,
        "score_per_ant": score_per_ant,
        "num_sources_est": np.array(k_est, dtype=np.int64),
        "num_sources_mdl": np.array(k_mdl, dtype=np.int64),
        "num_sources_rank": np.array(k_rank, dtype=np.int64),
        "num_sources_eigengap": np.array(k_eigengap, dtype=np.int64),
        "num_sources_energy": np.array(k_energy, dtype=np.int64),
        "source_estimation_used": np.array(str(source_estimation), dtype=object),
        "rank_relative_threshold": np.array(float(rank_relative_threshold), dtype=np.float64),
        "rank_noise_margin": np.array(float(rank_noise_margin), dtype=np.float64),
        "threshold_used": np.array(threshold_used, dtype=np.float64),
        "eigenvalues_desc": evals_desc,
        "covariance": rxx,
    }


def _rotation_matrix_xyz(
    yaw: float,
    pitch: float,
    roll: float,
    rotation_order: Literal["zyx", "zxy", "yxz", "yzx", "xyz", "xzy"] = "zyx",
) -> np.ndarray:
    """Create local->global rotation matrix from yaw/pitch/roll (radians).

    We interpret:
    - yaw   : rotation around global z-axis
    - pitch : rotation around local/global y-axis (per order)
    - roll  : rotation around local/global x-axis (per order)
    """
    cz, sz = np.cos(yaw), np.sin(yaw)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cx, sx = np.cos(roll), np.sin(roll)

    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)

    mats = {"x": rx, "y": ry, "z": rz}
    if len(rotation_order) != 3 or any(ax not in mats for ax in rotation_order):
        raise ValueError("rotation_order must be a permutation of 'x', 'y', 'z' (e.g., 'zyx').")

    r = np.eye(3, dtype=np.float64)
    for ax in rotation_order:
        r = r @ mats[ax]
    return r


def _unit_vector_from_angles(phi_deg: float, theta_deg: float) -> np.ndarray:
    """Global unit vector from spherical angles (phi azimuth, theta zenith)."""
    phi = np.deg2rad(phi_deg)
    theta = np.deg2rad(theta_deg)
    return np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ],
        dtype=np.float64,
    )


def _boresight_global_from_orientation(
    orientation_rad: Tuple[float, float, float],
    *,
    panel_plane: Literal["yz", "xz", "xy"] = "yz",
    rotation_order: Literal["zyx", "zxy", "yxz", "yzx", "xyz", "xzy"] = "zyx",
) -> np.ndarray:
    """Return panel boresight (local normal of panel plane) in global coordinates.

    Panel normal in local coordinates:
    - "yz" -> +x
    - "xz" -> +y
    - "xy" -> +z
    """
    yaw, pitch, roll = orientation_rad
    r_local_to_global = _rotation_matrix_xyz(yaw, pitch, roll, rotation_order=rotation_order)
    if panel_plane == "yz":
        local_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    elif panel_plane == "xz":
        local_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    elif panel_plane == "xy":
        local_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        raise ValueError("panel_plane must be 'yz', 'xz', or 'xy'.")
    return r_local_to_global @ local_axis


def upa_steering_global(
    phi_deg: float,
    theta_deg: float,
    num_rows: int,
    num_cols: int,
    *,
    orientation_rad: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    d_h: float = 0.5,
    d_v: float = 0.5,
    panel_plane: Literal["yz", "xz", "xy"] = "yz",
    phase_sign: Literal[-1, 1] = 1,
    horizontal_sign: Literal[-1, 1] = 1,
    flatten_order: Literal["C", "F"] = "C",
    rotation_order: Literal["zyx", "zxy", "yxz", "yzx", "xyz", "xzy"] = "zyx",
) -> np.ndarray:
    """UPA steering with global angles and panel orientation.

    Inputs phi/theta are interpreted as global spherical angles (degrees):
    - phi   : azimuth, in x-y plane from +x
    - theta : zenith, from +z (0..180)
    - panel_plane:
      - "yz": horizontal axis is local y, vertical axis is local z (default)
      - "xz": horizontal axis is local x, vertical axis is local z
      - "xy": horizontal axis is local x, vertical axis is local y
    - phase_sign:
      - +1 or -1 for convention matching with channel phasor sign.
    - horizontal_sign:
      - +1 or -1 to set horizontal-axis handedness in steering.
    """
    u_global = _unit_vector_from_angles(phi_deg, theta_deg)

    yaw, pitch, roll = orientation_rad
    r_local_to_global = _rotation_matrix_xyz(yaw, pitch, roll, rotation_order=rotation_order)

    # Convert global direction to local panel coordinates.
    u_local = r_local_to_global.T @ u_global
    if panel_plane == "yz":
        u_h = float(u_local[1])
        u_v = float(u_local[2])
    elif panel_plane == "xz":
        u_h = float(u_local[0])
        u_v = float(u_local[2])
    elif panel_plane == "xy":
        u_h = float(u_local[0])
        u_v = float(u_local[1])
    else:
        raise ValueError("panel_plane must be 'yz', 'xz', or 'xy'.")

    rows = np.arange(num_rows, dtype=np.float64)
    cols = np.arange(num_cols, dtype=np.float64)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")

    sgn = 1.0 if int(phase_sign) >= 0 else -1.0
    hsgn = 1.0 if int(horizontal_sign) >= 0 else -1.0
    if flatten_order not in ("C", "F"):
        raise ValueError("flatten_order must be 'C' or 'F'.")

    phase = sgn * 2.0 * np.pi * (d_h * cc * hsgn * u_h + d_v * rr * u_v)
    a = np.exp(1j * phase).reshape(-1, 1, order=flatten_order)
    return a / np.sqrt(float(num_rows * num_cols))


def array_position_steering_global(
    phi_deg: float,
    theta_deg: float,
    array_positions_local: np.ndarray,
    *,
    orientation_rad: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    phase_sign: Literal[-1, 1] = 1,
    rotation_order: Literal["zyx", "zxy", "yxz", "yzx", "xyz", "xzy"] = "zyx",
) -> np.ndarray:
    """Steering vector from wavelength-normalized array element positions.

    Sionna's synthetic-array channel applies ``exp(+j 2 pi p^T u)`` at
    the transmitter. Passing the exact ``PlanarArray.normalized_positions``
    here keeps MUSIC's antenna ordering, axis signs, and element spacing tied
    to the scene instead of reproducing them with a separate UPA convention.
    """
    positions = np.asarray(array_positions_local, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
        raise ValueError(
            "array_positions_local must have shape (num_ant, 3); "
            f"got {positions.shape}."
        )
    if not np.all(np.isfinite(positions)):
        raise ValueError("array_positions_local must contain only finite values.")

    yaw, pitch, roll = orientation_rad
    r_local_to_global = _rotation_matrix_xyz(yaw, pitch, roll, rotation_order=rotation_order)
    positions_global = positions @ r_local_to_global.T
    u_global = _unit_vector_from_angles(phi_deg, theta_deg)
    sgn = 1.0 if int(phase_sign) >= 0 else -1.0
    phase = sgn * 2.0 * np.pi * (positions_global @ u_global)
    a = np.exp(1j * phase).reshape(-1, 1)
    return a / np.sqrt(float(positions.shape[0]))


def _steering_vector_global(
    phi_deg: float,
    theta_deg: float,
    num_rows: int,
    num_cols: int,
    *,
    orientation_rad: Tuple[float, float, float],
    panel_plane: Literal["yz", "xz", "xy"],
    phase_sign: Literal[-1, 1],
    horizontal_sign: Literal[-1, 1],
    flatten_order: Literal["C", "F"],
    rotation_order: Literal["zyx", "zxy", "yxz", "yzx", "xyz", "xzy"],
    array_positions_local: Optional[np.ndarray],
    array_phase_sign: Literal[-1, 1],
) -> np.ndarray:
    if array_positions_local is not None:
        return array_position_steering_global(
            phi_deg,
            theta_deg,
            array_positions_local,
            orientation_rad=orientation_rad,
            phase_sign=array_phase_sign,
            rotation_order=rotation_order,
        )
    return upa_steering_global(
        phi_deg,
        theta_deg,
        num_rows,
        num_cols,
        orientation_rad=orientation_rad,
        panel_plane=panel_plane,
        phase_sign=phase_sign,
        horizontal_sign=horizontal_sign,
        flatten_order=flatten_order,
        rotation_order=rotation_order,
    )


def noise_subspace_from_music_out(music_out: Dict[str, np.ndarray]) -> np.ndarray:
    """Extract noise subspace En from MUSIC output dict."""
    rxx = np.asarray(music_out["covariance"])
    k_est = int(np.asarray(music_out["num_sources_est"]).item())
    m = int(rxx.shape[0])
    k_est = int(np.clip(k_est, 0, m - 1))

    evals, evecs = np.linalg.eigh(rxx)
    idx = np.argsort(np.real(evals))[::-1]
    evecs = evecs[:, idx]
    return evecs[:, k_est:] if k_est < m else np.empty((m, 0), dtype=np.complex128)


def music_top_peaks(
    en: np.ndarray,
    *,
    num_rows: int,
    num_cols: int,
    phi_grid_deg: Iterable[float],
    theta_grid_deg: Iterable[float],
    orientation_rad: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    panel_plane: Literal["yz", "xz", "xy"] = "yz",
    phase_sign: Literal[-1, 1] = 1,
    horizontal_sign: Literal[-1, 1] = 1,
    flatten_order: Literal["C", "F"] = "C",
    forward_only: bool = False,
    forward_cos_min: float = 0.0,
    rotation_order: Literal["zyx", "zxy", "yxz", "yzx", "xyz", "xzy"] = "zyx",
    array_positions_local: Optional[np.ndarray] = None,
    array_phase_sign: Literal[-1, 1] = 1,
    top_n: int = 24,
    min_sep_phi_deg: float = 4.0,
    min_sep_theta_deg: float = 4.0,
    max_peak_correlation: Optional[float] = None,
    refine_peaks: bool = False,
    refine_half_width_deg: float = 1.0,
    refine_maxiter: int = 40,
    local_maxima_only: bool = False,
    max_noise_projection: Optional[float] = None,
) -> List[Tuple[float, float, float, np.ndarray]]:
    """Find dominant MUSIC pseudo-spectrum peaks.

    Returns list of tuples:
        (pseudo_power, phi_deg, theta_deg, steering_vec)
    """
    if en.size == 0 or int(top_n) <= 0:
        return []
    if max_noise_projection is not None and not 0.0 <= float(max_noise_projection) <= 1.0:
        raise ValueError("max_noise_projection must be in [0, 1] or None.")
    if float(min_sep_phi_deg) < 0.0 or float(min_sep_theta_deg) < 0.0:
        raise ValueError("MUSIC peak separation values must be nonnegative.")
    if max_peak_correlation is not None and not 0.0 < float(max_peak_correlation) <= 1.0:
        raise ValueError("max_peak_correlation must be in (0, 1] or None.")
    if float(refine_half_width_deg) < 0.0:
        raise ValueError("refine_half_width_deg must be nonnegative.")
    if int(refine_maxiter) < 1:
        raise ValueError("refine_maxiter must be at least 1.")

    if array_positions_local is not None:
        positions = np.asarray(array_positions_local, dtype=np.float64)
        expected_ant = int(num_rows) * int(num_cols)
        if positions.shape != (expected_ant, 3):
            raise ValueError(
                "array_positions_local must match the MUSIC antenna dimension; "
                f"expected ({expected_ant}, 3), got {positions.shape}."
            )

    phi_values = np.asarray(list(phi_grid_deg), dtype=np.float64)
    theta_values = np.asarray(list(theta_grid_deg), dtype=np.float64)
    if phi_values.size == 0 or theta_values.size == 0:
        return []
    if not np.all(np.isfinite(phi_values)) or not np.all(np.isfinite(theta_values)):
        raise ValueError("MUSIC scan angles must be finite.")
    boresight_global = _boresight_global_from_orientation(
        orientation_rad=orientation_rad,
        panel_plane="yz" if array_positions_local is not None else panel_plane,
        rotation_order=rotation_order,
    )
    fwd_cos = float(np.clip(forward_cos_min, -1.0, 1.0))
    spectrum_floor = 1e-30

    def steering_at(ph: float, th: float) -> np.ndarray:
        return _steering_vector_global(
            float(ph) % 360.0,
            float(np.clip(th, 0.0, 180.0)),
            num_rows,
            num_cols,
            orientation_rad=orientation_rad,
            panel_plane=panel_plane,
            phase_sign=phase_sign,
            horizontal_sign=horizontal_sign,
            flatten_order=flatten_order,
            rotation_order=rotation_order,
            array_positions_local=array_positions_local,
            array_phase_sign=array_phase_sign,
        )

    def is_forward(ph: float, th: float) -> bool:
        if not forward_only:
            return True
        u_global = _unit_vector_from_angles(float(ph) % 360.0, float(np.clip(th, 0.0, 180.0)))
        return float(np.dot(u_global, boresight_global)) >= fwd_cos

    def projection_denominator(a: np.ndarray) -> float:
        return float(np.sum(np.abs(en.conj().T @ a) ** 2))

    def vector_correlation(a: np.ndarray, b: np.ndarray) -> float:
        av = np.asarray(a, dtype=np.complex128).reshape(-1)
        bv = np.asarray(b, dtype=np.complex128).reshape(-1)
        denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
        if denom <= 0.0:
            return 0.0
        return float(np.clip(np.abs(np.vdot(av, bv)) / denom, 0.0, 1.0))

    def refine_peak(
        ph0: float,
        th0: float,
        a0: np.ndarray,
        den0: float,
    ) -> Tuple[float, float, float, np.ndarray, float]:
        half_width = float(refine_half_width_deg)
        if not bool(refine_peaks) or half_width <= 0.0:
            return 1.0 / max(den0, spectrum_floor), ph0, th0, a0, den0

        dtheta_lo = max(-half_width, -float(th0))
        dtheta_hi = min(half_width, 180.0 - float(th0))

        def objective(delta: np.ndarray) -> float:
            ph = (float(ph0) + float(delta[0])) % 360.0
            th = float(np.clip(float(th0) + float(delta[1]), 0.0, 180.0))
            if not is_forward(ph, th):
                return 0.0
            den = projection_denominator(steering_at(ph, th))
            return float(np.log(max(den, spectrum_floor)))

        result = minimize(
            objective,
            x0=np.zeros((2,), dtype=np.float64),
            method="Powell",
            bounds=((-half_width, half_width), (dtheta_lo, dtheta_hi)),
            options={"maxiter": int(refine_maxiter), "xtol": 1e-4, "ftol": 1e-10},
        )
        if not np.all(np.isfinite(result.x)):
            return 1.0 / max(den0, spectrum_floor), ph0, th0, a0, den0

        ph_ref = (float(ph0) + float(result.x[0])) % 360.0
        th_ref = float(np.clip(float(th0) + float(result.x[1]), 0.0, 180.0))
        if not is_forward(ph_ref, th_ref):
            return 1.0 / max(den0, spectrum_floor), ph0, th0, a0, den0
        a_ref = steering_at(ph_ref, th_ref)
        den_ref = projection_denominator(a_ref)
        if not np.isfinite(den_ref) or den_ref > den0:
            return 1.0 / max(den0, spectrum_floor), ph0, th0, a0, den0
        return 1.0 / max(den_ref, spectrum_floor), ph_ref, th_ref, a_ref, den_ref

    ph_mesh, th_mesh = np.meshgrid(phi_values, theta_values)
    ph_rad, th_rad = np.deg2rad(ph_mesh.ravel()), np.deg2rad(th_mesh.ravel())
    directions = np.column_stack((np.sin(th_rad) * np.cos(ph_rad),
                                  np.sin(th_rad) * np.sin(ph_rad), np.cos(th_rad)))
    valid = (th_mesh.ravel() >= 0.0) & (th_mesh.ravel() <= 180.0)
    if forward_only:
        valid &= directions @ boresight_global >= fwd_cos
    denominators = np.full(len(directions), np.inf)
    valid_idx = np.flatnonzero(valid)
    if array_positions_local is not None:
        rotation = _rotation_matrix_xyz(*orientation_rad, rotation_order=rotation_order)
        global_positions = positions @ rotation.T
        # Chunking bounds memory while reusing BLAS for the exact array dictionary.
        for start in range(0, len(valid_idx), 2048):
            ids = valid_idx[start:start + 2048]
            bank = np.exp(1j * float(array_phase_sign) * 2.0 * np.pi *
                          (global_positions @ directions[ids].T)) / np.sqrt(len(positions))
            denominators[ids] = np.sum(np.abs(en.conj().T @ bank) ** 2, axis=0)
    else:
        for idx in valid_idx:
            denominators[idx] = projection_denominator(steering_at(ph_mesh.ravel()[idx], th_mesh.ravel()[idx]))
    if local_maxima_only:
        periodic = (phi_values.size > 2 and np.all(np.diff(phi_values) > 0)
                    and np.allclose(np.diff(phi_values), np.diff(phi_values)[0])
                    and np.isclose(phi_values[-1] - phi_values[0] + np.diff(phi_values)[0], 360.0))
        surface = denominators.reshape(ph_mesh.shape)
        neighbors = minimum_filter(surface, size=3, mode=("nearest", "wrap" if periodic else "nearest"))
        valid &= (surface <= neighbors).ravel()
    candidate_idx = np.flatnonzero(valid & np.isfinite(denominators))
    candidate_idx = candidate_idx[np.argsort(denominators[candidate_idx], kind="stable")]

    peaks: List[Tuple[float, float, float, np.ndarray]] = []
    for idx in candidate_idx:
        ph, th = float(ph_mesh.ravel()[idx]), float(th_mesh.ravel()[idx])
        a, den = steering_at(ph, th), float(denominators[idx])
        if max_peak_correlation is not None and any(
            vector_correlation(a, selected_a) >= float(max_peak_correlation)
            for _, _, _, selected_a in peaks
        ):
            continue

        keep = True
        for _, sph, sth, _ in peaks:
            dphi = abs(((ph - sph + 180.0) % 360.0) - 180.0)
            dth = abs(th - sth)
            if dphi < min_sep_phi_deg and dth < min_sep_theta_deg:
                keep = False
                break
        if not keep:
            continue

        p, ph, th, a, _den = refine_peak(ph, th, a, den)
        if max_noise_projection is not None and _den > float(max_noise_projection):
            continue
        if max_peak_correlation is not None and any(
            vector_correlation(a, selected_a) >= float(max_peak_correlation)
            for _, _, _, selected_a in peaks
        ):
            continue
        peaks.append((p, ph, th, a))
        if len(peaks) >= int(top_n):
            break
    return peaks


def assign_hat_angle_to_user(
    h_user_vec: np.ndarray,
    peaks: List[Tuple[float, float, float, np.ndarray]],
) -> Tuple[float, float]:
    """Assign one (phi,theta) estimate to one user channel vector by best steering match."""
    if len(peaks) == 0:
        return float("nan"), float("nan")

    h = _as_complex_array(h_user_vec).reshape(-1, 1)
    nh = float(np.linalg.norm(h))
    if nh <= 1e-12:
        return float("nan"), float("nan")
    h = h / nh

    best_idx = 0
    best_val = -1.0
    for i, (_p, ph, th, a) in enumerate(peaks):
        val = float(np.abs((a.conj().T @ h).item()))
        if val > best_val:
            best_val = val
            best_idx = i

    _p, ph, th, _a = peaks[best_idx]
    return float(ph), float(th)


def build_steering_bank(
    *,
    num_rows: int,
    num_cols: int,
    phi_grid_deg: Iterable[float],
    theta_grid_deg: Iterable[float],
    orientation_rad: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    panel_plane: Literal["yz", "xz", "xy"] = "yz",
    phase_sign: Literal[-1, 1] = 1,
    horizontal_sign: Literal[-1, 1] = 1,
    flatten_order: Literal["C", "F"] = "C",
    forward_only: bool = False,
    forward_cos_min: float = 0.0,
    rotation_order: Literal["zyx", "zxy", "yxz", "yzx", "xyz", "xzy"] = "zyx",
    array_positions_local: Optional[np.ndarray] = None,
    array_phase_sign: Literal[-1, 1] = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build steering dictionary A and corresponding angle arrays.

    Returns
    -------
    A : np.ndarray
        Shape (M, G), M=num_rows*num_cols.
    phi_bank_deg : np.ndarray
        Shape (G,).
    theta_bank_deg : np.ndarray
        Shape (G,).
    """
    if array_positions_local is not None:
        positions = np.asarray(array_positions_local, dtype=np.float64)
        expected_ant = int(num_rows) * int(num_cols)
        if positions.shape != (expected_ant, 3):
            raise ValueError(
                "array_positions_local must match the steering-bank antenna dimension; "
                f"expected ({expected_ant}, 3), got {positions.shape}."
            )

    boresight_global = _boresight_global_from_orientation(
        orientation_rad=orientation_rad,
        panel_plane="yz" if array_positions_local is not None else panel_plane,
        rotation_order=rotation_order,
    )
    fwd_cos = float(np.clip(forward_cos_min, -1.0, 1.0))

    vecs: List[np.ndarray] = []
    phi_list: List[float] = []
    theta_list: List[float] = []

    for th in theta_grid_deg:
        for ph in phi_grid_deg:
            phf = float(ph)
            thf = float(th)
            if forward_only:
                u_global = _unit_vector_from_angles(phf, thf)
                if float(np.dot(u_global, boresight_global)) < fwd_cos:
                    continue

            a = _steering_vector_global(
                phf,
                thf,
                num_rows,
                num_cols,
                orientation_rad=orientation_rad,
                panel_plane=panel_plane,
                phase_sign=phase_sign,
                horizontal_sign=horizontal_sign,
                flatten_order=flatten_order,
                rotation_order=rotation_order,
                array_positions_local=array_positions_local,
                array_phase_sign=array_phase_sign,
            )
            vecs.append(a)
            phi_list.append(phf)
            theta_list.append(thf)

    m = int(num_rows * num_cols)
    if len(vecs) == 0:
        return (
            np.empty((m, 0), dtype=np.complex128),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )

    a_bank = np.hstack(vecs)
    phi_bank = np.asarray(phi_list, dtype=np.float64)
    theta_bank = np.asarray(theta_list, dtype=np.float64)
    return a_bank, phi_bank, theta_bank


def estimate_angle_from_channel_scan(
    h_user_vec: np.ndarray,
    a_bank: np.ndarray,
    phi_bank_deg: np.ndarray,
    theta_bank_deg: np.ndarray,
    *,
    scan_mode: Literal["complex", "phase_only"] = "complex",
    return_index: bool = False,
) -> Tuple[float, float, float] | Tuple[float, float, float, int]:
    """Estimate one user angle by scanning steering dictionary against channel vector.

    Score is |a^H h| with normalized h and normalized steering columns.
    Returns (phi_hat_deg, theta_hat_deg, score_max) by default.
    If `return_index=True`, also returns the steering-bank index.
    """
    if a_bank.size == 0 or phi_bank_deg.size == 0:
        if return_index:
            return float("nan"), float("nan"), float("nan"), -1
        return float("nan"), float("nan"), float("nan")

    a = _as_complex_array(a_bank)
    if a.ndim != 2:
        raise ValueError("a_bank must be a 2D matrix with shape (num_ant, num_grid).")

    h = _as_complex_array(h_user_vec).reshape(-1, 1)
    if a.shape[0] != h.shape[0]:
        raise ValueError(
            f"Dimension mismatch: steering has {a.shape[0]} antennas, "
            f"but h_user_vec has {h.shape[0]} elements."
        )
    if a.shape[1] != int(np.asarray(phi_bank_deg).size) or a.shape[1] != int(np.asarray(theta_bank_deg).size):
        raise ValueError("a_bank column count must match phi_bank_deg/theta_bank_deg length.")

    nh = float(np.linalg.norm(h))
    if nh <= 1e-12:
        if return_index:
            return float("nan"), float("nan"), float("nan"), -1
        return float("nan"), float("nan"), float("nan")
    h = h / nh

    if scan_mode not in ("complex", "phase_only"):
        raise ValueError("scan_mode must be 'complex' or 'phase_only'.")

    if scan_mode == "complex":
        # A columns are normalized by construction.
        corrs = np.abs((a.conj().T @ h).reshape(-1))
    else:
        # Phase-only matching is robust when element-pattern amplitudes distort h.
        a_phase = np.exp(1j * np.angle(a))
        h_phase = np.exp(1j * np.angle(h))
        corrs = np.abs((a_phase.conj().T @ h_phase).reshape(-1))

    if corrs.size == 0:
        if return_index:
            return float("nan"), float("nan"), float("nan"), -1
        return float("nan"), float("nan"), float("nan")
    idx = int(np.argmax(corrs))
    if return_index:
        return float(phi_bank_deg[idx]), float(theta_bank_deg[idx]), float(corrs[idx]), idx
    return float(phi_bank_deg[idx]), float(theta_bank_deg[idx]), float(corrs[idx])


def aod_to_aoa_reverse(phi_t_deg: np.ndarray, theta_t_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert downlink AOD (TX->RX) to reversed-link AOA reference (RX<-TX)."""
    phi = (np.asarray(phi_t_deg, dtype=np.float64) + 180.0) % 360.0
    theta = 180.0 - np.asarray(theta_t_deg, dtype=np.float64)
    return phi, theta


def zenith_to_elevation_deg(theta_zenith_deg: np.ndarray) -> np.ndarray:
    """Convert zenith angle (0=+z, 90=horizontal) to elevation (0=horizontal)."""
    return 90.0 - np.asarray(theta_zenith_deg, dtype=np.float64)


def elevation_to_zenith_deg(theta_elevation_deg: np.ndarray) -> np.ndarray:
    """Convert elevation angle (0=horizontal) to zenith (0=+z, 90=horizontal)."""
    return 90.0 - np.asarray(theta_elevation_deg, dtype=np.float64)


def sector_index_from_tx_index(tx_index: np.ndarray, nsect: int) -> np.ndarray:
    """Map flat TX index to sector index."""
    if int(nsect) <= 0:
        raise ValueError("nsect must be > 0.")
    return np.asarray(tx_index, dtype=np.int64) % int(nsect)


def sector_yaw_offset_deg(sector_index: np.ndarray, nsect: int) -> np.ndarray:
    """Sector azimuth offsets in degrees for equally-spaced sectors."""
    if int(nsect) <= 0:
        raise ValueError("nsect must be > 0.")
    s = np.asarray(sector_index, dtype=np.float64)
    return (360.0 * s / float(nsect)) % 360.0


def sector_local_aod_to_global(
    phi_local_deg: np.ndarray,
    *,
    sector_index: np.ndarray,
    nsect: int,
    yaw_offset_deg: float = 0.0,
) -> np.ndarray:
    """Convert per-sector local azimuth to global azimuth.

    For 3-sector deployment, this applies +0/+120/+240 deg (plus optional offset).
    """
    phi_loc = np.asarray(phi_local_deg, dtype=np.float64)
    sec = np.asarray(sector_index, dtype=np.float64)
    if phi_loc.shape != sec.shape:
        raise ValueError("phi_local_deg and sector_index must have the same shape.")

    sec_off = sector_yaw_offset_deg(sec, int(nsect))
    return (phi_loc + sec_off + float(yaw_offset_deg)) % 360.0


def sionna_aod_to_uplink_aoa(
    phi_t_deg: np.ndarray,
    theta_t_deg: np.ndarray,
    *,
    phi_is_sector_local: bool = False,
    sector_index: Optional[np.ndarray] = None,
    nsect: Optional[int] = None,
    yaw_offset_deg: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert Sionna downlink AOD reference to uplink AOA at BS.

    - If phi_t is sector-local, set phi_is_sector_local=True and provide
      sector_index + nsect to first convert it to global azimuth.
    - Zenith angle conversion is always theta_aoa = 180 - theta_t.
    """
    phi_t = np.asarray(phi_t_deg, dtype=np.float64)
    theta_t = np.asarray(theta_t_deg, dtype=np.float64)
    if phi_t.shape != theta_t.shape:
        raise ValueError("phi_t_deg and theta_t_deg must have the same shape.")

    if phi_is_sector_local:
        if sector_index is None or nsect is None:
            raise ValueError(
                "When phi_is_sector_local=True, sector_index and nsect are required."
            )
        phi_global = sector_local_aod_to_global(
            phi_t,
            sector_index=np.asarray(sector_index),
            nsect=int(nsect),
            yaw_offset_deg=float(yaw_offset_deg),
        )
    else:
        phi_global = phi_t % 360.0

    return aod_to_aoa_reverse(phi_global, theta_t)


def wrapped_angle_diff_deg(a_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    """Wrapped angular error a-b in [-180,180)."""
    return ((np.asarray(a_deg) - np.asarray(b_deg) + 180.0) % 360.0) - 180.0


def angle_error_metrics(
    phi_true_deg: np.ndarray,
    theta_true_deg: np.ndarray,
    phi_hat_deg: np.ndarray,
    theta_hat_deg: np.ndarray,
    *,
    reference_mode: Literal["aod", "aoa_reverse"] = "aoa_reverse",
) -> Dict[str, float]:
    """Compute MAE/STD metrics for angle error under a reference convention."""
    phi_true = np.asarray(phi_true_deg, dtype=np.float64)
    theta_true = np.asarray(theta_true_deg, dtype=np.float64)
    phi_hat = np.asarray(phi_hat_deg, dtype=np.float64)
    theta_hat = np.asarray(theta_hat_deg, dtype=np.float64)

    mask = ~(np.isnan(phi_true) | np.isnan(theta_true) | np.isnan(phi_hat) | np.isnan(theta_hat))
    if np.count_nonzero(mask) == 0:
        return {
            "count": 0.0,
            "phi_mae_deg": float("nan"),
            "phi_std_deg": float("nan"),
            "theta_mae_deg": float("nan"),
            "theta_std_deg": float("nan"),
        }

    phi_true = phi_true[mask]
    theta_true = theta_true[mask]
    phi_hat = phi_hat[mask]
    theta_hat = theta_hat[mask]

    if reference_mode == "aod":
        phi_ref, theta_ref = phi_true, theta_true
    elif reference_mode == "aoa_reverse":
        phi_ref, theta_ref = aod_to_aoa_reverse(phi_true, theta_true)
    else:
        raise ValueError("reference_mode must be 'aod' or 'aoa_reverse'.")

    dphi = wrapped_angle_diff_deg(phi_hat, phi_ref)
    dtheta = theta_hat - theta_ref

    return {
        "count": float(phi_ref.shape[0]),
        "phi_mae_deg": float(np.mean(np.abs(dphi))),
        "phi_std_deg": float(np.std(dphi)),
        "theta_mae_deg": float(np.mean(np.abs(dtheta))),
        "theta_std_deg": float(np.std(dtheta)),
    }


def detect_music_from_hi(*args: Any, **kwargs: Any) -> Dict[str, np.ndarray]:
    """Generic alias of :func:`detect_ntn_music_from_hi`."""
    return detect_ntn_music_from_hi(*args, **kwargs)


def signal_noise_subspaces_from_music_out(
    music_out: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract signal/noise subspaces `(Us, En)` from MUSIC output."""
    rxx = np.asarray(music_out["covariance"], dtype=np.complex128)
    k_est = int(np.asarray(music_out["num_sources_est"]).item())
    m = int(rxx.shape[0])
    k_est = int(np.clip(k_est, 0, m - 1))

    evals, evecs = np.linalg.eigh(rxx)
    idx = np.argsort(np.real(evals))[::-1]
    evecs = evecs[:, idx]
    us = evecs[:, :k_est] if k_est > 0 else np.empty((m, 0), dtype=np.complex128)
    en = evecs[:, k_est:] if k_est < m else np.empty((m, 0), dtype=np.complex128)
    return us, en


def estimate_noise_power_from_music_out(
    music_out: Dict[str, np.ndarray],
    *,
    eps: float = 1e-12,
) -> float:
    """Estimate the residual noise power from the discarded eigenvalues."""
    evals_desc = np.asarray(music_out.get("eigenvalues_desc", []), dtype=np.float64).reshape(-1)
    if evals_desc.size == 0:
        return 0.0
    k_est = int(np.asarray(music_out["num_sources_est"]).item())
    k_est = int(np.clip(k_est, 0, evals_desc.size))
    if k_est >= evals_desc.size:
        return 0.0
    tail = evals_desc[k_est:]
    tail = tail[np.isfinite(tail)]
    if tail.size == 0:
        return 0.0
    return float(max(np.mean(tail), 0.0))


def estimate_noncoh_gains_from_covariance(
    covariance: np.ndarray,
    steering_vectors: np.ndarray,
    *,
    noise_power: float = 0.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Estimate noncoherent gains `g_k` in `R ~= sum_k g_k u_k u_k^H + sigma^2 I`."""
    rxx = np.asarray(covariance, dtype=np.complex128)
    u_arr = np.asarray(steering_vectors, dtype=np.complex128)

    if rxx.ndim != 2 or rxx.shape[0] != rxx.shape[1]:
        raise ValueError("covariance must be a square matrix.")
    m = int(rxx.shape[0])
    if u_arr.ndim == 1:
        if u_arr.size == 0:
            u_arr = np.empty((0, m), dtype=np.complex128)
        else:
            u_arr = u_arr.reshape(1, -1)
    if u_arr.ndim != 2 or u_arr.shape[1] != m:
        raise ValueError(f"steering_vectors must have shape (K, {m}); got {u_arr.shape}.")
    if u_arr.shape[0] == 0:
        return np.empty((0,), dtype=np.float64)

    u_norm = np.linalg.norm(u_arr, axis=1, keepdims=True)
    u_unit = u_arr / np.maximum(u_norm, float(eps))

    r_sig = rxx - float(max(noise_power, 0.0)) * np.eye(m, dtype=np.complex128)
    basis = np.column_stack(
        [np.outer(u_unit[k], np.conjugate(u_unit[k])).reshape(-1) for k in range(u_unit.shape[0])]
    )
    scale = float(np.linalg.norm(r_sig))
    if scale == 0.0:
        return np.zeros(u_unit.shape[0], dtype=np.float64)
    if not np.isfinite(scale):
        raise ValueError("Covariance must contain only finite values.")
    design = np.vstack((basis.real, basis.imag))
    target = np.concatenate((r_sig.real.ravel(), r_sig.imag.ravel())) / scale
    g_hat, _ = nnls(design, target, maxiter=max(100, 10 * u_unit.shape[0]))
    g_hat *= scale
    return np.asarray(g_hat, dtype=np.float64)



def refine_music_covariance(
    covariance: np.ndarray,
    peaks: List[Tuple[float, float, float, np.ndarray]],
    *,
    array_positions_local: np.ndarray,
    orientation_rad: Tuple[float, float, float],
    en: np.ndarray,
    noise_power: float,
    array_phase_sign: int = 1,
    rotation_order: str = "zyx",
    forward_cos_min: float = 0.0,
    max_direction_step: float = 0.1,
    maxiter: int = 60,
) -> Tuple[List[Tuple[float, float, float, np.ndarray]], np.ndarray, Dict[str, float]]:
    """Refine MUSIC directions by covariance fitting with nonnegative gains.

    Optimize local planar direction cosines. Each objective evaluation solves
    gains first; the envelope gradient updates directions without using truth.
    Reject infeasible or worse solutions, including boundary roundoff.
    """
    if not 0.0 <= forward_cos_min < 1.0:
        raise ValueError("Refinement requires 0 <= forward_cos_min < 1.")
    if not np.isfinite(max_direction_step) or max_direction_step <= 0 or int(maxiter) < 1:
        raise ValueError("Refinement step and iteration limit must be positive.")
    positions = np.asarray(array_positions_local, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or not np.all(np.isfinite(positions)):
        raise ValueError("Expected finite array positions of shape (M, 3).")
    positions = positions - positions.mean(axis=0)
    if not np.allclose(positions[:, 0], 0.0, atol=1e-10):
        raise ValueError("Refinement currently supports local yz planar arrays.")
    if not peaks:
        return [], np.empty(0), {"before": 0.0, "after": 0.0}
    rxx = np.asarray(covariance, dtype=np.complex128)
    if rxx.shape != (len(positions), len(positions)):
        raise ValueError("Covariance and array dimensions differ.")
    signal = (rxx + rxx.conj().T) / 2 - noise_power * np.eye(len(positions))
    scale = float(np.linalg.norm(signal))
    if not np.isfinite(scale):
        raise ValueError("Covariance must be finite.")
    if scale == 0.0:
        return peaks, np.zeros(len(peaks)), {"before": 0.0, "after": 0.0}
    target = signal / scale
    rotation = _rotation_matrix_xyz(*orientation_rad, rotation_order=rotation_order)
    global_dirs = np.array([_unit_vector_from_angles(ph, th) for _, ph, th, _ in peaks])
    q0 = (global_dirs @ rotation)[:, 1:]
    radius2 = 1.0 - forward_cos_min ** 2
    radius = np.sqrt(radius2)
    axes = float(array_phase_sign) * 2.0 * np.pi * positions[:, 1:]
    cached_x, cached_result = None, None

    def evaluate(x):
        nonlocal cached_x, cached_result
        if cached_x is not None and np.array_equal(cached_x, x):
            return cached_result
        q = x.reshape(-1, 2)
        u = np.exp(1j * (q @ axes.T)) / np.sqrt(len(positions))
        gains = estimate_noncoh_gains_from_covariance(target, u)
        residual = np.einsum("k,ka,kb->ab", gains, u, u.conj()) - target
        value = float(np.vdot(residual, residual).real)
        ea = (residual @ u.T).T
        derivatives = 1j * u[:, :, None] * axes[None, :, :]
        gradient = 4 * gains[:, None] * np.real(np.einsum("kmd,km->kd", derivatives.conj(), ea))
        cached_x = x.copy()
        cached_result = value, gradient.ravel(), u, gains
        return cached_result

    x0 = q0.ravel()
    initial = evaluate(x0)
    bounds = list(zip(np.maximum(x0 - max_direction_step, -radius),
                      np.minimum(x0 + max_direction_step, radius)))

    def disk_jacobian(x):
        jac = np.zeros((len(peaks), len(x)))
        for k in range(len(peaks)):
            jac[k, 2*k:2*k+2] = -2 * x[2*k:2*k+2]
        return jac

    result = minimize(
        lambda x: evaluate(x)[:2], x0, jac=True, method="SLSQP", bounds=bounds,
        constraints=[{"type": "ineq",
                      "fun": lambda x: radius2 - np.sum(x.reshape(-1, 2) ** 2, axis=1),
                      "jac": disk_jacobian}],
        options={"maxiter": int(maxiter), "ftol": 1e-12},
    )
    chosen = x0
    if np.all(np.isfinite(result.x)):
        feasible = np.all(np.sum(result.x.reshape(-1, 2) ** 2, axis=1) <= radius2 + 1e-10)
        if feasible and evaluate(result.x)[0] < initial[0]:
            chosen = result.x
    q = chosen.reshape(-1, 2).copy()
    lengths = np.linalg.norm(q, axis=1)
    outside = lengths > radius
    q[outside] *= (radius / lengths[outside])[:, None]
    final = evaluate(q.ravel())
    if final[0] > initial[0]:
        q, final = q0, initial
    directions = np.column_stack((np.sqrt(np.maximum(1 - np.sum(q*q, axis=1), 0)), q)) @ rotation.T
    refined = []
    for d, u in zip(directions, final[2]):
        ph = float(np.rad2deg(np.arctan2(d[1], d[0])) % 360)
        th = float(np.rad2deg(np.arccos(np.clip(d[2], -1, 1))))
        den = float(np.sum(np.abs(en.conj().T @ u) ** 2))
        refined.append((1 / max(den, 1e-30), ph, th, u.reshape(-1, 1)))
    return refined, final[3] * scale, {
        "before": float(np.sqrt(initial[0])), "after": float(np.sqrt(final[0]))
    }

def _parse_manifold_label(label: str) -> Tuple[str, str, int]:
    """Parse labels like 'yz:+1' into canonical `(key, plane, sign)`."""
    txt = str(label).strip().lower().replace(" ", "")
    if ":" in txt:
        plane, sgn = txt.split(":", 1)
    else:
        plane, sgn = txt, "+1"

    if plane not in ("yz", "xz", "xy"):
        raise ValueError(f"Invalid manifold plane: {plane}. Use yz/xz/xy.")
    if sgn in ("+1", "1", "+"):
        sign = 1
    elif sgn in ("-1", "-"):
        sign = -1
    else:
        raise ValueError(f"Invalid manifold sign: {sgn}. Use +1/-1.")

    return f"{plane}:{'+' if sign > 0 else '-'}1", plane, sign


def _build_manifold_candidates(
    manifold_mode: str,
    manifold_auto_list: Iterable[str],
) -> List[Tuple[str, str, int]]:
    src = list(manifold_auto_list) if str(manifold_mode).lower() == "auto" else [manifold_mode]
    out: List[Tuple[str, str, int]] = []
    seen = set()
    for x in src:
        key, plane, sign = _parse_manifold_label(str(x))
        if key in seen:
            continue
        seen.add(key)
        out.append((key, plane, sign))
    if len(out) == 0:
        out = [("yz:+1", "yz", 1)]
    return out


def _build_flatten_candidates(flatten_mode: str, flatten_mode_list: Iterable[str]) -> List[str]:
    src = list(flatten_mode_list) if str(flatten_mode).lower() == "auto" else [flatten_mode]
    out: List[str] = []
    seen = set()
    for x in src:
        key = str(x).strip().upper()
        if key not in ("C", "F"):
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    if len(out) == 0:
        out = ["C"]
    return out


def _build_scan_candidates(scan_mode: str, scan_mode_list: Iterable[str]) -> List[str]:
    src = list(scan_mode_list) if str(scan_mode).lower() == "auto" else [scan_mode]
    out: List[str] = []
    seen = set()
    for x in src:
        key = str(x).strip().lower()
        if key not in ("complex", "phase_only"):
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    if len(out) == 0:
        out = ["complex"]
    return out


def _build_phi_offset_candidates(phi_offset_mode: Any, phi_offset_list: Iterable[float]) -> List[float]:
    src = list(phi_offset_list) if str(phi_offset_mode).lower() == "auto" else [phi_offset_mode]
    out: List[float] = []
    seen = set()
    for x in src:
        try:
            v = float(x)
        except Exception:
            continue
        key = float(np.round(v % 360.0, 1))
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    if len(out) == 0:
        out = [0.0]
    return out


def build_true_pair_map(
    pair_rx_idx: np.ndarray,
    pair_t_idx: np.ndarray,
    pair_phi_deg: np.ndarray,
    pair_theta_deg: np.ndarray,
    pair_bs_idx: np.ndarray,
    pair_sector_idx: np.ndarray,
) -> Dict[Tuple[int, int], Tuple[float, float, int, int]]:
    """Build a standard pair->(phi,theta,bs,sec) map from arrays."""
    return {
        (int(rx), int(t)): (float(phi), float(theta), int(bs), int(sec))
        for rx, t, phi, theta, bs, sec in zip(
            pair_rx_idx,
            pair_t_idx,
            pair_phi_deg,
            pair_theta_deg,
            pair_bs_idx,
            pair_sector_idx,
        )
    }


def _canonical_rx_tx_paths(arr: np.ndarray, num_tx: int) -> np.ndarray:
    """Reorder path tensors to shape [num_rx, num_tx, num_paths_flat]."""
    a = np.squeeze(_to_numpy(arr))
    if a.ndim < 2:
        raise ValueError(f"Need >=2 dims for [rx,tx], got {a.shape}")
    tx_axes = [i for i, s in enumerate(a.shape) if s == num_tx]
    if not tx_axes:
        raise ValueError(f"Cannot find tx-axis size={num_tx} in {a.shape}")
    tx_axis = tx_axes[1] if (tx_axes[0] == 0 and len(tx_axes) > 1) else tx_axes[0]
    a = np.moveaxis(a, [0, tx_axis], [0, 1])
    return a.reshape(a.shape[0], a.shape[1], -1)


def _pair_and_path_power(
    cir: np.ndarray,
    num_tx: int,
    num_paths: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return pair power and per-path power with stable [rx, tx, path] indexing."""
    h = _to_numpy(cir)
    tx_axes = [i for i, s in enumerate(h.shape) if s == num_tx]
    if not tx_axes:
        raise ValueError(f"Cannot find tx-axis size={num_tx} in cir shape={h.shape}")
    tx_axis = tx_axes[1] if (tx_axes[0] == 0 and len(tx_axes) > 1) else tx_axes[0]

    h_rt = np.moveaxis(h, [0, tx_axis], [0, 1])
    pwr = np.abs(h_rt) ** 2
    pair_power = pwr.sum(axis=tuple(range(2, pwr.ndim)))

    rest = h_rt.shape[2:]
    path_cands = [i for i, s in enumerate(rest) if s == num_paths]
    if not path_cands:
        if num_paths == 1:
            return pair_power, pair_power[..., None]
        raise ValueError(f"Cannot align path axis: num_paths={num_paths}, cir_rest={rest}")

    pick = path_cands[-1]
    if len(path_cands) > 1 and pick == len(rest) - 1:
        pick = path_cands[-2]

    path_axis = 2 + pick
    sum_axes = tuple(ax for ax in range(2, h_rt.ndim) if ax != path_axis)
    path_power = pwr.sum(axis=sum_axes)
    return pair_power, path_power


def build_ntn_truth_from_paths(
    paths_ntn: Any,
    cir_ntn: np.ndarray,
    *,
    num_tx_total: int,
    nsect: int,
    sionna_phi_is_global: bool = True,
    channel_eps: float = 1e-18,
    angle_eps_deg: float = 1e-9,
) -> Dict[str, Any]:
    """Extract strongest-path NTN truth pairs from Sionna path outputs.

    For each valid `(rx, tx)` pair, the function keeps the strongest path and
    records its AOD angles. The resulting map is used to evaluate MUSIC angle
    estimates per macro simulation.
    """
    if not hasattr(paths_ntn, "phi_t") or not hasattr(paths_ntn, "theta_t"):
        raise AttributeError("paths_ntn must provide phi_t and theta_t.")

    phi_probe = np.squeeze(_to_numpy(paths_ntn.phi_t))
    num_tx_eff = int(num_tx_total)
    if num_tx_eff not in phi_probe.shape:
        if phi_probe.ndim >= 2:
            tx_dims = [int(s) for s in phi_probe.shape if int(s) > 1]
            if num_tx_eff in tx_dims:
                pass
            elif len(tx_dims) > 0:
                num_tx_eff = int(max(tx_dims))

    phi_t = _canonical_rx_tx_paths(paths_ntn.phi_t, num_tx_eff)
    theta_t = _canonical_rx_tx_paths(paths_ntn.theta_t, num_tx_eff)
    phi_t_raw_deg = np.mod(np.rad2deg(phi_t), 360.0)
    theta_t_deg = np.rad2deg(theta_t)
    tx_sector_map = sector_index_from_tx_index(np.arange(num_tx_eff), int(nsect))

    if sionna_phi_is_global:
        phi_t_deg = phi_t_raw_deg
    else:
        phi_t_deg = sector_local_aod_to_global(
            phi_t_raw_deg,
            sector_index=np.broadcast_to(tx_sector_map[None, :, None], phi_t_raw_deg.shape),
            nsect=int(nsect),
        )

    pair_power_ntn, path_power_ntn = _pair_and_path_power(cir_ntn, num_tx_eff, phi_t_deg.shape[2])
    angle_nonzero_pair_mask = np.any(
        (np.abs(phi_t_deg) > float(angle_eps_deg)) | (np.abs(theta_t_deg) > float(angle_eps_deg)),
        axis=2,
    )
    channel_nonzero_pair_mask = pair_power_ntn > float(channel_eps)
    valid_pair_mask = angle_nonzero_pair_mask & channel_nonzero_pair_mask

    valid_pairs = np.argwhere(valid_pair_mask)
    if valid_pairs.size > 0:
        rx_idx_valid = valid_pairs[:, 0]
        t_idx_valid = valid_pairs[:, 1]
        pair_best_path = np.argmax(path_power_ntn[rx_idx_valid, t_idx_valid, :], axis=1)
        pair_phi_t_deg = phi_t_deg[rx_idx_valid, t_idx_valid, pair_best_path]
        pair_theta_t_deg = theta_t_deg[rx_idx_valid, t_idx_valid, pair_best_path]
        pair_bs_idx = t_idx_valid // int(nsect)
        pair_sector_idx = t_idx_valid % int(nsect)
    else:
        rx_idx_valid = np.empty((0,), dtype=int)
        t_idx_valid = np.empty((0,), dtype=int)
        pair_best_path = np.empty((0,), dtype=int)
        pair_phi_t_deg = np.empty((0,), dtype=float)
        pair_theta_t_deg = np.empty((0,), dtype=float)
        pair_bs_idx = np.empty((0,), dtype=int)
        pair_sector_idx = np.empty((0,), dtype=int)

    return {
        "pair_rx_idx": rx_idx_valid,
        "pair_t_idx": t_idx_valid,
        "pair_bs_idx": pair_bs_idx,
        "pair_sector_idx": pair_sector_idx,
        "pair_path_idx": pair_best_path,
        "pair_phi_t_deg": pair_phi_t_deg,
        "pair_theta_t_deg": pair_theta_t_deg,
        "pair_map": build_true_pair_map(
            rx_idx_valid,
            t_idx_valid,
            pair_phi_t_deg,
            pair_theta_t_deg,
            pair_bs_idx,
            pair_sector_idx,
        ),
        "valid_pair_count": int(valid_pairs.shape[0]),
    }


def _vector_error_metrics(
    x_hat: np.ndarray,
    x_true: np.ndarray,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Compute reconstruction metrics for one tensor or one flattened vector.

    Metric notes:
    - `NRMSE` is `||x_hat - x_true|| / ||x_true||`; lower is better.
    - `CosSim` is normalized complex-vector similarity in `[0, 1]`; higher is better.
    - `|h| MAE` is mean absolute error on channel magnitude.
    - `PowerRatio` is `10*log10(sum|x_hat|^2 / sum|x_true|^2)`; `0 dB` is ideal.
    """
    xh = np.asarray(x_hat, dtype=np.complex128).reshape(-1)
    xt = np.asarray(x_true, dtype=np.complex128).reshape(-1)
    if xh.size == 0:
        return {
            "count": 0.0,
            "nrmse": float("nan"),
            "cos_sim": float("nan"),
            "mag_mae": float("nan"),
            "power_ratio_db": float("nan"),
        }

    err = xh - xt
    nrmse = np.linalg.norm(err) / (np.linalg.norm(xt) + float(eps))
    cos_sim = np.abs(np.vdot(xh, xt)) / (np.linalg.norm(xh) * np.linalg.norm(xt) + float(eps))
    mag_mae = np.mean(np.abs(np.abs(xh) - np.abs(xt)))
    p_hat = np.sum(np.abs(xh) ** 2)
    p_true = np.sum(np.abs(xt) ** 2)
    power_ratio_db = 10.0 * np.log10((p_hat + float(eps)) / (p_true + float(eps)))
    return {
        "count": float(xh.size),
        "nrmse": float(nrmse),
        "cos_sim": float(cos_sim),
        "mag_mae": float(mag_mae),
        "power_ratio_db": float(power_ratio_db),
    }


def summarize_ntn_music_quality(
    h_ntn_all: np.ndarray,
    ntn_music_out: Dict[str, Any],
    ntn_true_map: Optional[Dict[Tuple[int, int], Tuple[float, float, int, int]]] = None,
    *,
    theta_display_mode: Literal["zenith", "elevation"] = "elevation",
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """Summarize MUSIC angle and channel reconstruction quality.

    Returned groups:
    - `angle_metrics`: matched detected-pair angle error against strongest-path truth.
    - `detected_subset_metrics`: compare the full estimated tensor `h_hat_all` against
      the true tensor `h_ntn_all`, restricted to detected NTN users only.
    - `detected_pairs_summary`: compare each detected `(rx, tx, rx_ant)` channel vector
      independently, then report mean/median statistics across detected pairs.

    This matches the interpretation used in the earlier notebook validation cells:
    - `Detected-subset tensor metrics` describe aggregate tensor reconstruction quality.
    - `Detected pairs summary` describes per-link reconstruction quality.
    """
    h_ntn_all_np = np.asarray(h_ntn_all, dtype=np.complex128)
    h_ntn_hat_all_np = np.asarray(
        ntn_music_out.get("h_hat_all", np.zeros_like(h_ntn_all_np)),
        dtype=np.complex128,
    )
    if h_ntn_hat_all_np.shape != h_ntn_all_np.shape:
        raise ValueError(
            f"Shape mismatch: h_hat_all{h_ntn_hat_all_np.shape} vs h_ntn_all{h_ntn_all_np.shape}"
        )

    true_map = ntn_true_map if ntn_true_map is not None else {}
    pair_hat = ntn_music_out.get("pair_hat", {})
    det_pairs = sorted(pair_hat.keys(), key=lambda k: (k[1], k[0]))
    matched_pairs = [k for k in det_pairs if k in true_map]

    if len(matched_pairs) > 0:
        phi_true_match = np.array([true_map[k][0] for k in matched_pairs], dtype=float)
        theta_true_match = np.array([true_map[k][1] for k in matched_pairs], dtype=float)
        phi_hat_match = np.array([pair_hat[k]["phi_hat_deg"] for k in matched_pairs], dtype=float)
        theta_hat_match = np.array([pair_hat[k]["theta_hat_deg"] for k in matched_pairs], dtype=float)

        theta_mode = str(theta_display_mode).strip().lower()
        if theta_mode == "elevation":
            theta_true_metric = zenith_to_elevation_deg(theta_true_match)
            theta_hat_metric = zenith_to_elevation_deg(theta_hat_match)
            theta_label = "elev"
        elif theta_mode == "zenith":
            theta_true_metric = theta_true_match
            theta_hat_metric = theta_hat_match
            theta_label = "theta"
        else:
            raise ValueError("theta_display_mode must be 'zenith' or 'elevation'.")

        angle_metrics = angle_error_metrics(
            phi_true_match,
            theta_true_metric,
            phi_hat_match,
            theta_hat_metric,
            reference_mode="aod",
        )
        angle_metrics["theta_metric_name"] = theta_label
        angle_metrics["matched_pairs"] = int(len(matched_pairs))
        angle_metrics["elev_mae_deg"] = (
            float(angle_metrics["theta_mae_deg"]) if theta_label == "elev" else float("nan")
        )
    else:
        angle_metrics = {
            "count": 0.0,
            "matched_pairs": 0,
            "phi_mae_deg": float("nan"),
            "phi_std_deg": float("nan"),
            "theta_mae_deg": float("nan"),
            "theta_std_deg": float("nan"),
            "theta_metric_name": "elev" if str(theta_display_mode).lower() == "elevation" else "theta",
            "elev_mae_deg": float("nan"),
        }

    pair_rx = np.asarray(ntn_music_out.get("pair_rx_idx", []), dtype=int)
    if pair_rx.size == 0:
        det_rx_idx = np.asarray(ntn_music_out.get("detected_rx_indices_unique", []), dtype=int)
    else:
        det_rx_idx = np.unique(pair_rx)
    det_rx_idx = det_rx_idx[(det_rx_idx >= 0) & (det_rx_idx < h_ntn_all_np.shape[0])]
    det_rx_idx = np.unique(det_rx_idx)

    if det_rx_idx.size == 0:
        detected_subset_metrics = {
            "count": 0.0,
            "nrmse": float("nan"),
            "cos_sim": float("nan"),
            "mag_mae": float("nan"),
            "power_ratio_db": float("nan"),
        }
    else:
        detected_subset_metrics = _vector_error_metrics(
            h_ntn_hat_all_np[det_rx_idx, :, :, :],
            h_ntn_all_np[det_rx_idx, :, :, :],
            eps=eps,
        )

    pair_t = np.asarray(ntn_music_out.get("pair_t_idx", []), dtype=int)
    if "pair_rx_ant_idx" in ntn_music_out:
        pair_ant = np.asarray(ntn_music_out.get("pair_rx_ant_idx", []), dtype=int)
    else:
        pair_ant = np.zeros((pair_rx.size,), dtype=int)

    valid = (
        (pair_rx >= 0)
        & (pair_rx < h_ntn_all_np.shape[0])
        & (pair_t >= 0)
        & (pair_t < h_ntn_all_np.shape[2])
        & (pair_ant >= 0)
        & (pair_ant < h_ntn_all_np.shape[1])
    )

    pair_nrmse: List[float] = []
    pair_cos: List[float] = []
    for rx_i, ant_i, t_i in zip(pair_rx[valid], pair_ant[valid], pair_t[valid]):
        h_hat_vec = h_ntn_hat_all_np[int(rx_i), int(ant_i), int(t_i), :]
        h_true_vec = h_ntn_all_np[int(rx_i), int(ant_i), int(t_i), :]
        metrics = _vector_error_metrics(h_hat_vec, h_true_vec, eps=eps)
        pair_nrmse.append(float(metrics["nrmse"]))
        pair_cos.append(float(metrics["cos_sim"]))

    pair_nrmse_arr = np.asarray(pair_nrmse, dtype=np.float64)
    pair_cos_arr = np.asarray(pair_cos, dtype=np.float64)
    if pair_nrmse_arr.size > 0:
        detected_pairs_summary = {
            "pairs": int(pair_nrmse_arr.size),
            "nrmse_mean": float(np.mean(pair_nrmse_arr)),
            "nrmse_median": float(np.median(pair_nrmse_arr)),
            "cos_mean": float(np.mean(pair_cos_arr)),
            "cos_median": float(np.median(pair_cos_arr)),
        }
    else:
        detected_pairs_summary = {
            "pairs": 0,
            "nrmse_mean": float("nan"),
            "nrmse_median": float("nan"),
            "cos_mean": float("nan"),
            "cos_median": float("nan"),
        }

    return {
        "angle_metrics": angle_metrics,
        "detected_subset_metrics": detected_subset_metrics,
        "detected_pairs_summary": detected_pairs_summary,
        "detected_rx_indices": det_rx_idx,
        "matched_pairs": matched_pairs,
    }


def run_music_angle_pipeline(
    h_all: np.ndarray,
    *,
    tx_rows: int,
    tx_cols: int,
    nsect: int,
    true_pair_map: Optional[Dict[Tuple[int, int], Tuple[float, float, int, int]]] = None,
    pair_keys: Optional[Iterable[Tuple[int, int]]] = None,
    # detection options (used only when pair_keys is None)
    detect_num_sources: Optional[int] = None,
    detect_threshold: Optional[float] = 1.0,
    detect_user_powers: Optional[np.ndarray] = None,
    detect_noise_var: float = 0.0,
    detect_covariance_mode: Literal["analytic", "sample"] = "sample",
    detect_num_snapshots: int = 100,
    detect_rng_seed: Optional[int] = None,
    detect_source_estimation: Literal["mdl", "energy", "rank"] = "mdl",
    detect_energy_ratio: float = 0.95,
    detect_rank_relative_threshold: float = 1e-4,
    detect_rank_noise_margin: float = 1e-3,
    detect_reduce_rx_ant: Literal["max", "mean"] = "max",
    # angle/candidate options
    hat_channel_mode: Literal["raw", "conj", "auto"] = "conj",
    ref_mode_for_auto: Literal["aod", "aoa_reverse"] = "aod",
    auto_mode_min_pairs: int = 1,
    selection_strategy: Literal["truth_guided", "fit_score"] = "truth_guided",
    fit_selection_metric: Literal["mean", "median"] = "mean",
    phi_mirror_about_sector: bool = False,
    steering_horizontal_sign: Literal[-1, 1] = 1,
    use_sector_orientation: bool = True,
    sector_pitch_rad: float = 0.174533,
    sector_roll_rad: float = 0.0,
    rotation_order: Literal["zyx", "zxy", "yxz", "yzx", "xyz", "xzy"] = "zyx",
    manifold_mode: str = "auto",
    manifold_auto_list: Iterable[str] = ("yz:+1", "yz:-1", "xz:+1", "xz:-1"),
    flatten_mode: str = "auto",
    flatten_mode_list: Iterable[str] = ("C", "F"),
    scan_mode: str = "auto",
    scan_mode_list: Iterable[str] = ("complex", "phase_only"),
    phi_offset_mode: Any = "auto",
    phi_offset_list: Iterable[float] = (0.0, 90.0, 180.0, 270.0),
    sector_forward_only: bool = True,
    sector_forward_cos_min: float = 0.0,
    phi_grid_deg: Optional[Iterable[float]] = None,
    theta_grid_deg: Optional[Iterable[float]] = None,
) -> Dict[str, Any]:
    """Unified MUSIC angle pipeline for NTN/TN on the same `h_all` structure.

    Parameters
    ----------
    h_all : np.ndarray
        Shape (num_rx, num_rx_ant, num_tx, num_tx_ant).
    pair_keys : iterable[(rx,t)] | None
        If None, run detection via `detect_music_from_hi` for each TX and estimate
        angles for detected RX. If provided, skip detection and only estimate these
        explicit links (useful for TN serving pairs).
    true_pair_map : dict | None
        Optional true-angle map used for candidate auto-selection metrics.
    selection_strategy : {"truth_guided", "fit_score"}
        Candidate selection strategy.
        - "truth_guided": choose by angle error on matched `true_pair_map` pairs.
        - "fit_score"   : choose by MUSIC channel-steering fit score only.
    fit_selection_metric : {"mean", "median"}
        Aggregation metric over per-link fit scores when `selection_strategy="fit_score"`.
    phi_mirror_about_sector : bool
        If True, mirror azimuth around the serving sector yaw:
            phi <- (2*yaw_sector_deg - phi) mod 360.
        This fixes left-right azimuth handedness mismatch when present.
    steering_horizontal_sign : {-1, +1}
        Horizontal-axis sign used in steering model. Prefer fixing handedness
        here instead of using output-space mirror compensation.

    Returns
    -------
    Dict[str, Any]
        Includes pair-hat dict, flattened pair arrays, and selection diagnostics.
    """
    h = _as_complex_array(h_all)
    if h.ndim != 4:
        raise ValueError("h_all must have shape (num_rx, num_rx_ant, num_tx, num_tx_ant).")

    num_rx, _num_rx_ant, num_tx, num_tx_ant = h.shape
    nsect_eff = int(max(int(nsect), 1))
    if phi_grid_deg is None:
        phi_grid_deg = np.arange(0.0, 360.0, 2.0)
    if theta_grid_deg is None:
        theta_grid_deg = np.arange(0.0, 181.0, 2.0)

    true_map = true_pair_map if true_pair_map is not None else {}
    strategy = str(selection_strategy).lower().strip()
    if strategy not in ("truth_guided", "fit_score"):
        raise ValueError("selection_strategy must be 'truth_guided' or 'fit_score'.")
    fit_metric = str(fit_selection_metric).lower().strip()
    if fit_metric not in ("mean", "median"):
        raise ValueError("fit_selection_metric must be 'mean' or 'median'.")
    if int(steering_horizontal_sign) not in (-1, 1):
        raise ValueError("steering_horizontal_sign must be -1 or +1.")

    # Optional constrained pair mode (typically TN serving links)
    pair_keys_by_tx: Optional[Dict[int, List[int]]] = None
    if pair_keys is not None:
        pair_keys_by_tx = {}
        for rx, t in pair_keys:
            rx_i = int(rx)
            t_i = int(t)
            if rx_i < 0 or rx_i >= num_rx or t_i < 0 or t_i >= num_tx:
                continue
            pair_keys_by_tx.setdefault(t_i, []).append(rx_i)
        for t_i in list(pair_keys_by_tx.keys()):
            pair_keys_by_tx[t_i] = sorted(set(pair_keys_by_tx[t_i]))

    mode_candidates = [hat_channel_mode] if hat_channel_mode in ("raw", "conj") else ["raw", "conj"]
    manifold_candidates = _build_manifold_candidates(manifold_mode, manifold_auto_list)
    flatten_candidates = _build_flatten_candidates(flatten_mode, flatten_mode_list)
    scan_candidates = _build_scan_candidates(scan_mode, scan_mode_list)
    phi_offset_candidates = _build_phi_offset_candidates(phi_offset_mode, phi_offset_list)

    pair_hat: Dict[Tuple[int, int], Dict[str, Any]] = {}
    candidate_metric_log: Dict[str, List[Dict[str, float]]] = {}
    candidate_fit_log: Dict[str, List[Dict[str, float]]] = {}
    selected_metric_log: List[Dict[str, float]] = []
    selected_fit_record: List[float] = []
    selected_mode_record: List[str] = []
    selected_manifold_record: List[str] = []
    selected_flatten_record: List[str] = []
    selected_scan_record: List[str] = []
    selected_phi_offset_record: List[float] = []
    num_sources_record: List[int] = []
    detected_rx_indices_by_tx: Dict[int, np.ndarray] = {}

    steering_cache: Dict[Tuple[Any, ...], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for t in range(num_tx):
        if pair_keys_by_tx is not None and t not in pair_keys_by_tx:
            continue

        hi_raw = h[:, :, t, :]
        if pair_keys_by_tx is not None:
            rx_indices_fixed = np.asarray(pair_keys_by_tx[t], dtype=int)
            if rx_indices_fixed.size == 0:
                continue

        candidate_results: Dict[str, Dict[str, Any]] = {}
        for mode in mode_candidates:
            hi_mode = hi_raw if mode == "raw" else np.conj(hi_raw)

            if pair_keys_by_tx is None:
                music_out = detect_music_from_hi(
                    hi=hi_mode,
                    num_sources=detect_num_sources,
                    threshold=detect_threshold,
                    user_powers=detect_user_powers,
                    noise_var=detect_noise_var,
                    covariance_mode=detect_covariance_mode,
                    num_snapshots=detect_num_snapshots,
                    rng_seed=detect_rng_seed,
                    source_estimation=detect_source_estimation,
                    energy_ratio=detect_energy_ratio,
                    rank_relative_threshold=detect_rank_relative_threshold,
                    rank_noise_margin=detect_rank_noise_margin,
                    reduce_ntn_ant=detect_reduce_rx_ant,
                )
                detected_mask_user_mode = np.asarray(music_out["detected_mask_user"], dtype=bool)
                detected_mask_per_ant_mode = np.asarray(music_out["detected_mask_per_ant"], dtype=bool)
                score_user_mode = np.asarray(music_out["score_user"], dtype=float)
                rx_indices_mode = np.where(detected_mask_user_mode)[0]
                num_sources_est_mode = int(np.asarray(music_out["num_sources_est"]).item())
            else:
                detected_mask_user_mode = np.zeros((num_rx,), dtype=bool)
                detected_mask_user_mode[rx_indices_fixed] = True
                detected_mask_per_ant_mode = np.repeat(
                    detected_mask_user_mode[:, None], hi_mode.shape[1], axis=1
                )
                score_user_mode = np.linalg.norm(hi_mode, axis=(1, 2))
                rx_indices_mode = rx_indices_fixed
                # Pair-constrained mode does not estimate K from covariance.
                num_sources_est_mode = -1
                music_out = None

            for mani_label, panel_plane, phase_sign in manifold_candidates:
                for flat in flatten_candidates:
                    sec_idx = int(t) % nsect_eff
                    if use_sector_orientation:
                        yaw = 2.0 * np.pi * sec_idx / float(nsect_eff)
                        orientation_rad = (float(yaw), float(sector_pitch_rad), float(sector_roll_rad))
                    else:
                        orientation_rad = (0.0, 0.0, 0.0)

                    nr = int(tx_rows)
                    nc = int(tx_cols)
                    if nr * nc != num_tx_ant:
                        nr, nc = 1, int(num_tx_ant)

                    skey = (
                        sec_idx,
                        nr,
                        nc,
                        mani_label,
                        flat,
                        bool(use_sector_orientation),
                        float(sector_pitch_rad),
                        float(sector_roll_rad),
                        str(rotation_order),
                        int(steering_horizontal_sign),
                        bool(sector_forward_only),
                        float(sector_forward_cos_min),
                    )
                    if skey not in steering_cache:
                        steering_cache[skey] = build_steering_bank(
                            num_rows=nr,
                            num_cols=nc,
                            phi_grid_deg=phi_grid_deg,
                            theta_grid_deg=theta_grid_deg,
                            orientation_rad=orientation_rad,
                            panel_plane=panel_plane,
                            phase_sign=phase_sign,
                            horizontal_sign=int(steering_horizontal_sign),
                            flatten_order=flat,
                            forward_only=bool(sector_forward_only),
                            forward_cos_min=float(sector_forward_cos_min),
                            rotation_order=rotation_order,
                        )
                    a_bank, phi_bank_deg, theta_bank_deg = steering_cache[skey]

                    for sm in scan_candidates:
                        for poff in phi_offset_candidates:
                            hats_mode: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
                            fit_vals_mode: List[float] = []
                            for rx_i in rx_indices_mode:
                                ant_norms = np.linalg.norm(hi_mode[int(rx_i), :, :], axis=1)
                                ant_idx = int(np.argmax(ant_norms))
                                phi_hat, theta_hat, fit_val = estimate_angle_from_channel_scan(
                                    hi_mode[int(rx_i), ant_idx, :],
                                    a_bank,
                                    phi_bank_deg,
                                    theta_bank_deg,
                                    scan_mode=sm,
                                )
                                if np.isfinite(fit_val):
                                    fit_vals_mode.append(float(fit_val))
                                if np.isfinite(phi_hat):
                                    if bool(phi_mirror_about_sector):
                                        yaw_deg = 360.0 * float(sec_idx) / float(nsect_eff)
                                        phi_hat = float((2.0 * yaw_deg - float(phi_hat)) % 360.0)
                                    phi_hat = float((phi_hat + float(poff)) % 360.0)
                                hats_mode[(int(rx_i), int(t))] = (
                                    float(phi_hat),
                                    float(theta_hat),
                                    float(score_user_mode[int(rx_i)]),
                                )

                            if len(fit_vals_mode) > 0:
                                fit_mean_mode = float(np.mean(fit_vals_mode))
                                fit_median_mode = float(np.median(fit_vals_mode))
                            else:
                                fit_mean_mode = float("nan")
                                fit_median_mode = float("nan")

                            metric_mode: Optional[Dict[str, float]] = None
                            matched = [k for k in hats_mode.keys() if k in true_map]
                            if len(matched) >= int(auto_mode_min_pairs):
                                phi_true = np.array([true_map[k][0] for k in matched], dtype=float)
                                theta_true = np.array([true_map[k][1] for k in matched], dtype=float)
                                phi_hat = np.array([hats_mode[k][0] for k in matched], dtype=float)
                                theta_hat = np.array([hats_mode[k][1] for k in matched], dtype=float)
                                metric_mode = angle_error_metrics(
                                    phi_true,
                                    theta_true,
                                    phi_hat,
                                    theta_hat,
                                    reference_mode=ref_mode_for_auto,
                                )
                                metric_mode["mode"] = mode
                                metric_mode["manifold"] = mani_label
                                metric_mode["flatten"] = flat
                                metric_mode["scan"] = sm
                                metric_mode["phi_offset_deg"] = float(poff)
                                metric_mode["t"] = int(t)
                                ckey = f"{mode}|{mani_label}|{flat}|{sm}|{float(poff):.1f}"
                                candidate_metric_log.setdefault(ckey, []).append(metric_mode)

                            ckey = f"{mode}|{mani_label}|{flat}|{sm}|{float(poff):.1f}"
                            candidate_fit_log.setdefault(ckey, []).append(
                                {
                                    "fit_mean": float(fit_mean_mode),
                                    "fit_median": float(fit_median_mode),
                                    "fit_count": float(len(fit_vals_mode)),
                                    "t": float(int(t)),
                                }
                            )
                            candidate_results[ckey] = {
                                "mode": mode,
                                "manifold": mani_label,
                                "flatten": flat,
                                "scan": sm,
                                "phi_offset_deg": float(poff),
                                "fit_mean": float(fit_mean_mode),
                                "fit_median": float(fit_median_mode),
                                "fit_count": int(len(fit_vals_mode)),
                                "music_out": music_out,
                                "detected_mask_user": detected_mask_user_mode,
                                "detected_mask_per_ant": detected_mask_per_ant_mode,
                                "score_user": score_user_mode,
                                "hats": hats_mode,
                                "metric": metric_mode,
                                "num_sources_est": num_sources_est_mode,
                            }

        if len(candidate_results) == 0:
            continue

        if (
            hat_channel_mode in ("raw", "conj")
            and str(manifold_mode).lower() != "auto"
            and str(flatten_mode).lower() != "auto"
            and str(scan_mode).lower() != "auto"
            and str(phi_offset_mode).lower() != "auto"
        ):
            fixed_key, _, _ = _parse_manifold_label(str(manifold_mode))
            fixed_flat = str(flatten_mode).upper()
            fixed_scan = str(scan_mode).lower()
            fixed_poff = float(np.round(float(phi_offset_mode) % 360.0, 1))
            selected_key = f"{hat_channel_mode}|{fixed_key}|{fixed_flat}|{fixed_scan}|{fixed_poff:.1f}"
            if selected_key not in candidate_results:
                selected_key = list(candidate_results.keys())[0]
        else:
            selected_key: Optional[str] = None
            if strategy == "truth_guided":
                scored: List[Tuple[float, str]] = []
                for ckey, cres in candidate_results.items():
                    met = cres["metric"]
                    if met is None:
                        continue
                    val = float(met["phi_mae_deg"] + met["theta_mae_deg"])
                    if np.isfinite(val):
                        scored.append((val, ckey))
                if len(scored) > 0:
                    scored.sort(key=lambda x: x[0])
                    selected_key = scored[0][1]

            if selected_key is None:
                fit_scored: List[Tuple[float, str]] = []
                fit_key = "fit_median" if fit_metric == "median" else "fit_mean"
                for ckey, cres in candidate_results.items():
                    fit_val = float(cres.get(fit_key, np.nan))
                    if np.isfinite(fit_val):
                        # maximize fit score
                        fit_scored.append((-fit_val, ckey))
                if len(fit_scored) > 0:
                    fit_scored.sort(key=lambda x: x[0])
                    selected_key = fit_scored[0][1]

            if selected_key is None:
                pref: List[Tuple[int, int, int, int, str]] = []
                for ckey, cres in candidate_results.items():
                    rank_mode = 0 if cres["mode"] == "raw" else 1
                    rank_flat = 0 if cres.get("flatten", "C") == "C" else 1
                    rank_scan = 0 if cres.get("scan", "complex") == "complex" else 1
                    rank_poff = 0 if abs(float(cres.get("phi_offset_deg", 0.0))) < 1e-9 else 1
                    pref.append((rank_mode, rank_flat, rank_scan, rank_poff, ckey))
                pref.sort()
                selected_key = pref[0][4]

        selected = candidate_results[selected_key]
        selected_mode_record.append(str(selected["mode"]))
        selected_manifold_record.append(str(selected["manifold"]))
        selected_flatten_record.append(str(selected["flatten"]))
        selected_scan_record.append(str(selected["scan"]))
        selected_phi_offset_record.append(float(selected.get("phi_offset_deg", 0.0)))
        selected_fit_record.append(float(selected.get("fit_mean", np.nan)))
        num_sources_record.append(int(selected.get("num_sources_est", 0)))

        if selected["metric"] is not None:
            selected_metric_log.append(selected["metric"])

        detected_user = np.asarray(selected["detected_mask_user"], dtype=bool)
        rx_kept = np.where(detected_user)[0]
        detected_rx_indices_by_tx[int(t)] = rx_kept
        hats_sel = selected["hats"]
        score_user = np.asarray(selected["score_user"], dtype=float)

        for rx_i in rx_kept:
            key = (int(rx_i), int(t))
            if key in hats_sel:
                phi_hat, theta_hat, score = hats_sel[key]
            else:
                phi_hat, theta_hat, score = np.nan, np.nan, float(score_user[int(rx_i)])

            prev = pair_hat.get(key)
            if prev is None or float(score) > float(prev["score"]):
                pair_hat[key] = {
                    "score": float(score),
                    "bs": int(t) // nsect_eff,
                    "sec": int(t) % nsect_eff,
                    "phi_hat_deg": float(phi_hat),
                    "theta_hat_deg": float(theta_hat),
                    "hat_mode": selected["mode"],
                    "manifold": selected["manifold"],
                    "flatten": selected["flatten"],
                    "scan": selected["scan"],
                    "phi_offset_deg": float(selected.get("phi_offset_deg", 0.0)),
                }

    det_pairs = sorted(pair_hat.keys(), key=lambda k: (k[1], k[0]))
    if len(det_pairs) > 0:
        pair_rx_idx = np.array([k[0] for k in det_pairs], dtype=int)
        pair_t_idx = np.array([k[1] for k in det_pairs], dtype=int)
        pair_bs_idx = np.array([pair_hat[k]["bs"] for k in det_pairs], dtype=int)
        pair_sector_idx = np.array([pair_hat[k]["sec"] for k in det_pairs], dtype=int)
        pair_phi_hat_deg = np.array([pair_hat[k]["phi_hat_deg"] for k in det_pairs], dtype=float)
        pair_theta_hat_deg = np.array([pair_hat[k]["theta_hat_deg"] for k in det_pairs], dtype=float)
    else:
        pair_rx_idx = np.empty((0,), dtype=int)
        pair_t_idx = np.empty((0,), dtype=int)
        pair_bs_idx = np.empty((0,), dtype=int)
        pair_sector_idx = np.empty((0,), dtype=int)
        pair_phi_hat_deg = np.empty((0,), dtype=float)
        pair_theta_hat_deg = np.empty((0,), dtype=float)

    if len(detected_rx_indices_by_tx) > 0:
        detected_rx_indices_unique = np.unique(np.concatenate(list(detected_rx_indices_by_tx.values())))
    else:
        detected_rx_indices_unique = np.empty((0,), dtype=int)

    return {
        "pair_hat": pair_hat,
        "pair_rx_idx": pair_rx_idx,
        "pair_t_idx": pair_t_idx,
        "pair_bs_idx": pair_bs_idx,
        "pair_sector_idx": pair_sector_idx,
        "pair_phi_hat_deg": pair_phi_hat_deg,
        "pair_theta_hat_deg": pair_theta_hat_deg,
        "selected_mode_record": np.asarray(selected_mode_record, dtype=object),
        "selected_manifold_record": np.asarray(selected_manifold_record, dtype=object),
        "selected_flatten_record": np.asarray(selected_flatten_record, dtype=object),
        "selected_scan_record": np.asarray(selected_scan_record, dtype=object),
        "selected_phi_offset_record": np.asarray(selected_phi_offset_record, dtype=float),
        "selected_fit_record": np.asarray(selected_fit_record, dtype=float),
        "num_sources_record": np.asarray(num_sources_record, dtype=int),
        "candidate_metric_log": candidate_metric_log,
        "candidate_fit_log": candidate_fit_log,
        "selected_metric_log": selected_metric_log,
        "detected_rx_indices_by_tx": detected_rx_indices_by_tx,
        "detected_rx_indices_unique": detected_rx_indices_unique,
    }


def _resolve_tx_orientations(
    *,
    num_tx: int,
    nsect: int,
    use_sector_orientation: bool,
    sector_yaw_offset_rad: float,
    sector_pitch_rad: float,
    sector_roll_rad: float,
    tx_orientations_rad: Optional[np.ndarray],
) -> np.ndarray:
    """Return one Sionna-style ``[yaw, pitch, roll]`` row per TX."""
    if tx_orientations_rad is not None:
        orientations = np.asarray(tx_orientations_rad, dtype=np.float64)
        if orientations.shape != (int(num_tx), 3):
            raise ValueError(
                "tx_orientations_rad must have shape (num_tx, 3); "
                f"expected ({int(num_tx)}, 3), got {orientations.shape}."
            )
        if not np.all(np.isfinite(orientations)):
            raise ValueError("tx_orientations_rad must contain only finite values.")
        return orientations

    if not bool(use_sector_orientation):
        return np.zeros((int(num_tx), 3), dtype=np.float64)

    nsect_eff = int(max(int(nsect), 1))
    orientations = np.empty((int(num_tx), 3), dtype=np.float64)
    for t in range(int(num_tx)):
        sec_idx = int(t) % nsect_eff
        orientations[t] = (
            float(sector_yaw_offset_rad) + 2.0 * np.pi * sec_idx / float(nsect_eff),
            float(sector_pitch_rad),
            float(sector_roll_rad),
        )
    orientations[:, 0] = np.mod(orientations[:, 0], 2.0 * np.pi)
    return orientations


def _run_music_standard_blind_pipeline(
    h: np.ndarray,
    *,
    tx_rows: int,
    tx_cols: int,
    nsect: int,
    detect_num_sources: Optional[int],
    detect_threshold: Optional[float],
    detect_user_powers: Optional[np.ndarray],
    detect_noise_var: float,
    detect_covariance_mode: Literal["analytic", "sample"],
    detect_num_snapshots: int,
    detect_rng_seed: Optional[int],
    detect_source_estimation: Literal["mdl", "energy", "rank"],
    detect_energy_ratio: float,
    detect_rank_relative_threshold: float,
    detect_rank_noise_margin: float,
    detect_reduce_rx_ant: Literal["max", "mean"],
    channel_mode: Literal["raw", "conj"],
    manifold_label: str,
    flatten_order: Literal["C", "F"],
    scan_mode: Literal["complex", "phase_only"],
    phi_offset_deg: float,
    phi_mirror_about_sector: bool,
    steering_horizontal_sign: Literal[-1, 1],
    use_sector_orientation: bool,
    sector_yaw_offset_rad: float,
    sector_pitch_rad: float,
    sector_roll_rad: float,
    tx_orientations_rad: Optional[np.ndarray],
    array_positions_local: Optional[np.ndarray],
    rotation_order: Literal["zyx", "zxy", "yxz", "yzx", "xyz", "xzy"],
    sector_forward_only: bool,
    sector_forward_cos_min: float,
    peak_min_sep_phi_deg: float,
    peak_min_sep_theta_deg: float,
    peak_max_correlation: Optional[float],
    peak_refine: bool,
    peak_refine_half_width_deg: float,
    peak_refine_maxiter: int,
    peak_local_maxima: bool,
    peak_max_noise_projection: Optional[float],
    covariance_refine: bool,
    covariance_refine_maxiter: int,
    covariance_refine_max_direction_step: float,
    phi_grid_deg: Iterable[float],
    theta_grid_deg: Iterable[float],
) -> Dict[str, Any]:
    """Run the blind anonymous MUSIC detector on one full NTN tensor.

    Detection model
    ---------------
        Rxx ~= sum_k g_k u_k u_k^H + sigma^2 I

    where the peaks `(u_k, g_k)` are anonymous interference terms. When
    ``detect_source_estimation == "mdl"``, this corresponds to
    ``Blind MDL-MUSIC Detection``.

    Notes
    -----
    ``peak_*`` outputs are the primary detector outputs. ``pair_*`` outputs are
    evaluation-only mappings that assign anonymous peaks back to per-RX links so
    the existing notebook angle/channel metrics stay comparable to the earlier
    paired-channel workflow.
    """
    num_rx, num_rx_ant, num_tx, num_tx_ant = h.shape
    nsect_eff = int(max(int(nsect), 1))
    detection_name = blind_music_detection_name(detect_source_estimation)
    if covariance_refine:
        detection_name += " + covariance fit"

    manifold_key, panel_plane, phase_sign = _parse_manifold_label(manifold_label)
    phi_off = float(np.round(float(phi_offset_deg) % 360.0, 1))
    orientation_by_tx = _resolve_tx_orientations(
        num_tx=num_tx,
        nsect=nsect_eff,
        use_sector_orientation=use_sector_orientation,
        sector_yaw_offset_rad=sector_yaw_offset_rad,
        sector_pitch_rad=sector_pitch_rad,
        sector_roll_rad=sector_roll_rad,
        tx_orientations_rad=tx_orientations_rad,
    )
    exact_array_geometry = array_positions_local is not None
    array_phase_sign = -1 if channel_mode == "conj" else 1

    pair_hat: Dict[Tuple[int, int], Dict[str, Any]] = {}
    h_hat_all_mode = np.zeros_like(h, dtype=np.complex128)
    h_hat_all_raw = np.zeros_like(h, dtype=np.complex128)
    selected_mode_record: List[str] = []
    selected_manifold_record: List[str] = []
    selected_flatten_record: List[str] = []
    selected_scan_record: List[str] = []
    selected_phi_offset_record: List[float] = []
    selected_fit_record: List[float] = []
    num_sources_record: List[int] = []
    num_sources_mdl_record: List[int] = []
    num_sources_rank_record: List[int] = []
    num_sources_eigengap_record: List[int] = []
    fit_residual_before: List[float] = []
    fit_residual_after: List[float] = []
    accepted_peak_counts: List[int] = []
    candidate_metric_log: Dict[str, List[Dict[str, float]]] = {}
    candidate_fit_log: Dict[str, List[Dict[str, float]]] = {}
    selected_metric_log: List[Dict[str, float]] = []
    detected_rx_indices_by_tx: Dict[int, np.ndarray] = {}

    peak_t_idx_record: List[int] = []
    peak_bs_idx_record: List[int] = []
    peak_sector_idx_record: List[int] = []
    peak_phi_hat_record: List[float] = []
    peak_theta_hat_record: List[float] = []
    peak_score_record: List[float] = []
    peak_selection_score_record: List[float] = []
    peak_g_hat_record: List[float] = []
    peak_alpha_hat_mode_record: List[np.complex128] = []
    peak_alpha_hat_raw_record: List[np.complex128] = []
    peak_u_hat_mode_record: List[np.ndarray] = []
    peak_u_hat_raw_record: List[np.ndarray] = []
    peak_h_hat_mode_record: List[np.ndarray] = []
    peak_h_hat_raw_record: List[np.ndarray] = []

    ckey = f"{channel_mode}|{manifold_key}|{flatten_order}|{scan_mode}|{phi_off:.1f}"

    for t in range(num_tx):
        hi_raw = h[:, :, t, :]
        hi_mode = hi_raw if channel_mode == "raw" else np.conj(hi_raw)

        music_out = detect_music_from_hi(
            hi=hi_mode,
            num_sources=detect_num_sources,
            threshold=detect_threshold,
            user_powers=detect_user_powers,
            noise_var=detect_noise_var,
            covariance_mode=detect_covariance_mode,
            num_snapshots=detect_num_snapshots,
            rng_seed=detect_rng_seed,
            source_estimation=detect_source_estimation,
            energy_ratio=detect_energy_ratio,
            rank_relative_threshold=detect_rank_relative_threshold,
            rank_noise_margin=detect_rank_noise_margin,
            reduce_ntn_ant=detect_reduce_rx_ant,
            compute_user_scores=False,
        )
        num_sources_est = int(np.asarray(music_out["num_sources_est"]).item())
        num_sources_mdl_record.append(int(np.asarray(music_out["num_sources_mdl"]).item()))
        num_sources_rank_record.append(int(np.asarray(music_out["num_sources_rank"]).item()))
        num_sources_eigengap_record.append(int(np.asarray(music_out["num_sources_eigengap"]).item()))
        us, en = signal_noise_subspaces_from_music_out(music_out)

        sec_idx = int(t) % nsect_eff
        orientation_rad = tuple(float(v) for v in orientation_by_tx[int(t)])

        nr = int(tx_rows)
        nc = int(tx_cols)
        if nr * nc != num_tx_ant:
            nr, nc = 1, int(num_tx_ant)

        peaks = music_top_peaks(
            en,
            num_rows=nr,
            num_cols=nc,
            phi_grid_deg=phi_grid_deg,
            theta_grid_deg=theta_grid_deg,
            orientation_rad=orientation_rad,
            panel_plane=panel_plane,
            phase_sign=phase_sign,
            horizontal_sign=int(steering_horizontal_sign),
            flatten_order=flatten_order,
            forward_only=bool(sector_forward_only),
            forward_cos_min=float(sector_forward_cos_min),
            rotation_order=rotation_order,
            array_positions_local=array_positions_local,
            array_phase_sign=array_phase_sign,
            top_n=max(int(num_sources_est), 0),
            local_maxima_only=peak_local_maxima,
            max_noise_projection=peak_max_noise_projection,
            min_sep_phi_deg=float(peak_min_sep_phi_deg),
            min_sep_theta_deg=float(peak_min_sep_theta_deg),
            max_peak_correlation=peak_max_correlation,
            refine_peaks=bool(peak_refine),
            refine_half_width_deg=float(peak_refine_half_width_deg),
            refine_maxiter=int(peak_refine_maxiter),
        )

        noise_power_hat = (float(detect_noise_var) if detect_covariance_mode == "analytic"
                           else estimate_noise_power_from_music_out(music_out))
        fit_diag = {"before": float("nan"), "after": float("nan")}
        if covariance_refine and peaks:
            if array_positions_local is None:
                raise ValueError("covariance_refine requires exact Sionna array positions.")
            peaks, _, fit_diag = refine_music_covariance(
                music_out["covariance"], peaks, array_positions_local=array_positions_local,
                orientation_rad=orientation_rad, en=en, noise_power=noise_power_hat,
                array_phase_sign=array_phase_sign, rotation_order=rotation_order,
                forward_cos_min=float(sector_forward_cos_min) if sector_forward_only else 0.0,
                max_direction_step=covariance_refine_max_direction_step,
                maxiter=covariance_refine_maxiter,
            )
        fit_residual_before.append(fit_diag["before"])
        fit_residual_after.append(fit_diag["after"])
        if len(peaks) > 0:
            u_hat_mode_arr = np.vstack(
                [np.asarray(a, dtype=np.complex128).reshape(1, -1) for _p, _ph, _th, a in peaks]
            )
            g_hat_arr = estimate_noncoh_gains_from_covariance(
                music_out["covariance"],
                u_hat_mode_arr,
                noise_power=noise_power_hat,
            )
            active = g_hat_arr > 0.0
            peaks = [peak for peak, keep in zip(peaks, active) if keep]
            u_hat_mode_arr = u_hat_mode_arr[active]
            g_hat_arr = g_hat_arr[active]
            peak_score_arr = np.asarray([float(p) for p, _ph, _th, _a in peaks], dtype=np.float64)
            if channel_mode == "conj":
                u_hat_raw_arr = np.conjugate(u_hat_mode_arr)
            else:
                u_hat_raw_arr = np.asarray(u_hat_mode_arr, dtype=np.complex128)
            alpha_hat_mode_arr = np.sqrt(np.maximum(g_hat_arr, 0.0)).astype(np.complex128)
            alpha_hat_raw_arr = np.conjugate(alpha_hat_mode_arr) if channel_mode == "conj" else alpha_hat_mode_arr
            h_hat_mode_arr = alpha_hat_mode_arr[:, None] * u_hat_mode_arr
            h_hat_raw_arr = alpha_hat_raw_arr[:, None] * u_hat_raw_arr
            peak_selection_arr = np.maximum(peak_score_arr, 0.0) * np.maximum(g_hat_arr, 0.0)
        else:
            u_hat_mode_arr = np.empty((0, num_tx_ant), dtype=np.complex128)
            u_hat_raw_arr = np.empty((0, num_tx_ant), dtype=np.complex128)
            g_hat_arr = np.empty((0,), dtype=np.float64)
            peak_score_arr = np.empty((0,), dtype=np.float64)
            peak_selection_arr = np.empty((0,), dtype=np.float64)
            alpha_hat_mode_arr = np.empty((0,), dtype=np.complex128)
            alpha_hat_raw_arr = np.empty((0,), dtype=np.complex128)
            h_hat_mode_arr = np.empty((0, num_tx_ant), dtype=np.complex128)
            h_hat_raw_arr = np.empty((0, num_tx_ant), dtype=np.complex128)

        accepted_peak_counts.append(len(peaks))
        peak_phi_store_t: List[float] = []
        peak_theta_store_t: List[float] = []

        eval_fit_vals: List[float] = []
        score_per_ant_eval, score_user_eval = _compute_user_scores(
            hi=hi_mode,
            us=us,
            en=en,
            reduce_ntn_ant=detect_reduce_rx_ant,
        )
        if detect_threshold is None:
            k_users = int(np.clip(num_sources_est, 0, num_rx))
            detected_mask_user = np.zeros((num_rx,), dtype=bool)
            if k_users > 0 and score_user_eval.size > 0:
                top_idx = np.argsort(score_user_eval)[::-1][:k_users]
                detected_mask_user[top_idx] = True
        else:
            detected_mask_user = score_user_eval >= float(detect_threshold)
        detected_rx_indices_by_tx[int(t)] = np.where(detected_mask_user)[0]

        for peak_idx, (p_peak, phi_hat_raw, theta_hat, _a) in enumerate(peaks):
            phi_store = float(phi_hat_raw)
            if np.isfinite(phi_store):
                if bool(phi_mirror_about_sector):
                    yaw_deg = np.rad2deg(float(orientation_rad[0]))
                    phi_store = float((2.0 * yaw_deg - float(phi_store)) % 360.0)
                phi_store = float((phi_store + phi_off) % 360.0)

            peak_t_idx_record.append(int(t))
            peak_bs_idx_record.append(int(t) // nsect_eff)
            peak_sector_idx_record.append(int(t) % nsect_eff)
            peak_phi_hat_record.append(float(phi_store))
            peak_theta_hat_record.append(float(theta_hat))
            peak_phi_store_t.append(float(phi_store))
            peak_theta_store_t.append(float(theta_hat))
            peak_score_record.append(float(p_peak))
            peak_selection_score_record.append(float(peak_selection_arr[peak_idx]))
            peak_g_hat_record.append(float(g_hat_arr[peak_idx]))
            peak_alpha_hat_mode_record.append(np.complex128(alpha_hat_mode_arr[peak_idx]))
            peak_alpha_hat_raw_record.append(np.complex128(alpha_hat_raw_arr[peak_idx]))
            peak_u_hat_mode_record.append(np.asarray(u_hat_mode_arr[peak_idx], dtype=np.complex128))
            peak_u_hat_raw_record.append(np.asarray(u_hat_raw_arr[peak_idx], dtype=np.complex128))
            peak_h_hat_mode_record.append(np.asarray(h_hat_mode_arr[peak_idx], dtype=np.complex128))
            peak_h_hat_raw_record.append(np.asarray(h_hat_raw_arr[peak_idx], dtype=np.complex128))

        for rx_i in detected_rx_indices_by_tx[int(t)]:
            if u_hat_mode_arr.shape[0] == 0:
                continue
            ant_norms = np.linalg.norm(hi_mode[int(rx_i), :, :], axis=1)
            ant_idx = int(np.argmax(ant_norms))
            h_obs_mode = np.asarray(hi_mode[int(rx_i), int(ant_idx), :], dtype=np.complex128).reshape(-1)
            h_norm = float(np.linalg.norm(h_obs_mode))
            if not np.isfinite(h_norm) or h_norm <= 1e-12:
                continue
            h_obs_unit = h_obs_mode / h_norm
            fit_vec = np.abs(np.einsum("ka,a->k", np.conjugate(u_hat_mode_arr), h_obs_unit, optimize=True))
            best_idx = int(np.argmax(fit_vec))
            fit_score = float(fit_vec[best_idx])
            eval_fit_vals.append(fit_score)

            phi_hat = float(peak_phi_store_t[best_idx])
            theta_hat = float(peak_theta_store_t[best_idx])
            score = float(score_user_eval[int(rx_i)])

            rec: Dict[str, Any] = {
                "score": score,
                "score_user": score,
                "bs": int(t) // nsect_eff,
                "sec": int(t) % nsect_eff,
                "phi_hat_deg": phi_hat,
                "theta_hat_deg": theta_hat,
                "hat_mode": str(channel_mode),
                "manifold": str(manifold_key),
                "flatten": str(flatten_order),
                "scan": str(scan_mode),
                "phi_offset_deg": float(phi_off),
                "rx_ant_idx": int(ant_idx),
                "steering_idx": int(best_idx),
                "fit_score": fit_score,
                "u_hat": np.asarray(u_hat_mode_arr[best_idx], dtype=np.complex128),
                "u_hat_raw": np.asarray(u_hat_raw_arr[best_idx], dtype=np.complex128),
                "alpha_hat_mode": np.complex128(alpha_hat_mode_arr[best_idx]),
                "alpha_hat_raw": np.complex128(alpha_hat_raw_arr[best_idx]),
                "h_hat_vec_mode": np.asarray(h_hat_mode_arr[best_idx], dtype=np.complex128),
                "h_hat_vec_raw": np.asarray(h_hat_raw_arr[best_idx], dtype=np.complex128),
                "selection_score": float(max(score, 0.0) * max(fit_score, 0.0)),
            }
            pair_hat[(int(rx_i), int(t))] = rec
            h_hat_all_mode[int(rx_i), int(ant_idx), int(t), :] = rec["h_hat_vec_mode"]
            h_hat_all_raw[int(rx_i), int(ant_idx), int(t), :] = rec["h_hat_vec_raw"]

        if len(eval_fit_vals) > 0:
            fit_mean_mode = float(np.mean(eval_fit_vals))
            fit_median_mode = float(np.median(eval_fit_vals))
        else:
            fit_mean_mode = float("nan")
            fit_median_mode = float("nan")

        candidate_fit_log.setdefault(ckey, []).append(
            {
                "fit_mean": fit_mean_mode,
                "fit_median": fit_median_mode,
                "fit_count": float(len(eval_fit_vals)),
                "t": float(int(t)),
            }
        )

        selected_mode_record.append(str(channel_mode))
        selected_manifold_record.append(str(manifold_key))
        selected_flatten_record.append(str(flatten_order))
        selected_scan_record.append(str(scan_mode))
        selected_phi_offset_record.append(float(phi_off))
        selected_fit_record.append(float(fit_mean_mode))
        num_sources_record.append(int(num_sources_est))

    det_pairs = sorted(pair_hat.keys(), key=lambda k: (k[1], k[0]))
    if len(det_pairs) > 0:
        pair_rx_idx = np.array([k[0] for k in det_pairs], dtype=int)
        pair_t_idx = np.array([k[1] for k in det_pairs], dtype=int)
        pair_bs_idx = np.array([pair_hat[k]["bs"] for k in det_pairs], dtype=int)
        pair_sector_idx = np.array([pair_hat[k]["sec"] for k in det_pairs], dtype=int)
        pair_phi_hat_deg = np.array([pair_hat[k]["phi_hat_deg"] for k in det_pairs], dtype=float)
        pair_theta_hat_deg = np.array([pair_hat[k]["theta_hat_deg"] for k in det_pairs], dtype=float)
        pair_rx_ant_idx = np.array([int(pair_hat[k].get("rx_ant_idx", -1)) for k in det_pairs], dtype=int)
        pair_score_user = np.array(
            [float(pair_hat[k].get("score_user", pair_hat[k].get("score", np.nan))) for k in det_pairs],
            dtype=float,
        )
        pair_fit_score = np.array(
            [float(pair_hat[k].get("fit_score", np.nan)) for k in det_pairs],
            dtype=float,
        )
        pair_selection_score = np.array(
            [float(pair_hat[k].get("selection_score", np.nan)) for k in det_pairs],
            dtype=float,
        )
        pair_alpha_hat_mode = np.array(
            [np.complex128(pair_hat[k].get("alpha_hat_mode", np.nan + 1j * np.nan)) for k in det_pairs],
            dtype=np.complex128,
        )
        pair_alpha_hat_raw = np.array(
            [np.complex128(pair_hat[k].get("alpha_hat_raw", np.nan + 1j * np.nan)) for k in det_pairs],
            dtype=np.complex128,
        )
        nan_vec = np.full((num_tx_ant,), np.nan + 1j * np.nan, dtype=np.complex128)
        pair_u_hat = np.vstack(
            [
                np.asarray(pair_hat[k].get("u_hat", nan_vec), dtype=np.complex128).reshape(1, -1)
                for k in det_pairs
            ]
        )
        pair_u_hat_raw = np.vstack(
            [
                np.asarray(pair_hat[k].get("u_hat_raw", nan_vec), dtype=np.complex128).reshape(1, -1)
                for k in det_pairs
            ]
        )
        pair_h_hat_vec_mode = np.vstack(
            [
                np.asarray(pair_hat[k].get("h_hat_vec_mode", nan_vec), dtype=np.complex128).reshape(1, -1)
                for k in det_pairs
            ]
        )
        pair_h_hat_vec_raw = np.vstack(
            [
                np.asarray(pair_hat[k].get("h_hat_vec_raw", nan_vec), dtype=np.complex128).reshape(1, -1)
                for k in det_pairs
            ]
        )
    else:
        pair_rx_idx = np.empty((0,), dtype=int)
        pair_t_idx = np.empty((0,), dtype=int)
        pair_bs_idx = np.empty((0,), dtype=int)
        pair_sector_idx = np.empty((0,), dtype=int)
        pair_phi_hat_deg = np.empty((0,), dtype=float)
        pair_theta_hat_deg = np.empty((0,), dtype=float)
        pair_rx_ant_idx = np.empty((0,), dtype=int)
        pair_score_user = np.empty((0,), dtype=float)
        pair_fit_score = np.empty((0,), dtype=float)
        pair_selection_score = np.empty((0,), dtype=float)
        pair_alpha_hat_mode = np.empty((0,), dtype=np.complex128)
        pair_alpha_hat_raw = np.empty((0,), dtype=np.complex128)
        pair_u_hat = np.empty((0, num_tx_ant), dtype=np.complex128)
        pair_u_hat_raw = np.empty((0, num_tx_ant), dtype=np.complex128)
        pair_h_hat_vec_mode = np.empty((0, num_tx_ant), dtype=np.complex128)
        pair_h_hat_vec_raw = np.empty((0, num_tx_ant), dtype=np.complex128)

    if len(detected_rx_indices_by_tx) > 0:
        detected_rx_indices_unique = np.unique(np.concatenate(list(detected_rx_indices_by_tx.values())))
    else:
        detected_rx_indices_unique = np.empty((0,), dtype=int)

    peak_t_idx = np.asarray(peak_t_idx_record, dtype=int)
    peak_bs_idx = np.asarray(peak_bs_idx_record, dtype=int)
    peak_sector_idx = np.asarray(peak_sector_idx_record, dtype=int)
    peak_phi_hat_deg = np.asarray(peak_phi_hat_record, dtype=float)
    peak_theta_hat_deg = np.asarray(peak_theta_hat_record, dtype=float)
    peak_score = np.asarray(peak_score_record, dtype=float)
    peak_selection_score = np.asarray(peak_selection_score_record, dtype=float)
    peak_g_hat = np.asarray(peak_g_hat_record, dtype=np.float64)
    if len(peak_u_hat_mode_record) > 0:
        peak_u_hat = np.vstack([u.reshape(1, -1) for u in peak_u_hat_mode_record])
        peak_u_hat_raw = np.vstack([u.reshape(1, -1) for u in peak_u_hat_raw_record])
        peak_h_hat_vec_mode = np.vstack([u.reshape(1, -1) for u in peak_h_hat_mode_record])
        peak_h_hat_vec_raw = np.vstack([u.reshape(1, -1) for u in peak_h_hat_raw_record])
    else:
        peak_u_hat = np.empty((0, num_tx_ant), dtype=np.complex128)
        peak_u_hat_raw = np.empty((0, num_tx_ant), dtype=np.complex128)
        peak_h_hat_vec_mode = np.empty((0, num_tx_ant), dtype=np.complex128)
        peak_h_hat_vec_raw = np.empty((0, num_tx_ant), dtype=np.complex128)

    return {
        "pair_hat": pair_hat,
        "pair_rx_idx": pair_rx_idx,
        "pair_t_idx": pair_t_idx,
        "pair_bs_idx": pair_bs_idx,
        "pair_sector_idx": pair_sector_idx,
        "pair_phi_hat_deg": pair_phi_hat_deg,
        "pair_theta_hat_deg": pair_theta_hat_deg,
        "pair_rx_ant_idx": pair_rx_ant_idx,
        "pair_score_user": pair_score_user,
        "pair_fit_score": pair_fit_score,
        "pair_selection_score": pair_selection_score,
        "pair_u_hat": pair_u_hat,
        "pair_u_hat_raw": pair_u_hat_raw,
        "pair_alpha_hat_mode": pair_alpha_hat_mode,
        "pair_alpha_hat_raw": pair_alpha_hat_raw,
        "pair_h_hat_vec_mode": pair_h_hat_vec_mode,
        "pair_h_hat_vec_raw": pair_h_hat_vec_raw,
        "peak_t_idx": peak_t_idx,
        "peak_bs_idx": peak_bs_idx,
        "peak_sector_idx": peak_sector_idx,
        "peak_phi_hat_deg": peak_phi_hat_deg,
        "peak_theta_hat_deg": peak_theta_hat_deg,
        "peak_score": peak_score,
        "peak_selection_score": peak_selection_score,
        "peak_g_hat": peak_g_hat,
        "peak_u_hat": peak_u_hat,
        "peak_u_hat_raw": peak_u_hat_raw,
        "peak_alpha_hat_mode": np.asarray(peak_alpha_hat_mode_record, dtype=np.complex128),
        "peak_alpha_hat_raw": np.asarray(peak_alpha_hat_raw_record, dtype=np.complex128),
        "peak_h_hat_vec_mode": peak_h_hat_vec_mode,
        "peak_h_hat_vec_raw": peak_h_hat_vec_raw,
        "h_hat_all_mode": h_hat_all_mode,
        "h_hat_all_raw": h_hat_all_raw,
        "h_hat_all": h_hat_all_raw,
        "selected_mode_record": np.asarray(selected_mode_record, dtype=object),
        "selected_manifold_record": np.asarray(selected_manifold_record, dtype=object),
        "selected_flatten_record": np.asarray(selected_flatten_record, dtype=object),
        "selected_scan_record": np.asarray(selected_scan_record, dtype=object),
        "selected_phi_offset_record": np.asarray(selected_phi_offset_record, dtype=float),
        "selected_fit_record": np.asarray(selected_fit_record, dtype=float),
        "num_sources_record": np.asarray(num_sources_record, dtype=int),
        "num_sources_mdl_record": np.asarray(num_sources_mdl_record, dtype=int),
        "num_sources_rank_record": np.asarray(num_sources_rank_record, dtype=int),
        "num_sources_eigengap_record": np.asarray(num_sources_eigengap_record, dtype=int),
        "accepted_peak_counts": np.asarray(accepted_peak_counts, dtype=int),
        "covariance_fit_before": np.asarray(fit_residual_before, dtype=float),
        "covariance_fit_after": np.asarray(fit_residual_after, dtype=float),
        "covariance_refine": bool(covariance_refine),
        "candidate_metric_log": candidate_metric_log,
        "candidate_fit_log": candidate_fit_log,
        "selected_metric_log": selected_metric_log,
        "detected_rx_indices_by_tx": detected_rx_indices_by_tx,
        "detected_rx_indices_unique": detected_rx_indices_unique,
        "detection_name": detection_name,
        "blind_detection_model": BLIND_MDL_MUSIC_MODEL_SUMMARY,
        "blind_pair_mapping_eval_only": np.array(True, dtype=bool),
        "blind_mdl_music_mode": np.array(
            str(detect_source_estimation).strip().lower() == "mdl",
            dtype=bool,
        ),
        "blind_music_mode": np.array(True, dtype=bool),
        "steering_geometry_source": np.array(
            "sionna_array_positions" if exact_array_geometry else "ideal_upa",
            dtype=object,
        ),
        "tx_orientations_rad": np.asarray(orientation_by_tx, dtype=np.float64),
        "peak_min_sep_phi_deg": np.array(float(peak_min_sep_phi_deg), dtype=np.float64),
        "peak_min_sep_theta_deg": np.array(float(peak_min_sep_theta_deg), dtype=np.float64),
        "peak_max_correlation": np.array(
            np.nan if peak_max_correlation is None else float(peak_max_correlation),
            dtype=np.float64,
        ),
        "peak_refine": np.array(bool(peak_refine), dtype=bool),
        "peak_refine_half_width_deg": np.array(float(peak_refine_half_width_deg), dtype=np.float64),
        "source_count_method": np.array(str(detect_source_estimation), dtype=object),
        "rank_relative_threshold": np.array(float(detect_rank_relative_threshold), dtype=np.float64),
        "rank_noise_margin": np.array(float(detect_rank_noise_margin), dtype=np.float64),
    }


def run_music_standard_pipeline(
    h_all: np.ndarray,
    *,
    tx_rows: int,
    tx_cols: int,
    nsect: int,
    pair_keys: Optional[Iterable[Tuple[int, int]]] = None,
    detect_num_sources: Optional[int] = None,
    detect_threshold: Optional[float] = 1.0,
    detect_user_powers: Optional[np.ndarray] = None,
    detect_noise_var: float = 0.0,
    detect_covariance_mode: Literal["analytic", "sample"] = "sample",
    detect_num_snapshots: int = 100,
    detect_rng_seed: Optional[int] = None,
    detect_source_estimation: Literal["mdl", "energy", "rank"] = "mdl",
    detect_energy_ratio: float = 0.95,
    detect_rank_relative_threshold: float = 1e-4,
    detect_rank_noise_margin: float = 1e-3,
    detect_reduce_rx_ant: Literal["max", "mean"] = "max",
    channel_mode: Literal["raw", "conj"] = "conj",
    manifold_label: str = "yz:+1",
    flatten_order: Literal["C", "F"] = "F",
    scan_mode: Literal["complex", "phase_only"] = "complex",
    phi_offset_deg: float = 0.0,
    phi_mirror_about_sector: bool = False,
    steering_horizontal_sign: Literal[-1, 1] = 1,
    use_sector_orientation: bool = True,
    sector_yaw_offset_rad: float = 0.0,
    sector_pitch_rad: float = 0.174533,
    sector_roll_rad: float = 0.0,
    tx_orientations_rad: Optional[np.ndarray] = None,
    array_positions_local: Optional[np.ndarray] = None,
    rotation_order: Literal["zyx", "zxy", "yxz", "yzx", "xyz", "xzy"] = "zyx",
    sector_forward_only: bool = True,
    sector_forward_cos_min: float = 0.0,
    peak_min_sep_phi_deg: float = 8.0,
    peak_min_sep_theta_deg: float = 8.0,
    peak_max_correlation: Optional[float] = None,
    peak_refine: bool = False,
    peak_refine_half_width_deg: float = 1.0,
    peak_refine_maxiter: int = 40,
    peak_local_maxima: bool = True,
    peak_max_noise_projection: Optional[float] = 0.2,
    covariance_refine: bool = False,
    covariance_refine_maxiter: int = 60,
    covariance_refine_max_direction_step: float = 0.1,
    phi_grid_deg: Optional[Iterable[float]] = None,
    theta_grid_deg: Optional[Iterable[float]] = None,
) -> Dict[str, Any]:
    """Run a fixed-convention MUSIC pipeline without truth-guided mode search.

    This is the "standard" path when you want one deterministic MUSIC setup
    (e.g., conj + yz:+1 + F + complex + 0-deg offset), and only compare to
    Sionna truth at the end for evaluation.

    When ``pair_keys is None``, this path executes the blind anonymous detector
    above and returns ``peak_*`` as the primary outputs plus evaluation-only
    ``pair_*`` bookkeeping.
    """
    h = _as_complex_array(h_all)
    if h.ndim != 4:
        raise ValueError("h_all must have shape (num_rx, num_rx_ant, num_tx, num_tx_ant).")

    num_rx, _num_rx_ant, num_tx, num_tx_ant = h.shape
    nsect_eff = int(max(int(nsect), 1))

    if phi_grid_deg is None:
        phi_grid_deg = np.arange(0.0, 360.0, 2.0)
    if theta_grid_deg is None:
        theta_grid_deg = np.arange(0.0, 181.0, 2.0)

    if channel_mode not in ("raw", "conj"):
        raise ValueError("channel_mode must be 'raw' or 'conj'.")
    if flatten_order not in ("C", "F"):
        raise ValueError("flatten_order must be 'C' or 'F'.")
    if scan_mode not in ("complex", "phase_only"):
        raise ValueError("scan_mode must be 'complex' or 'phase_only'.")
    if int(steering_horizontal_sign) not in (-1, 1):
        raise ValueError("steering_horizontal_sign must be -1 or +1.")
    if float(peak_min_sep_phi_deg) < 0.0 or float(peak_min_sep_theta_deg) < 0.0:
        raise ValueError("MUSIC peak separation values must be nonnegative.")
    if peak_max_correlation is not None and not 0.0 < float(peak_max_correlation) <= 1.0:
        raise ValueError("peak_max_correlation must be in (0, 1] or None.")
    if float(peak_refine_half_width_deg) < 0.0:
        raise ValueError("peak_refine_half_width_deg must be nonnegative.")
    if int(peak_refine_maxiter) < 1:
        raise ValueError("peak_refine_maxiter must be at least 1.")
    if array_positions_local is not None:
        array_positions_local = np.asarray(array_positions_local, dtype=np.float64)
        if array_positions_local.shape != (num_tx_ant, 3):
            raise ValueError(
                "array_positions_local must match h_all's TX antenna dimension; "
                f"expected ({num_tx_ant}, 3), got {array_positions_local.shape}."
            )
    manifold_key, panel_plane, phase_sign = _parse_manifold_label(manifold_label)
    phi_off = float(np.round(float(phi_offset_deg) % 360.0, 1))
    orientation_by_tx = _resolve_tx_orientations(
        num_tx=num_tx,
        nsect=nsect_eff,
        use_sector_orientation=use_sector_orientation,
        sector_yaw_offset_rad=sector_yaw_offset_rad,
        sector_pitch_rad=sector_pitch_rad,
        sector_roll_rad=sector_roll_rad,
        tx_orientations_rad=tx_orientations_rad,
    )
    exact_array_geometry = array_positions_local is not None
    array_phase_sign = -1 if channel_mode == "conj" else 1

    pair_keys_by_tx: Optional[Dict[int, List[int]]] = None
    if pair_keys is not None:
        pair_keys_by_tx = {}
        for rx, t in pair_keys:
            rx_i = int(rx)
            t_i = int(t)
            if rx_i < 0 or rx_i >= num_rx or t_i < 0 or t_i >= num_tx:
                continue
            pair_keys_by_tx.setdefault(t_i, []).append(rx_i)
        for t_i in list(pair_keys_by_tx.keys()):
            pair_keys_by_tx[t_i] = sorted(set(pair_keys_by_tx[t_i]))

    if covariance_refine:
        if pair_keys_by_tx is not None or array_positions_local is None:
            raise ValueError("covariance_refine requires blind mode and exact array positions.")
        if not sector_forward_only or not 0.0 <= float(sector_forward_cos_min) < 1.0:
            raise ValueError("covariance_refine requires a front-side search with 0 <= cos_min < 1.")
        if (not np.isfinite(covariance_refine_max_direction_step)
                or covariance_refine_max_direction_step <= 0 or covariance_refine_maxiter < 1):
            raise ValueError("Covariance refinement step and iteration limit must be positive.")

    if pair_keys_by_tx is None:
        return _run_music_standard_blind_pipeline(
            h,
            tx_rows=tx_rows,
            tx_cols=tx_cols,
            nsect=nsect,
            detect_num_sources=detect_num_sources,
            detect_threshold=detect_threshold,
            detect_user_powers=detect_user_powers,
            detect_noise_var=detect_noise_var,
            detect_covariance_mode=detect_covariance_mode,
            detect_num_snapshots=detect_num_snapshots,
            detect_rng_seed=detect_rng_seed,
            detect_source_estimation=detect_source_estimation,
            detect_energy_ratio=detect_energy_ratio,
            detect_rank_relative_threshold=detect_rank_relative_threshold,
            detect_rank_noise_margin=detect_rank_noise_margin,
            detect_reduce_rx_ant=detect_reduce_rx_ant,
            channel_mode=channel_mode,
            manifold_label=manifold_label,
            flatten_order=flatten_order,
            scan_mode=scan_mode,
            phi_offset_deg=phi_offset_deg,
            phi_mirror_about_sector=phi_mirror_about_sector,
            steering_horizontal_sign=steering_horizontal_sign,
            use_sector_orientation=use_sector_orientation,
            sector_yaw_offset_rad=sector_yaw_offset_rad,
            sector_pitch_rad=sector_pitch_rad,
            sector_roll_rad=sector_roll_rad,
            tx_orientations_rad=orientation_by_tx,
            array_positions_local=array_positions_local,
            rotation_order=rotation_order,
            sector_forward_only=sector_forward_only,
            sector_forward_cos_min=sector_forward_cos_min,
            peak_min_sep_phi_deg=peak_min_sep_phi_deg,
            peak_min_sep_theta_deg=peak_min_sep_theta_deg,
            peak_max_correlation=peak_max_correlation,
            peak_refine=peak_refine,
            peak_refine_half_width_deg=peak_refine_half_width_deg,
            peak_refine_maxiter=peak_refine_maxiter,
            peak_local_maxima=peak_local_maxima,
            peak_max_noise_projection=peak_max_noise_projection,
            covariance_refine=covariance_refine,
            covariance_refine_maxiter=covariance_refine_maxiter,
            covariance_refine_max_direction_step=covariance_refine_max_direction_step,
            phi_grid_deg=phi_grid_deg,
            theta_grid_deg=theta_grid_deg,
        )

    pair_hat: Dict[Tuple[int, int], Dict[str, Any]] = {}
    h_hat_all_mode = np.zeros_like(h, dtype=np.complex128)
    h_hat_all_raw = np.zeros_like(h, dtype=np.complex128)
    selected_mode_record: List[str] = []
    selected_manifold_record: List[str] = []
    selected_flatten_record: List[str] = []
    selected_scan_record: List[str] = []
    selected_phi_offset_record: List[float] = []
    selected_fit_record: List[float] = []
    num_sources_record: List[int] = []
    candidate_metric_log: Dict[str, List[Dict[str, float]]] = {}
    candidate_fit_log: Dict[str, List[Dict[str, float]]] = {}
    selected_metric_log: List[Dict[str, float]] = []
    detected_rx_indices_by_tx: Dict[int, np.ndarray] = {}

    steering_cache: Dict[Tuple[Any, ...], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    ckey = f"{channel_mode}|{manifold_key}|{flatten_order}|{scan_mode}|{phi_off:.1f}"

    for t in range(num_tx):
        if pair_keys_by_tx is not None and t not in pair_keys_by_tx:
            continue

        hi_raw = h[:, :, t, :]
        hi_mode = hi_raw if channel_mode == "raw" else np.conj(hi_raw)

        if pair_keys_by_tx is None:
            music_out = detect_music_from_hi(
                hi=hi_mode,
                num_sources=detect_num_sources,
                threshold=detect_threshold,
                user_powers=detect_user_powers,
                noise_var=detect_noise_var,
                covariance_mode=detect_covariance_mode,
                num_snapshots=detect_num_snapshots,
                rng_seed=detect_rng_seed,
                source_estimation=detect_source_estimation,
                energy_ratio=detect_energy_ratio,
                reduce_ntn_ant=detect_reduce_rx_ant,
            )
            detected_mask_user = np.asarray(music_out["detected_mask_user"], dtype=bool)
            detected_mask_per_ant = np.asarray(music_out["detected_mask_per_ant"], dtype=bool)
            score_user = np.asarray(music_out["score_user"], dtype=float)
            rx_indices = np.where(detected_mask_user)[0]
            num_sources_est = int(np.asarray(music_out["num_sources_est"]).item())
        else:
            rx_indices_fixed = np.asarray(pair_keys_by_tx[t], dtype=int)
            if rx_indices_fixed.size == 0:
                continue
            detected_mask_user = np.zeros((num_rx,), dtype=bool)
            detected_mask_user[rx_indices_fixed] = True
            detected_mask_per_ant = np.repeat(detected_mask_user[:, None], hi_mode.shape[1], axis=1)
            score_user = np.linalg.norm(hi_mode, axis=(1, 2))
            rx_indices = rx_indices_fixed
            num_sources_est = -1
            music_out = None

        sec_idx = int(t) % nsect_eff
        orientation_rad = tuple(float(v) for v in orientation_by_tx[int(t)])

        nr = int(tx_rows)
        nc = int(tx_cols)
        if nr * nc != num_tx_ant:
            nr, nc = 1, int(num_tx_ant)

        skey = (
            tuple(float(v) for v in orientation_rad),
            nr,
            nc,
            manifold_key,
            flatten_order,
            bool(use_sector_orientation),
            str(rotation_order),
            int(steering_horizontal_sign),
            bool(exact_array_geometry),
            bool(sector_forward_only),
            float(sector_forward_cos_min),
        )
        if skey not in steering_cache:
            steering_cache[skey] = build_steering_bank(
                num_rows=nr,
                num_cols=nc,
                phi_grid_deg=phi_grid_deg,
                theta_grid_deg=theta_grid_deg,
                orientation_rad=orientation_rad,
                panel_plane=panel_plane,
                phase_sign=phase_sign,
                horizontal_sign=int(steering_horizontal_sign),
                flatten_order=flatten_order,
                forward_only=bool(sector_forward_only),
                forward_cos_min=float(sector_forward_cos_min),
                rotation_order=rotation_order,
                array_positions_local=array_positions_local,
                array_phase_sign=array_phase_sign,
            )
        a_bank, phi_bank_deg, theta_bank_deg = steering_cache[skey]

        fit_vals_mode: List[float] = []
        hats_mode: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
        extras_mode: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for rx_i in rx_indices:
            ant_norms = np.linalg.norm(hi_mode[int(rx_i), :, :], axis=1)
            ant_idx = int(np.argmax(ant_norms))
            phi_hat, theta_hat, fit_val, steer_idx = estimate_angle_from_channel_scan(
                hi_mode[int(rx_i), ant_idx, :],
                a_bank,
                phi_bank_deg,
                theta_bank_deg,
                scan_mode=scan_mode,
                return_index=True,
            )
            if np.isfinite(fit_val):
                fit_vals_mode.append(float(fit_val))

            h_obs_mode = np.asarray(hi_mode[int(rx_i), int(ant_idx), :], dtype=np.complex128).reshape(-1)
            if int(steer_idx) >= 0:
                u_hat = np.asarray(a_bank[:, int(steer_idx)], dtype=np.complex128).reshape(-1)
                den = np.vdot(u_hat, u_hat)
                alpha_hat_mode = np.vdot(u_hat, h_obs_mode) / (den + 1e-12)
                h_hat_vec_mode = alpha_hat_mode * u_hat
            else:
                u_hat = np.full((num_tx_ant,), np.nan + 1j * np.nan, dtype=np.complex128)
                alpha_hat_mode = np.complex128(np.nan + 1j * np.nan)
                h_hat_vec_mode = np.full((num_tx_ant,), np.nan + 1j * np.nan, dtype=np.complex128)

            if channel_mode == "conj":
                u_hat_raw = np.conj(u_hat)
                alpha_hat_raw = np.conj(alpha_hat_mode)
                h_hat_vec_raw = np.conj(h_hat_vec_mode)
            else:
                u_hat_raw = u_hat
                alpha_hat_raw = alpha_hat_mode
                h_hat_vec_raw = h_hat_vec_mode

            if np.isfinite(phi_hat):
                if bool(phi_mirror_about_sector):
                    yaw_deg = np.rad2deg(float(orientation_rad[0]))
                    phi_hat = float((2.0 * yaw_deg - float(phi_hat)) % 360.0)
                phi_hat = float((phi_hat + phi_off) % 360.0)
            hats_mode[(int(rx_i), int(t))] = (
                float(phi_hat),
                float(theta_hat),
                float(score_user[int(rx_i)]),
            )
            extras_mode[(int(rx_i), int(t))] = {
                "rx_ant_idx": int(ant_idx),
                "steering_idx": int(steer_idx),
                "fit_score": float(fit_val),
                "u_hat": np.asarray(u_hat, dtype=np.complex128),
                "u_hat_raw": np.asarray(u_hat_raw, dtype=np.complex128),
                "alpha_hat_mode": np.complex128(alpha_hat_mode),
                "alpha_hat_raw": np.complex128(alpha_hat_raw),
                "h_hat_vec_mode": np.asarray(h_hat_vec_mode, dtype=np.complex128),
                "h_hat_vec_raw": np.asarray(h_hat_vec_raw, dtype=np.complex128),
            }

        if len(fit_vals_mode) > 0:
            fit_mean_mode = float(np.mean(fit_vals_mode))
            fit_median_mode = float(np.median(fit_vals_mode))
        else:
            fit_mean_mode = float("nan")
            fit_median_mode = float("nan")

        candidate_fit_log.setdefault(ckey, []).append(
            {
                "fit_mean": float(fit_mean_mode),
                "fit_median": float(fit_median_mode),
                "fit_count": float(len(fit_vals_mode)),
                "t": float(int(t)),
            }
        )

        selected_mode_record.append(str(channel_mode))
        selected_manifold_record.append(str(manifold_key))
        selected_flatten_record.append(str(flatten_order))
        selected_scan_record.append(str(scan_mode))
        selected_phi_offset_record.append(float(phi_off))
        selected_fit_record.append(float(fit_mean_mode))
        num_sources_record.append(int(num_sources_est))

        detected_rx_indices_by_tx[int(t)] = np.where(detected_mask_user)[0]

        for rx_i in detected_rx_indices_by_tx[int(t)]:
            key = (int(rx_i), int(t))
            if key in hats_mode:
                phi_hat, theta_hat, score = hats_mode[key]
            else:
                phi_hat, theta_hat, score = np.nan, np.nan, float(score_user[int(rx_i)])

            prev = pair_hat.get(key)
            if prev is None or float(score) > float(prev["score"]):
                if prev is not None and "rx_ant_idx" in prev:
                    old_ant = int(prev["rx_ant_idx"])
                    if old_ant >= 0:
                        h_hat_all_mode[int(rx_i), old_ant, int(t), :] = 0.0 + 0.0j
                        h_hat_all_raw[int(rx_i), old_ant, int(t), :] = 0.0 + 0.0j

                rec: Dict[str, Any] = {
                    "score": float(score),
                    "score_user": float(score),
                    "bs": int(t) // nsect_eff,
                    "sec": int(t) % nsect_eff,
                    "phi_hat_deg": float(phi_hat),
                    "theta_hat_deg": float(theta_hat),
                    "hat_mode": str(channel_mode),
                    "manifold": str(manifold_key),
                    "flatten": str(flatten_order),
                    "scan": str(scan_mode),
                    "phi_offset_deg": float(phi_off),
                }
                extra = extras_mode.get(key)
                if extra is not None:
                    rec["rx_ant_idx"] = int(extra["rx_ant_idx"])
                    rec["steering_idx"] = int(extra["steering_idx"])
                    rec["fit_score"] = float(extra.get("fit_score", np.nan))
                    rec["u_hat"] = np.asarray(extra["u_hat"], dtype=np.complex128)
                    rec["u_hat_raw"] = np.asarray(extra["u_hat_raw"], dtype=np.complex128)
                    rec["alpha_hat_mode"] = np.complex128(extra["alpha_hat_mode"])
                    rec["alpha_hat_raw"] = np.complex128(extra["alpha_hat_raw"])
                    rec["h_hat_vec_mode"] = np.asarray(extra["h_hat_vec_mode"], dtype=np.complex128)
                    rec["h_hat_vec_raw"] = np.asarray(extra["h_hat_vec_raw"], dtype=np.complex128)
                    fit_nonneg = float(np.nan_to_num(extra.get("fit_score", np.nan), nan=0.0, posinf=0.0, neginf=0.0))
                    score_nonneg = float(np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0))
                    rec["selection_score"] = score_nonneg * max(fit_nonneg, 0.0)

                    ant_idx_use = int(extra["rx_ant_idx"])
                    if ant_idx_use >= 0:
                        h_hat_all_mode[int(rx_i), ant_idx_use, int(t), :] = rec["h_hat_vec_mode"]
                        h_hat_all_raw[int(rx_i), ant_idx_use, int(t), :] = rec["h_hat_vec_raw"]
                else:
                    rec["fit_score"] = float("nan")
                    rec["selection_score"] = float("nan")
                pair_hat[key] = rec

    det_pairs = sorted(pair_hat.keys(), key=lambda k: (k[1], k[0]))
    if len(det_pairs) > 0:
        pair_rx_idx = np.array([k[0] for k in det_pairs], dtype=int)
        pair_t_idx = np.array([k[1] for k in det_pairs], dtype=int)
        pair_bs_idx = np.array([pair_hat[k]["bs"] for k in det_pairs], dtype=int)
        pair_sector_idx = np.array([pair_hat[k]["sec"] for k in det_pairs], dtype=int)
        pair_phi_hat_deg = np.array([pair_hat[k]["phi_hat_deg"] for k in det_pairs], dtype=float)
        pair_theta_hat_deg = np.array([pair_hat[k]["theta_hat_deg"] for k in det_pairs], dtype=float)
        pair_rx_ant_idx = np.array([int(pair_hat[k].get("rx_ant_idx", -1)) for k in det_pairs], dtype=int)
        pair_score_user = np.array(
            [float(pair_hat[k].get("score_user", pair_hat[k].get("score", np.nan))) for k in det_pairs],
            dtype=float,
        )
        pair_fit_score = np.array(
            [float(pair_hat[k].get("fit_score", np.nan)) for k in det_pairs],
            dtype=float,
        )
        pair_selection_score = np.array(
            [float(pair_hat[k].get("selection_score", np.nan)) for k in det_pairs],
            dtype=float,
        )
        pair_alpha_hat_mode = np.array(
            [np.complex128(pair_hat[k].get("alpha_hat_mode", np.nan + 1j * np.nan)) for k in det_pairs],
            dtype=np.complex128,
        )
        pair_alpha_hat_raw = np.array(
            [np.complex128(pair_hat[k].get("alpha_hat_raw", np.nan + 1j * np.nan)) for k in det_pairs],
            dtype=np.complex128,
        )
        nan_vec = np.full((num_tx_ant,), np.nan + 1j * np.nan, dtype=np.complex128)
        pair_u_hat = np.vstack(
            [
                np.asarray(pair_hat[k].get("u_hat", nan_vec), dtype=np.complex128).reshape(1, -1)
                for k in det_pairs
            ]
        )
        pair_u_hat_raw = np.vstack(
            [
                np.asarray(pair_hat[k].get("u_hat_raw", nan_vec), dtype=np.complex128).reshape(1, -1)
                for k in det_pairs
            ]
        )
        pair_h_hat_vec_mode = np.vstack(
            [
                np.asarray(pair_hat[k].get("h_hat_vec_mode", nan_vec), dtype=np.complex128).reshape(1, -1)
                for k in det_pairs
            ]
        )
        pair_h_hat_vec_raw = np.vstack(
            [
                np.asarray(pair_hat[k].get("h_hat_vec_raw", nan_vec), dtype=np.complex128).reshape(1, -1)
                for k in det_pairs
            ]
        )
    else:
        pair_rx_idx = np.empty((0,), dtype=int)
        pair_t_idx = np.empty((0,), dtype=int)
        pair_bs_idx = np.empty((0,), dtype=int)
        pair_sector_idx = np.empty((0,), dtype=int)
        pair_phi_hat_deg = np.empty((0,), dtype=float)
        pair_theta_hat_deg = np.empty((0,), dtype=float)
        pair_rx_ant_idx = np.empty((0,), dtype=int)
        pair_score_user = np.empty((0,), dtype=float)
        pair_fit_score = np.empty((0,), dtype=float)
        pair_selection_score = np.empty((0,), dtype=float)
        pair_alpha_hat_mode = np.empty((0,), dtype=np.complex128)
        pair_alpha_hat_raw = np.empty((0,), dtype=np.complex128)
        pair_u_hat = np.empty((0, num_tx_ant), dtype=np.complex128)
        pair_u_hat_raw = np.empty((0, num_tx_ant), dtype=np.complex128)
        pair_h_hat_vec_mode = np.empty((0, num_tx_ant), dtype=np.complex128)
        pair_h_hat_vec_raw = np.empty((0, num_tx_ant), dtype=np.complex128)

    if len(detected_rx_indices_by_tx) > 0:
        detected_rx_indices_unique = np.unique(np.concatenate(list(detected_rx_indices_by_tx.values())))
    else:
        detected_rx_indices_unique = np.empty((0,), dtype=int)

    return {
        "pair_hat": pair_hat,
        "pair_rx_idx": pair_rx_idx,
        "pair_t_idx": pair_t_idx,
        "pair_bs_idx": pair_bs_idx,
        "pair_sector_idx": pair_sector_idx,
        "pair_phi_hat_deg": pair_phi_hat_deg,
        "pair_theta_hat_deg": pair_theta_hat_deg,
        "pair_rx_ant_idx": pair_rx_ant_idx,
        "pair_score_user": pair_score_user,
        "pair_fit_score": pair_fit_score,
        "pair_selection_score": pair_selection_score,
        "pair_u_hat": pair_u_hat,
        "pair_u_hat_raw": pair_u_hat_raw,
        "pair_alpha_hat_mode": pair_alpha_hat_mode,
        "pair_alpha_hat_raw": pair_alpha_hat_raw,
        "pair_h_hat_vec_mode": pair_h_hat_vec_mode,
        "pair_h_hat_vec_raw": pair_h_hat_vec_raw,
        "h_hat_all_mode": h_hat_all_mode,
        "h_hat_all_raw": h_hat_all_raw,
        "h_hat_all": h_hat_all_raw,
        "selected_mode_record": np.asarray(selected_mode_record, dtype=object),
        "selected_manifold_record": np.asarray(selected_manifold_record, dtype=object),
        "selected_flatten_record": np.asarray(selected_flatten_record, dtype=object),
        "selected_scan_record": np.asarray(selected_scan_record, dtype=object),
        "selected_phi_offset_record": np.asarray(selected_phi_offset_record, dtype=float),
        "selected_fit_record": np.asarray(selected_fit_record, dtype=float),
        "num_sources_record": np.asarray(num_sources_record, dtype=int),
        "candidate_metric_log": candidate_metric_log,
        "candidate_fit_log": candidate_fit_log,
        "selected_metric_log": selected_metric_log,
        "detected_rx_indices_by_tx": detected_rx_indices_by_tx,
        "detected_rx_indices_unique": detected_rx_indices_unique,
        "steering_geometry_source": np.array(
            "sionna_array_positions" if exact_array_geometry else "ideal_upa",
            dtype=object,
        ),
        "tx_orientations_rad": np.asarray(orientation_by_tx, dtype=np.float64),
    }
