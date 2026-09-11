from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import json
import multipath_support as mps
from scipy.optimize import linear_sum_assignment

from BeamformingCalc import nulling_bf, nulling_bf_music_noncoh, svd_bf
from ntn_music_detection import (
    array_position_steering_global,
    build_ntn_truth_from_paths,
    blind_music_detection_name,
    collapse_cir_to_narrowband,
    run_music_standard_pipeline,
    summarize_ntn_music_quality,
)


def _safe_db(power_linear: np.ndarray | float, eps: float = 1e-12) -> np.ndarray | float:
    arr = np.asarray(power_linear, dtype=np.float64)
    out = 10.0 * np.log10(np.maximum(arr, float(eps)))
    if np.isscalar(power_linear):
        return float(out)
    return out


def _scene_tx_array_positions_local(scene_config: Any) -> np.ndarray | None:
    """Read wavelength-normalized TX element positions from a Sionna scene."""
    scene = getattr(scene_config, "scene", None)
    tx_array = getattr(scene, "tx_array", None)
    positions = getattr(tx_array, "normalized_positions", None)
    if positions is None:
        return None

    components: List[np.ndarray] = []
    for axis in ("x", "y", "z"):
        value = getattr(positions, axis, None)
        if value is None:
            return None
        arr = value.numpy() if hasattr(value, "numpy") else np.asarray(value)
        components.append(np.asarray(arr, dtype=np.float64).reshape(-1))
    if not (components[0].size == components[1].size == components[2].size):
        raise ValueError("Sionna TX array position components have inconsistent lengths.")
    return np.column_stack(components)


def _interference_power_per_rx(h_ntn_tx: np.ndarray, beam: np.ndarray) -> np.ndarray:
    """Per-NTN received interference power for one TX beam."""
    h = np.asarray(h_ntn_tx, dtype=np.complex128)
    v = np.asarray(beam, dtype=np.complex128).reshape(-1)
    if h.ndim != 3:
        raise ValueError("h_ntn_tx must have shape (num_ntn_rx, num_ntn_rx_ant, num_tx_ant).")
    if h.shape[2] != v.shape[0]:
        raise ValueError(
            f"Beam dimension mismatch: h_ntn_tx has {h.shape[2]} TX antennas, beam has {v.shape[0]}."
        )
    # Match the legacy Lambda_CDF_det notebook metric:
    #   |w_t^H h_i|^2  (or |v_null^H h_i|^2),
    # where each per-user channel vector is treated as a TX-antenna column vector.
    eff = np.einsum("nra,a->nr", h, np.conjugate(v), optimize=True)
    return np.sum(np.abs(eff) ** 2, axis=1).real.astype(np.float64)


def _tn_link_power(
    h_tn_link: np.ndarray,
    beam: np.ndarray,
    rx_combiner: np.ndarray,
) -> float:
    """Received TN-link power |v^H H w_r|^2 for one BS-sector beam."""
    h = np.asarray(h_tn_link, dtype=np.complex128)
    v = np.asarray(beam, dtype=np.complex128)
    w_r = np.asarray(rx_combiner, dtype=np.complex128)

    if h.ndim != 2:
        raise ValueError("h_tn_link must have shape (num_tx_ant, num_tn_rx_ant).")
    if v.ndim == 1:
        v = v.reshape(-1, 1)
    if w_r.ndim == 1:
        w_r = w_r.reshape(-1, 1)
    if v.ndim != 2 or v.shape[1] != 1:
        raise ValueError(f"beam must have shape (num_tx_ant, 1); got {v.shape}.")
    if w_r.ndim != 2 or w_r.shape[1] != 1:
        raise ValueError(f"rx_combiner must have shape (num_tn_rx_ant, 1); got {w_r.shape}.")
    if h.shape[0] != v.shape[0]:
        raise ValueError(
            f"Beam dimension mismatch: h_tn_link has {h.shape[0]} TX antennas, beam has {v.shape[0]}."
        )
    if h.shape[1] != w_r.shape[0]:
        raise ValueError(
            "RX combiner dimension mismatch: "
            f"h_tn_link has {h.shape[1]} TN RX antennas, rx_combiner has {w_r.shape[0]}."
        )
    return float(np.abs((v.conj().T @ h @ w_r).item()) ** 2)


def _covariance_from_channel_vectors(
    h_vectors: np.ndarray,
    *,
    num_tx_ant: int,
) -> np.ndarray:
    """Build sum_k h_k h_k^H from channel vectors."""
    h = np.asarray(h_vectors, dtype=np.complex128)
    if h.size == 0:
        return np.zeros((num_tx_ant, num_tx_ant), dtype=np.complex128)
    if h.ndim == 1:
        h = h.reshape(1, -1)
    if h.ndim != 2 or h.shape[1] != num_tx_ant:
        raise ValueError(f"h_vectors must have shape (K, {num_tx_ant}); got {h.shape}.")
    return np.einsum("ka,kb->ab", h, np.conjugate(h), optimize=True)


def _channel_vectors_to_noncoh_terms(
    h_vectors: np.ndarray,
    *,
    num_tx_ant: int,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert channel vectors h_k into normalized directions u_k and powers g_k.

    This makes
        sum_k h_k h_k^H = sum_k g_k u_k u_k^H
    with g_k = ||h_k||^2 and ||u_k|| = 1.
    """
    h = np.asarray(h_vectors, dtype=np.complex128)
    if h.size == 0:
        return (
            np.empty((0, num_tx_ant), dtype=np.complex128),
            np.empty((0,), dtype=np.float64),
        )
    if h.ndim == 1:
        h = h.reshape(1, -1)
    if h.ndim != 2 or h.shape[1] != num_tx_ant:
        raise ValueError(f"h_vectors must have shape (K, {num_tx_ant}); got {h.shape}.")

    finite_rows = np.all(np.isfinite(np.real(h)) & np.isfinite(np.imag(h)), axis=1)
    h = h[finite_rows]
    if h.size == 0:
        return (
            np.empty((0, num_tx_ant), dtype=np.complex128),
            np.empty((0,), dtype=np.float64),
        )

    norms = np.linalg.norm(h, axis=1)
    valid = np.isfinite(norms) & (norms > float(eps))
    h = h[valid]
    norms = norms[valid]
    if h.size == 0:
        return (
            np.empty((0, num_tx_ant), dtype=np.complex128),
            np.empty((0,), dtype=np.float64),
        )

    u = h / norms[:, None]
    g = np.square(norms).astype(np.float64)
    return np.asarray(u, dtype=np.complex128), np.asarray(g, dtype=np.float64)


def _match_true_g_to_music_u(
    u_music: np.ndarray,
    u_true: np.ndarray,
    g_true: np.ndarray,
    *,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match true per-channel powers to anonymous MUSIC directions using only `u` correlation."""
    u_est = np.asarray(u_music, dtype=np.complex128)
    u_ref = np.asarray(u_true, dtype=np.complex128)
    g_ref = np.asarray(g_true, dtype=np.float64).reshape(-1)
    if u_est.ndim != 2 or u_ref.ndim != 2 or u_est.shape[1] != u_ref.shape[1]:
        raise ValueError(
            "u_music and u_true must be 2D arrays with the same antenna dimension; "
            f"got {u_est.shape} and {u_ref.shape}."
        )
    if u_ref.shape[0] != g_ref.size:
        raise ValueError("u_true and g_true must contain the same number of terms.")
    if u_est.shape[0] == 0:
        return (
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=int),
        )
    if u_ref.shape[0] == 0:
        return (
            np.zeros((u_est.shape[0],), dtype=np.float64),
            np.zeros((u_est.shape[0],), dtype=np.float64),
            np.full((u_est.shape[0],), -1, dtype=int),
        )

    est_norm = np.linalg.norm(u_est, axis=1, keepdims=True)
    ref_norm = np.linalg.norm(u_ref, axis=1, keepdims=True)
    u_est_n = u_est / np.maximum(est_norm, float(eps))
    u_ref_n = u_ref / np.maximum(ref_norm, float(eps))
    corr = np.abs(u_est_n @ np.conjugate(u_ref_n).T)

    matched_true_idx = np.full((u_est.shape[0],), -1, dtype=int)
    row_idx, col_idx = linear_sum_assignment(-corr)
    matched_true_idx[row_idx] = col_idx
    for est_idx in np.where(matched_true_idx < 0)[0]:
        matched_true_idx[int(est_idx)] = int(np.argmax(corr[int(est_idx)]))

    matched_g = g_ref[matched_true_idx]
    matched_rho = np.clip(corr[np.arange(u_est.shape[0]), matched_true_idx], 0.0, 1.0)
    return (
        np.asarray(matched_g, dtype=np.float64),
        np.asarray(matched_rho, dtype=np.float64),
        np.asarray(matched_true_idx, dtype=int),
    )


def _rowwise_vector_correlation(
    ref_vectors: np.ndarray,
    est_vectors: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Per-row normalized correlation rho = |u_ref^H u_est| / (||u_ref|| ||u_est||)."""
    ref = np.asarray(ref_vectors, dtype=np.complex128)
    est = np.asarray(est_vectors, dtype=np.complex128)

    if ref.ndim == 1:
        ref = ref.reshape(1, -1)
    if est.ndim == 1:
        est = est.reshape(1, -1)
    if ref.shape != est.shape:
        raise ValueError(
            f"ref_vectors and est_vectors must have the same shape; got {ref.shape} and {est.shape}."
        )
    if ref.size == 0:
        return np.empty((0,), dtype=np.float64)

    ref_norm = np.linalg.norm(ref, axis=1)
    est_norm = np.linalg.norm(est, axis=1)
    denom = ref_norm * est_norm

    rho = np.zeros((ref.shape[0],), dtype=np.float64)
    valid = np.isfinite(ref_norm) & np.isfinite(est_norm) & (denom > float(eps))
    if np.any(valid):
        inner = np.sum(np.conjugate(ref[valid]) * est[valid], axis=1)
        rho[valid] = np.abs(inner) / np.maximum(denom[valid], float(eps))
    return np.clip(rho, 0.0, 1.0)


def summarize_sionna_manifold_alignment(
    h_ntn_all: np.ndarray,
    pair_map: Dict[Tuple[int, int], Tuple[float, float, int, int]],
    *,
    array_positions_local: np.ndarray,
    tx_orientations_rad: np.ndarray,
    rotation_order: str = "zyx",
    eps: float = 1e-12,
) -> Dict[str, float | int]:
    """Measure the grid-free correlation between Sionna channels and its array geometry."""
    h_ntn = np.asarray(h_ntn_all, dtype=np.complex128)
    positions = np.asarray(array_positions_local, dtype=np.float64)
    orientations = np.asarray(tx_orientations_rad, dtype=np.float64)
    if h_ntn.ndim != 4:
        raise ValueError("h_ntn_all must have shape (num_rx, num_rx_ant, num_tx, num_tx_ant).")
    if positions.shape != (h_ntn.shape[3], 3):
        raise ValueError(
            "array_positions_local does not match h_ntn_all; "
            f"got {positions.shape} for {h_ntn.shape[3]} TX antennas."
        )
    if orientations.shape != (h_ntn.shape[2], 3):
        raise ValueError(
            "tx_orientations_rad does not match h_ntn_all; "
            f"got {orientations.shape} for {h_ntn.shape[2]} transmitters."
        )

    rho_values: List[float] = []
    for (rx_idx, tx_idx), truth in pair_map.items():
        if not (0 <= int(rx_idx) < h_ntn.shape[0] and 0 <= int(tx_idx) < h_ntn.shape[2]):
            continue
        h_rx = np.asarray(h_ntn[int(rx_idx), :, int(tx_idx), :], dtype=np.complex128)
        ant_norms = np.linalg.norm(h_rx, axis=1)
        if ant_norms.size == 0:
            continue
        h_vec = h_rx[int(np.argmax(ant_norms))]
        h_norm = float(np.linalg.norm(h_vec))
        if not np.isfinite(h_norm) or h_norm <= float(eps):
            continue
        phi_deg, theta_deg = float(truth[0]), float(truth[1])
        a = array_position_steering_global(
            phi_deg,
            theta_deg,
            positions,
            orientation_rad=tuple(float(v) for v in orientations[int(tx_idx)]),
            phase_sign=1,
            rotation_order=str(rotation_order),
        ).reshape(-1)
        rho = np.abs(np.vdot(h_vec, a)) / max(float(np.linalg.norm(h_vec) * np.linalg.norm(a)), float(eps))
        if np.isfinite(rho):
            rho_values.append(float(np.clip(rho, 0.0, 1.0)))

    if len(rho_values) == 0:
        return {"count": 0, "rho_mean": float("nan"), "rho_median": float("nan"), "rho_min": float("nan")}
    rho_arr = np.asarray(rho_values, dtype=np.float64)
    return {
        "count": int(rho_arr.size),
        "rho_mean": float(np.mean(rho_arr)),
        "rho_median": float(np.median(rho_arr)),
        "rho_min": float(np.min(rho_arr)),
    }



def summarize_music_covariance_quality(
    h_ntn_all: np.ndarray,
    music_lookup: Dict[int, Dict[str, np.ndarray]],
    *,
    max_detected_b_terms: str | int | None = "all",
) -> Dict[str, float]:
    """Truth-only diagnostics including missed sources; never used by detection."""
    h = np.asarray(h_ntn_all, dtype=np.complex128)
    total_power = covered_power = error_squared = truth_squared = 0.0
    for tx in range(h.shape[2]):
        vectors = h[:, :, tx, :].reshape(-1, h.shape[3])
        u_true, g_true = _channel_vectors_to_noncoh_terms(vectors, num_tx_ant=h.shape[3])
        terms = _extract_tx_music_terms(
            music_lookup.get(tx, {}), num_tx_ant=h.shape[3],
            max_detected_b_terms=max_detected_b_terms,
        )
        u_est, g_est = terms["u"], terms["g"]
        truth = _covariance_from_channel_vectors(vectors, num_tx_ant=h.shape[3])
        estimate = np.einsum("k,ka,kb->ab", g_est, u_est, u_est.conj())
        error_squared += float(np.linalg.norm(estimate - truth) ** 2)
        truth_squared += float(np.linalg.norm(truth) ** 2)
        total_power += float(g_true.sum())
        active = g_est > 0
        if len(u_true) and np.any(active):
            unit_est = u_est[active] / np.maximum(np.linalg.norm(u_est[active], axis=1, keepdims=True), 1e-12)
            best = np.max(np.abs(u_true @ unit_est.conj().T), axis=1)
            covered_power += float(g_true[best >= 0.95].sum())
    return {
        "covariance_nrmse": float(np.sqrt(error_squared / truth_squared)) if truth_squared else float("nan"),
        "power_coverage_rho95": covered_power / total_power if total_power else float("nan"),
    }

def summarize_music_noncoh_quality(
    h_ntn_all: np.ndarray,
    music_lookup: Dict[int, Dict[str, np.ndarray]],
    *,
    max_detected_b_terms: str | int | None = "all",
    eps: float = 1e-12,
) -> Dict[str, float | int]:
    """Summarize per-simulation Blind MDL-MUSIC noncoherent `(u, g)` quality."""
    h_ntn = np.asarray(h_ntn_all, dtype=np.complex128)
    if h_ntn.ndim != 4:
        raise ValueError("h_ntn_all must have shape (num_ntn_rx, num_ntn_rx_ant, num_tx, num_tx_ant).")

    num_tx_total = int(h_ntn.shape[2])
    num_tx_ant = int(h_ntn.shape[3])
    u_rho_all: List[np.ndarray] = []
    g_rel_err_all: List[np.ndarray] = []
    matched_power_all: List[np.ndarray] = []
    tx_with_pairs = 0

    for tx_idx in range(num_tx_total):
        lookup = music_lookup.get(int(tx_idx), {})
        pair_rx_lookup = np.asarray(lookup.get("pair_rx", np.empty((0,), dtype=int)), dtype=int)
        pair_rx_ant_lookup = np.asarray(lookup.get("pair_rx_ant", np.empty((0,), dtype=int)), dtype=int)
        u_lookup = np.asarray(
            lookup.get("u", np.empty((0, num_tx_ant), dtype=np.complex128)),
            dtype=np.complex128,
        )
        g_lookup = np.asarray(lookup.get("g", np.empty((0,), dtype=np.float64)), dtype=np.float64).reshape(-1)

        anonymous_mode = (
            pair_rx_lookup.size == 0
            and pair_rx_ant_lookup.size == 0
            and u_lookup.ndim == 2
            and u_lookup.shape[0] == g_lookup.size
            and g_lookup.size > 0
        )

        if anonymous_mode:
            est_inputs = _extract_tx_music_terms(
                lookup,
                num_tx_ant=num_tx_ant,
                max_detected_b_terms=max_detected_b_terms,
            )
            h_true_all = np.asarray(h_ntn[:, :, tx_idx, :], dtype=np.complex128).reshape(-1, num_tx_ant)
            u_true_all, g_true_all = _channel_vectors_to_noncoh_terms(
                h_true_all,
                num_tx_ant=num_tx_ant,
                eps=eps,
            )
            if u_true_all.shape[0] == 0:
                continue

            u_est_t = np.asarray(est_inputs["u"], dtype=np.complex128)
            g_est_t = np.asarray(est_inputs["g"], dtype=np.float64)
            matched_g, u_rho_t, _ = _match_true_g_to_music_u(
                u_est_t, u_true_all, g_true_all, eps=eps,
            )
            g_rel_err_t = np.abs(g_est_t - matched_g) / np.maximum(matched_g, np.finfo(float).tiny)

            if len(u_rho_t) == 0:
                continue
            tx_with_pairs += 1
            matched_power_all.append(matched_g)
            u_rho_all.append(np.asarray(u_rho_t, dtype=np.float64))
            g_rel_err_all.append(np.asarray(g_rel_err_t, dtype=np.float64))
            continue

        tx_inputs = _extract_tx_detected_pairs(
            h_ntn[:, :, tx_idx, :],
            lookup,
            num_tx_ant=num_tx_ant,
            max_detected_b_terms=max_detected_b_terms,
        )
        h_true_t = np.asarray(tx_inputs["h_true"], dtype=np.complex128)
        if h_true_t.size == 0:
            continue

        u_est_t = np.asarray(tx_inputs["u"], dtype=np.complex128)
        g_est_t = np.asarray(tx_inputs["g"], dtype=np.float64).reshape(-1)
        u_true_t, g_true_t = _channel_vectors_to_noncoh_terms(
            h_true_t,
            num_tx_ant=num_tx_ant,
            eps=eps,
        )
        if u_true_t.shape != u_est_t.shape or g_true_t.shape != g_est_t.shape:
            raise ValueError(
                "Inconsistent MUSIC noncoherent summary shapes: "
                f"u_true={u_true_t.shape}, u_est={u_est_t.shape}, "
                f"g_true={g_true_t.shape}, g_est={g_est_t.shape}."
            )
        if u_true_t.shape[0] == 0:
            continue

        u_rho_t = _rowwise_vector_correlation(
            u_true_t,
            u_est_t,
            eps=eps,
        )
        g_rel_err_t = np.abs(g_true_t - g_est_t) / np.maximum(np.abs(g_true_t), np.finfo(float).tiny)
        valid = np.isfinite(u_rho_t) & np.isfinite(g_rel_err_t)
        if not np.any(valid):
            continue

        tx_with_pairs += 1
        u_rho_all.append(np.asarray(u_rho_t[valid], dtype=np.float64))
        g_rel_err_all.append(np.asarray(g_rel_err_t[valid], dtype=np.float64))
        matched_power_all.append(g_true_t[valid])

    if len(u_rho_all) == 0:
        return {
            "pairs": 0,
            "tx_with_pairs": 0,
            "u_rho_mean": float("nan"),
            "u_err_mean": float("nan"),
            "g_rel_err_mean": float("nan"),
        }

    u_rho_arr = np.concatenate(u_rho_all, axis=0)
    g_rel_err_arr = np.concatenate(g_rel_err_all, axis=0)
    matched_power = np.concatenate(matched_power_all)
    return {
        "pairs": int(u_rho_arr.size),
        "tx_with_pairs": int(tx_with_pairs),
        "u_rho_mean": float(np.mean(u_rho_arr)),
        "u_err_mean": float(np.mean(1.0 - u_rho_arr)),
        "g_rel_err_mean": float(np.mean(g_rel_err_arr)),
        "u_rho_median": float(np.median(u_rho_arr)),
        "u_rho_p10": float(np.quantile(u_rho_arr, 0.1)),
        "u_rho_power_weighted": float(np.average(u_rho_arr, weights=matched_power)),
        "g_rel_err_median": float(np.median(g_rel_err_arr)),
    }


def _mask_channel_tensor_to_pairs(
    h_ntn_tx: np.ndarray,
    pair_rx: np.ndarray,
    pair_rx_ant: np.ndarray,
) -> np.ndarray:
    """Keep only selected `(rx, rx_ant)` channel vectors and zero the rest."""
    h = np.asarray(h_ntn_tx, dtype=np.complex128)
    if h.ndim != 3:
        raise ValueError("h_ntn_tx must have shape (num_ntn_rx, num_ntn_rx_ant, num_tx_ant).")

    pair_rx_arr = np.asarray(pair_rx, dtype=int).reshape(-1)
    pair_rx_ant_arr = np.asarray(pair_rx_ant, dtype=int).reshape(-1)
    if pair_rx_arr.size != pair_rx_ant_arr.size:
        raise ValueError(
            f"pair_rx and pair_rx_ant must have the same length; got "
            f"{pair_rx_arr.size} and {pair_rx_ant_arr.size}."
        )

    masked = np.zeros_like(h)
    if pair_rx_arr.size == 0:
        return masked

    valid = (
        (pair_rx_arr >= 0)
        & (pair_rx_arr < h.shape[0])
        & (pair_rx_ant_arr >= 0)
        & (pair_rx_ant_arr < h.shape[1])
    )
    if not np.any(valid):
        return masked

    masked[pair_rx_arr[valid], pair_rx_ant_arr[valid], :] = h[
        pair_rx_arr[valid],
        pair_rx_ant_arr[valid],
        :,
    ]
    return masked


def _resolve_b_term_limit(max_detected_b_terms: str | int | None) -> int | None:
    """Normalize the user-facing detected-pair limit."""
    if max_detected_b_terms is None:
        return None

    if isinstance(max_detected_b_terms, str):
        value = max_detected_b_terms.strip().lower()
        if value == "all":
            return None
        if value.isdigit():
            max_detected_b_terms = int(value)
        else:
            raise ValueError(
                "max_detected_b_terms must be 'all', None, or a positive integer; "
                f"got {max_detected_b_terms!r}."
            )

    if isinstance(max_detected_b_terms, (int, np.integer)):
        limit = int(max_detected_b_terms)
        if limit <= 0:
            raise ValueError(
                f"max_detected_b_terms must be positive when numeric; got {limit}."
            )
        return limit

    raise ValueError(
        "max_detected_b_terms must be 'all', None, or a positive integer; "
        f"got type {type(max_detected_b_terms).__name__}."
    )


def _pair_rank_score(selection_score: np.ndarray, gain: np.ndarray) -> np.ndarray:
    score = np.asarray(selection_score, dtype=float).reshape(-1)
    gain_arr = np.asarray(gain, dtype=np.float64).reshape(-1)
    if score.size != gain_arr.size:
        raise ValueError(
            f"selection_score and gain must have the same length; got {score.size} and {gain_arr.size}."
        )
    rank_score = score.copy()
    invalid = ~np.isfinite(rank_score)
    if np.any(invalid):
        rank_score[invalid] = gain_arr[invalid]
    return rank_score


def _dedupe_tx_pair_candidates(
    pair_rx: np.ndarray,
    pair_rx_ant: np.ndarray,
    selection_score: np.ndarray,
    u: np.ndarray,
    g: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep at most one candidate for each `(rx, rx_ant)` pair within one TX."""
    pair_rx_arr = np.asarray(pair_rx, dtype=int).reshape(-1)
    pair_rx_ant_arr = np.asarray(pair_rx_ant, dtype=int).reshape(-1)
    selection_score_arr = np.asarray(selection_score, dtype=float).reshape(-1)
    u_arr = np.asarray(u, dtype=np.complex128)
    g_arr = np.asarray(g, dtype=np.float64).reshape(-1)

    if pair_rx_arr.size <= 1:
        return pair_rx_arr, pair_rx_ant_arr, selection_score_arr, u_arr, g_arr

    rank_score = _pair_rank_score(selection_score_arr, g_arr)
    best_idx_by_key: Dict[Tuple[int, int], int] = {}

    for idx, (rx_i, ant_i) in enumerate(zip(pair_rx_arr.tolist(), pair_rx_ant_arr.tolist())):
        key = (int(rx_i), int(ant_i))
        prev_idx = best_idx_by_key.get(key)
        if prev_idx is None or rank_score[idx] > rank_score[prev_idx]:
            best_idx_by_key[key] = idx

    keep_idx = np.array(sorted(best_idx_by_key.values()), dtype=int)
    return (
        pair_rx_arr[keep_idx],
        pair_rx_ant_arr[keep_idx],
        selection_score_arr[keep_idx],
        u_arr[keep_idx],
        g_arr[keep_idx],
    )


def pair_tn_to_strongest_tx(
    h_tn_all: np.ndarray,
    *,
    h_tn_th: float,
    tx_antennas: int,
    tx_power: float,
    snr_noise_power: float,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """Pair each TN to its strongest valid TX over all BS sectors."""
    h = np.asarray(h_tn_all, dtype=np.complex128)
    if h.ndim != 4:
        raise ValueError("h_tn_all must have shape (num_tn_rx, num_tn_rx_ant, num_tx, num_tx_ant).")

    num_tn_rx, _num_tn_rx_ant, num_tx_total, _num_tx_ant = h.shape
    h_flat = np.transpose(h, (0, 2, 1, 3)).reshape(num_tn_rx, num_tx_total, -1)
    h_norms = np.linalg.norm(h_flat, axis=2)

    nonzero_mask = h_norms > float(eps)
    valid_mask = nonzero_mask & (h_norms > float(h_tn_th))

    best_tx_idx = np.full((num_tn_rx,), -1, dtype=int)
    best_h_norm = np.zeros((num_tn_rx,), dtype=np.float64)
    pairs_by_tx: Dict[int, List[Dict[str, Any]]] = {int(t): [] for t in range(num_tx_total)}

    for tn_idx in range(num_tn_rx):
        tx_candidates = np.flatnonzero(valid_mask[tn_idx])
        if tx_candidates.size == 0:
            continue

        tx_local_best = tx_candidates[int(np.argmax(h_norms[tn_idx, tx_candidates]))]
        best_tx_idx[tn_idx] = int(tx_local_best)
        best_h_norm[tn_idx] = float(h_norms[tn_idx, tx_local_best])

        h_tn = np.asarray(h[tn_idx, :, tx_local_best, :], dtype=np.complex128).T
        w_t, w_r = svd_bf(h_tn, tx_antennas)
        snr_raw_linear = (
            np.abs((w_t.conj().T @ h_tn @ w_r).item()) ** 2
            * float(tx_power)
            / float(snr_noise_power)
        )

        pairs_by_tx[int(tx_local_best)].append(
            {
                "tn_idx": int(tn_idx),
                "tx_idx": int(tx_local_best),
                "h_tn": h_tn,
                "h_norm": float(best_h_norm[tn_idx]),
                "w_t": np.asarray(w_t, dtype=np.complex128),
                "w_r": np.asarray(w_r, dtype=np.complex128),
                "snr_raw_db": float(_safe_db(snr_raw_linear, eps=eps)),
            }
        )

    pair_counts_by_tx = np.array([len(pairs_by_tx[int(t)]) for t in range(num_tx_total)], dtype=int)
    min_count = int(pair_counts_by_tx.min()) if pair_counts_by_tx.size > 0 else 0

    return {
        "h_norms": h_norms,
        "nonzero_mask": nonzero_mask,
        "valid_mask": valid_mask,
        "best_tx_idx": best_tx_idx,
        "best_h_norm": best_h_norm,
        "pairs_by_tx": pairs_by_tx,
        "pair_counts_by_tx": pair_counts_by_tx,
        "min_count": min_count,
    }


def build_precoding_matrix_from_tx_beams(
    tx_name_list: Iterable[str],
    tx_beams: Dict[int, np.ndarray],
    *,
    nsect: int,
    num_tx_ant: int | None = None,
    dtype: np.dtype = np.complex64,
    apply_conjugate: bool = True,
) -> np.ndarray:
    """Map per-TX beam vectors into scene-transmitter order.

    By default, this applies complex conjugation before exporting the beam.
    This matches the legacy notebook/Sionna convention where the optimization
    uses vectors in expressions like ``v^H H w_r`` while Sionna's precoder
    expects the actual TX weight vector ``conj(v)``.
    """
    tx_names = [str(name) for name in tx_name_list]
    if num_tx_ant is None:
        if len(tx_beams) == 0:
            raise ValueError("num_tx_ant is required when tx_beams is empty.")
        num_tx_ant = int(
            np.asarray(next(iter(tx_beams.values())), dtype=np.complex128).reshape(-1).shape[0]
        )
    else:
        num_tx_ant = int(num_tx_ant)

    precoding = np.zeros((len(tx_names), num_tx_ant), dtype=dtype)
    for row_idx, name in enumerate(tx_names):
        if not name.startswith("tx-"):
            continue
        parts = name.split("-")
        if len(parts) != 3:
            continue
        try:
            bs_idx = int(parts[1])
            sec_idx = int(parts[2])
        except ValueError:
            continue
        tx_idx = int(bs_idx) * int(nsect) + int(sec_idx)
        beam = tx_beams.get(int(tx_idx))
        if beam is None:
            continue
        beam_vec = np.asarray(beam, dtype=np.complex128).reshape(-1)
        if beam_vec.shape[0] != num_tx_ant:
            raise ValueError(
                f"Beam dimension mismatch for tx_idx={tx_idx}: "
                f"expected {num_tx_ant}, got {beam_vec.shape[0]}."
            )
        if bool(apply_conjugate):
            beam_vec = np.conjugate(beam_vec)
        precoding[row_idx, :] = beam_vec.astype(dtype, copy=False)
    return precoding


def build_music_tx_lookup(
    ntn_music_out: Dict[str, Any],
    *,
    num_ntn_rx: int,
    num_tx_total: int,
    num_tx_ant: int,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Collect per-TX Blind MDL-MUSIC terms for downstream nulling.

    In blind mode the primary detector outputs are anonymous `peak_*` terms.
    The paired-link fields are only a fallback for legacy evaluation paths.
    """
    peak_t = np.asarray(ntn_music_out.get("peak_t_idx", []), dtype=int)
    peak_u = np.asarray(ntn_music_out.get("peak_u_hat_raw", []), dtype=np.complex128)
    peak_g = np.asarray(ntn_music_out.get("peak_g_hat", []), dtype=np.float64)
    peak_selection_score = np.asarray(ntn_music_out.get("peak_selection_score", []), dtype=float)
    detected_rx_by_tx_raw = ntn_music_out.get("detected_rx_indices_by_tx", {})

    if peak_t.size > 0:
        if peak_u.ndim == 1:
            if peak_u.size == num_tx_ant:
                peak_u = peak_u.reshape(1, -1)
            else:
                raise ValueError(f"Unexpected peak_u shape: {peak_u.shape}")
        if peak_u.ndim != 2:
            raise ValueError(f"peak_u_hat must be 2D after reshape, got {peak_u.shape}")
        if peak_u.shape[1] != num_tx_ant:
            raise ValueError(
                f"peak_u_hat antenna dimension mismatch: expected {num_tx_ant}, got {peak_u.shape[1]}."
            )
        if not (peak_t.size == peak_u.shape[0] == peak_g.size):
            raise ValueError("Inconsistent blind detector peak lengths in ntn_music_out.")
        if peak_selection_score.size not in (0, peak_t.size):
            raise ValueError("peak_selection_score length does not match blind peak count.")

        lookup: Dict[int, Dict[str, np.ndarray]] = {}
        for tx_idx in range(num_tx_total):
            tx_mask = peak_t == int(tx_idx)
            if peak_selection_score.size > 0:
                selection_score_t = np.asarray(peak_selection_score[tx_mask], dtype=float)
            else:
                selection_score_t = np.full((int(np.count_nonzero(tx_mask)),), np.nan, dtype=float)

            detected_rx_t = np.asarray(
                detected_rx_by_tx_raw.get(int(tx_idx), np.empty((0,), dtype=int)),
                dtype=int,
            )
            detected_rx_t = detected_rx_t[(detected_rx_t >= 0) & (detected_rx_t < num_ntn_rx)]
            lookup[int(tx_idx)] = {
                "rx_detected": np.unique(detected_rx_t).astype(int),
                "pair_rx": np.empty((0,), dtype=int),
                "pair_rx_ant": np.empty((0,), dtype=int),
                "selection_score": selection_score_t,
                "u": np.asarray(peak_u[tx_mask], dtype=np.complex128),
                "g": np.asarray(peak_g[tx_mask], dtype=np.float64),
            }
        return lookup

    pair_rx = np.asarray(ntn_music_out.get("pair_rx_idx", []), dtype=int)
    pair_t = np.asarray(ntn_music_out.get("pair_t_idx", []), dtype=int)
    # pair_u = np.asarray(ntn_music_out.get("pair_u_hat", []), dtype=np.complex128)
    pair_u = np.asarray(ntn_music_out.get("pair_u_hat_raw", []), dtype=np.complex128)
    pair_alpha_hat = np.asarray(ntn_music_out.get("pair_alpha_hat_raw", []), dtype=np.complex128)
    pair_rx_ant_idx = np.asarray(ntn_music_out.get("pair_rx_ant_idx", []), dtype=int)
    pair_score_user = np.asarray(ntn_music_out.get("pair_score_user", []), dtype=float)
    pair_fit_score = np.asarray(ntn_music_out.get("pair_fit_score", []), dtype=float)
    pair_selection_score = np.asarray(ntn_music_out.get("pair_selection_score", []), dtype=float)

    if pair_u.ndim == 1:
        if pair_u.size == 0:
            pair_u = np.empty((0, num_tx_ant), dtype=np.complex128)
        elif pair_u.size == num_tx_ant:
            pair_u = pair_u.reshape(1, -1)
        else:
            raise ValueError(f"Unexpected pair_u shape: {pair_u.shape}")
    if pair_u.ndim != 2:
        raise ValueError(f"pair_u_hat must be 2D after reshape, got {pair_u.shape}")
    if pair_u.shape[1] != num_tx_ant and pair_u.shape[0] > 0:
        raise ValueError(
            f"pair_u_hat antenna dimension mismatch: expected {num_tx_ant}, got {pair_u.shape[1]}."
        )

    if not (
        pair_rx.size
        == pair_t.size
        == pair_rx_ant_idx.size
        == pair_alpha_hat.size
        == pair_u.shape[0]
    ):
        raise ValueError("Inconsistent MUSIC pair lengths in ntn_music_out.")
    if pair_score_user.size not in (0, pair_rx.size):
        raise ValueError("pair_score_user length does not match MUSIC pair count.")
    if pair_fit_score.size not in (0, pair_rx.size):
        raise ValueError("pair_fit_score length does not match MUSIC pair count.")
    if pair_selection_score.size not in (0, pair_rx.size):
        raise ValueError("pair_selection_score length does not match MUSIC pair count.")

    lookup: Dict[int, Dict[str, np.ndarray]] = {}
    for tx_idx in range(num_tx_total):
        tx_mask = pair_t == int(tx_idx)
        if not np.any(tx_mask):
            lookup[int(tx_idx)] = {
                "rx_detected": np.empty((0,), dtype=int),
                "pair_rx": np.empty((0,), dtype=int),
                "pair_rx_ant": np.empty((0,), dtype=int),
                "selection_score": np.empty((0,), dtype=np.float64),
                "u": np.empty((0, num_tx_ant), dtype=np.complex128),
                "g": np.empty((0,), dtype=np.float64),
            }
            continue

        pair_rx_t = np.asarray(pair_rx[tx_mask], dtype=int)
        pair_rx_ant_t = np.asarray(pair_rx_ant_idx[tx_mask], dtype=int)
        rx_valid = pair_rx_t[(pair_rx_t >= 0) & (pair_rx_t < num_ntn_rx)]
        rx_detected = np.unique(rx_valid)

        selection_score_t = (
            np.asarray(pair_selection_score[tx_mask], dtype=float)
            if pair_selection_score.size > 0
            else np.full((int(np.count_nonzero(tx_mask)),), np.nan, dtype=float)
        )
        if not np.any(np.isfinite(selection_score_t)):
            score_user_t = (
                np.asarray(pair_score_user[tx_mask], dtype=float)
                if pair_score_user.size > 0
                else np.full((int(np.count_nonzero(tx_mask)),), np.nan, dtype=float)
            )
            fit_score_t = (
                np.asarray(pair_fit_score[tx_mask], dtype=float)
                if pair_fit_score.size > 0
                else np.full((int(np.count_nonzero(tx_mask)),), np.nan, dtype=float)
            )
            selection_score_t = np.maximum(
                np.nan_to_num(score_user_t, nan=0.0, posinf=0.0, neginf=0.0),
                0.0,
            ) * np.maximum(
                np.nan_to_num(fit_score_t, nan=0.0, posinf=0.0, neginf=0.0),
                0.0,
            )

        u_t = np.asarray(pair_u[tx_mask], dtype=np.complex128)
        alpha_t = np.asarray(pair_alpha_hat[tx_mask], dtype=np.complex128)
        g_t = np.abs(alpha_t) ** 2

        finite_u = np.all(np.isfinite(np.real(u_t)) & np.isfinite(np.imag(u_t)), axis=1)
        finite_g = np.isfinite(g_t)
        keep = finite_u & finite_g

        lookup[int(tx_idx)] = {
            "rx_detected": rx_detected.astype(int),
            "pair_rx": np.asarray(pair_rx_t[keep], dtype=int),
            "pair_rx_ant": np.asarray(pair_rx_ant_t[keep], dtype=int),
            "selection_score": np.asarray(selection_score_t[keep], dtype=float),
            "u": np.asarray(u_t[keep], dtype=np.complex128),
            "g": np.asarray(g_t[keep], dtype=np.float64),
        }

    return lookup


def _extract_tx_music_terms(
    lookup: Dict[str, np.ndarray],
    *,
    num_tx_ant: int,
    max_detected_b_terms: str | int | None = "all",
) -> Dict[str, np.ndarray]:
    """Collect one TX's deduped MUSIC pair entries for legacy-style est nulling."""
    pair_rx_t = np.asarray(lookup.get("pair_rx", np.empty((0,), dtype=int)), dtype=int)
    pair_rx_ant_t = np.asarray(lookup.get("pair_rx_ant", np.empty((0,), dtype=int)), dtype=int)
    selection_score_t = np.asarray(
        lookup.get("selection_score", np.empty((0,), dtype=float)),
        dtype=float,
    )
    u_t = np.asarray(
        lookup.get("u", np.empty((0, num_tx_ant), dtype=np.complex128)),
        dtype=np.complex128,
    )
    g_t = np.asarray(
        lookup.get("g", np.empty((0,), dtype=np.float64)),
        dtype=np.float64,
    )

    if u_t.ndim == 1:
        if u_t.size == 0:
            u_t = np.empty((0, num_tx_ant), dtype=np.complex128)
        elif u_t.size == num_tx_ant:
            u_t = u_t.reshape(1, -1)
        else:
            raise ValueError(f"Unexpected u shape for one TX: {u_t.shape}")
    if u_t.ndim != 2 or u_t.shape[1] != num_tx_ant:
        raise ValueError(f"u must have shape (K, {num_tx_ant}) for one TX; got {u_t.shape}.")

    anonymous_mode = pair_rx_t.size == 0 and pair_rx_ant_t.size == 0 and u_t.shape[0] == g_t.size
    if not anonymous_mode:
        if not (pair_rx_t.size == pair_rx_ant_t.size == u_t.shape[0] == g_t.size):
            raise ValueError(
                "Inconsistent per-TX MUSIC lookup lengths: "
                f"pair_rx={pair_rx_t.size}, pair_rx_ant={pair_rx_ant_t.size}, u={u_t.shape[0]}, g={g_t.size}."
            )
    if selection_score_t.size not in (0, g_t.size):
        raise ValueError(
            "Inconsistent per-TX selection score length: "
            f"selection_score={selection_score_t.size}, g={g_t.size}."
        )
    if selection_score_t.size == 0 and g_t.size > 0:
        selection_score_t = np.full((g_t.size,), np.nan, dtype=float)

    if not anonymous_mode:
        pair_rx_t, pair_rx_ant_t, selection_score_t, u_t, g_t = _dedupe_tx_pair_candidates(
            pair_rx_t,
            pair_rx_ant_t,
            selection_score_t,
            u_t,
            g_t,
        )

    max_terms = _resolve_b_term_limit(max_detected_b_terms)
    if max_terms is not None and g_t.size > max_terms:
        rank_score = _pair_rank_score(selection_score_t, g_t)
        keep_idx = np.argsort(-rank_score, kind="mergesort")[:max_terms]
        keep_idx = np.sort(keep_idx)
        if not anonymous_mode:
            pair_rx_t = pair_rx_t[keep_idx]
            pair_rx_ant_t = pair_rx_ant_t[keep_idx]
        selection_score_t = selection_score_t[keep_idx]
        u_t = u_t[keep_idx]
        g_t = g_t[keep_idx]

    return {
        "pair_rx": np.asarray(pair_rx_t, dtype=int),
        "pair_rx_ant": np.asarray(pair_rx_ant_t, dtype=int),
        "selection_score": np.asarray(selection_score_t, dtype=float),
        "u": np.asarray(u_t, dtype=np.complex128),
        "g": np.asarray(g_t, dtype=np.float64),
    }


def _extract_tx_detected_pairs(
    h_ntn_tx: np.ndarray,
    lookup: Dict[str, np.ndarray],
    *,
    num_tx_ant: int,
    max_detected_b_terms: str | int | None = "all",
) -> Dict[str, np.ndarray]:
    """Collect one TX's detected NTN pair set for true/music-real nulling."""
    h_ntn_t = np.asarray(h_ntn_tx, dtype=np.complex128)
    if h_ntn_t.ndim != 3:
        raise ValueError("h_ntn_tx must have shape (num_ntn_rx, num_ntn_rx_ant, num_tx_ant).")
    if h_ntn_t.shape[2] != int(num_tx_ant):
        raise ValueError(
            f"h_ntn_tx antenna dimension mismatch: expected {num_tx_ant}, got {h_ntn_t.shape[2]}."
        )

    est_inputs = _extract_tx_music_terms(
        lookup,
        num_tx_ant=num_tx_ant,
        max_detected_b_terms=max_detected_b_terms,
    )
    pair_rx_t = np.asarray(est_inputs["pair_rx"], dtype=int)
    pair_rx_ant_t = np.asarray(est_inputs["pair_rx_ant"], dtype=int)
    u_t = np.asarray(est_inputs["u"], dtype=np.complex128)
    g_t = np.asarray(est_inputs["g"], dtype=np.float64)

    valid = (
        (pair_rx_t >= 0)
        & (pair_rx_t < h_ntn_t.shape[0])
        & (pair_rx_ant_t >= 0)
        & (pair_rx_ant_t < h_ntn_t.shape[1])
    )
    if np.any(valid):
        pair_rx_keep = np.asarray(pair_rx_t[valid], dtype=int)
        pair_rx_ant_keep = np.asarray(pair_rx_ant_t[valid], dtype=int)
        u_keep = np.asarray(u_t[valid], dtype=np.complex128)
        g_keep = np.asarray(g_t[valid], dtype=np.float64)
        h_true_keep = np.asarray(
            h_ntn_t[pair_rx_keep, pair_rx_ant_keep, :],
            dtype=np.complex128,
        )
    else:
        pair_rx_keep = np.empty((0,), dtype=int)
        pair_rx_ant_keep = np.empty((0,), dtype=int)
        u_keep = np.empty((0, num_tx_ant), dtype=np.complex128)
        g_keep = np.empty((0,), dtype=np.float64)
        h_true_keep = np.empty((0, num_tx_ant), dtype=np.complex128)

    h_eval = _mask_channel_tensor_to_pairs(h_ntn_t, pair_rx_keep, pair_rx_ant_keep)
    selected_rx = (
        np.unique(pair_rx_keep).astype(int)
        if pair_rx_keep.size > 0
        else np.empty((0,), dtype=int)
    )

    return {
        "pair_rx": pair_rx_keep,
        "pair_rx_ant": pair_rx_ant_keep,
        "selected_rx": selected_rx,
        "u": u_keep,
        "g": g_keep,
        "h_true": h_true_keep,
        "h_eval": h_eval,
    }


def run_small_round(
    h_tn_all: np.ndarray,
    h_ntn_all: np.ndarray,
    *,
    pairs_by_tx: Dict[int, List[Dict[str, Any]]],
    music_lookup: Dict[int, Dict[str, np.ndarray]],
    sim_idx: int | None = None,
    round_idx: int,
    lambda_ranges: Iterable[float] | None = None,
    lambda_ranges_music_est: Iterable[float] | None = None,
    lambda_ranges_music_real: Iterable[float] | None = None,
    tx_power: float,
    snr_noise_power: float,
    inr_noise_power: float,
    max_detected_b_terms: str | int | None = "all",
    print_music_u_corr: bool = False,
    oracle_path_channels: np.ndarray | None = None,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """Run one small round with true, MUSIC, and oracle comparison modes."""
    h_tn_all_arr = np.asarray(h_tn_all, dtype=np.complex128)
    h_ntn = np.asarray(h_ntn_all, dtype=np.complex128)
    if h_tn_all_arr.ndim != 4:
        raise ValueError("h_tn_all must have shape (num_tn_rx, num_tn_rx_ant, num_tx, num_tx_ant).")
    if h_ntn.ndim != 4:
        raise ValueError("h_ntn_all must have shape (num_ntn_rx, num_ntn_rx_ant, num_tx, num_tx_ant).")

    num_tn_rx = int(h_tn_all_arr.shape[0])
    num_ntn_rx = int(h_ntn.shape[0])
    num_tx_total = int(h_ntn.shape[2])
    lambda_list = [] if lambda_ranges is None else [float(v) for v in lambda_ranges]
    lambda_list_music_est = (
        [] if lambda_ranges_music_est is None else [float(v) for v in lambda_ranges_music_est]
    )
    lambda_list_music_real = [float(v) for v in (lambda_ranges_music_real or [])]

    raw_snr_db: List[float] = []
    raw_sinr_db: List[float] = []
    true_snr_db: Dict[float, List[float]] = {lambda_: [] for lambda_ in lambda_list}
    true_sinr_db: Dict[float, List[float]] = {lambda_: [] for lambda_ in lambda_list}
    est_snr_db: Dict[float, List[float]] = {lambda_: [] for lambda_ in lambda_list_music_est}
    est_sinr_db: Dict[float, List[float]] = {lambda_: [] for lambda_ in lambda_list_music_est}
    music_real_snr_db: Dict[float, List[float]] = {lambda_: [] for lambda_ in lambda_list_music_real}
    music_real_sinr_db: Dict[float, List[float]] = {lambda_: [] for lambda_ in lambda_list_music_real}
    raw_inr_power = np.zeros((num_ntn_rx,), dtype=np.float64)
    true_inr_power = {lambda_: np.zeros((num_ntn_rx,), dtype=np.float64) for lambda_ in lambda_list}
    est_inr_power = {lambda_: np.zeros((num_ntn_rx,), dtype=np.float64) for lambda_ in lambda_list_music_est}
    music_real_inr_power = {
        lambda_: np.zeros((num_ntn_rx,), dtype=np.float64) for lambda_ in lambda_list_music_real
    }
    interfered_rx_mask = np.any(np.abs(h_ntn) > eps, axis=(1, 2, 3))
    detected_rx_mask = np.zeros((num_ntn_rx,), dtype=bool)
    eval_mask = np.asarray(interfered_rx_mask, dtype=bool).copy()
    round_pairs: Dict[int, Dict[str, Any]] = {}
    raw_beams: Dict[int, np.ndarray] = {}
    true_beams: Dict[float, Dict[int, np.ndarray]] = {lambda_: {} for lambda_ in lambda_list}
    est_beams: Dict[float, Dict[int, np.ndarray]] = {lambda_: {} for lambda_ in lambda_list_music_est}
    music_real_beams: Dict[float, Dict[int, np.ndarray]] = {
        lambda_: {} for lambda_ in lambda_list_music_real
    }

    for tx_idx in range(num_tx_total):
        tx_pairs = pairs_by_tx.get(int(tx_idx), [])
        if round_idx >= len(tx_pairs):
            raise ValueError(
                f"round_idx={round_idx} exceeds paired TN count for tx_idx={tx_idx}. "
                "Check min_count scheduling."
            )

        pair = tx_pairs[round_idx]
        h_tn = np.asarray(pair["h_tn"], dtype=np.complex128)
        w_t = np.asarray(pair["w_t"], dtype=np.complex128)
        w_r = np.asarray(pair["w_r"], dtype=np.complex128)
        tn_idx = int(pair["tn_idx"])
        if tn_idx < 0 or tn_idx >= num_tn_rx:
            raise ValueError(f"Invalid TN index for round pair: tn_idx={tn_idx}, num_tn_rx={num_tn_rx}.")
        round_pairs[int(tx_idx)] = {
            "tn_idx": tn_idx,
            "h_tn": h_tn,
            "w_r": w_r,
        }
        raw_beams[int(tx_idx)] = w_t
        raw_snr_db.append(float(pair["snr_raw_db"]))

        h_ntn_tx = np.asarray(h_ntn[:, :, tx_idx, :], dtype=np.complex128)
        lookup = music_lookup.get(int(tx_idx), {})
        rx_detected_t = np.asarray(lookup.get("rx_detected", np.empty((0,), dtype=int)), dtype=int)
        if rx_detected_t.size > 0:
            detected_rx_mask[rx_detected_t] = True

        est_inputs = _extract_tx_music_terms(
            lookup,
            num_tx_ant=h_tn.shape[0],
            max_detected_b_terms=max_detected_b_terms,
        )
        u_t = np.asarray(est_inputs["u"], dtype=np.complex128)
        g_t = np.asarray(est_inputs["g"], dtype=np.float64)
        h_ntn_eval_tx = np.asarray(h_ntn_tx, dtype=np.complex128)
        h_true_flat = np.asarray(h_ntn_tx, dtype=np.complex128).reshape(-1, h_tn.shape[0])
        finite_rows = np.all(np.isfinite(np.real(h_true_flat)) & np.isfinite(np.imag(h_true_flat)), axis=1)
        nonzero_rows = np.linalg.norm(h_true_flat, axis=1) > float(eps)
        h_true_t = np.asarray(h_true_flat[finite_rows & nonzero_rows], dtype=np.complex128)

        interference_term_true = _covariance_from_channel_vectors(
            h_true_t,
            num_tx_ant=h_tn.shape[0],
        )
        oracle_vectors = h_true_t if oracle_path_channels is None else np.asarray(
            oracle_path_channels[:, :, tx_idx, :], dtype=np.complex128).reshape(-1, h_tn.shape[0])
        u_true_t, g_true_t = _channel_vectors_to_noncoh_terms(
            oracle_vectors,
            num_tx_ant=h_tn.shape[0],
            eps=eps,
        )
        raw_inr_power += _interference_power_per_rx(h_ntn_tx, w_t)

        for lambda_ in lambda_list:
            v_null_true, _, _, _ = nulling_bf(h_tn, w_r, interference_term_true, lambda_)
            true_beams[lambda_][int(tx_idx)] = np.asarray(v_null_true, dtype=np.complex128)

            true_snr_linear = (
                np.abs((v_null_true.conj().T @ h_tn @ w_r).item()) ** 2
                * float(tx_power)
                / float(snr_noise_power)
            )

            true_snr_db[lambda_].append(float(_safe_db(true_snr_linear, eps=eps)))
            true_inr_power[lambda_] += _interference_power_per_rx(h_ntn_eval_tx, v_null_true)

        for lambda_ in lambda_list_music_est:
            v_null_est, _, _, _ = nulling_bf_music_noncoh(h_tn, w_r, u_t, g_t, lambda_, eps=eps)
            est_beams[lambda_][int(tx_idx)] = np.asarray(v_null_est, dtype=np.complex128)
            est_snr_linear = (
                np.abs((v_null_est.conj().T @ h_tn @ w_r).item()) ** 2
                * float(tx_power)
                / float(snr_noise_power)
            )
            est_snr_db[lambda_].append(float(_safe_db(est_snr_linear, eps=eps)))
            est_inr_power[lambda_] += _interference_power_per_rx(h_ntn_tx, v_null_est)


        for lambda_ in lambda_list_music_real:
            v_null_music_real, _, _, _ = nulling_bf_music_noncoh(h_tn, w_r, u_true_t, g_true_t,  lambda_, eps=eps)
            music_real_beams[lambda_][int(tx_idx)] = np.asarray(v_null_music_real, dtype=np.complex128)
            music_real_snr_linear = (
                np.abs((v_null_music_real.conj().T @ h_tn @ w_r).item()) ** 2
                * float(tx_power)
                / float(snr_noise_power)
            )
            music_real_snr_db[lambda_].append(float(_safe_db(music_real_snr_linear, eps=eps)))
            music_real_inr_power[lambda_] += _interference_power_per_rx(h_ntn_eval_tx, v_null_music_real)

    for tx_idx in range(num_tx_total):
        pair = round_pairs[int(tx_idx)]
        tn_idx = int(pair["tn_idx"])
        h_tn = np.asarray(pair["h_tn"], dtype=np.complex128)
        w_r = np.asarray(pair["w_r"], dtype=np.complex128)

        desired_raw = _tn_link_power(h_tn, raw_beams[int(tx_idx)], w_r)
        interf_raw = 0.0
        for other_tx in range(num_tx_total):
            if int(other_tx) == int(tx_idx):
                continue
            h_interf = np.asarray(h_tn_all_arr[tn_idx, :, other_tx, :], dtype=np.complex128).T
            interf_raw += _tn_link_power(h_interf, raw_beams[int(other_tx)], w_r)
        raw_sinr_linear = desired_raw * float(tx_power) / (
            float(snr_noise_power) + interf_raw * float(tx_power)
        )
        raw_sinr_db.append(float(_safe_db(raw_sinr_linear, eps=eps)))

        for lambda_ in lambda_list:
            beam_true = true_beams[lambda_][int(tx_idx)]
            desired_true = _tn_link_power(h_tn, beam_true, w_r)
            interf_true = 0.0
            for other_tx in range(num_tx_total):
                if int(other_tx) == int(tx_idx):
                    continue
                h_interf = np.asarray(h_tn_all_arr[tn_idx, :, other_tx, :], dtype=np.complex128).T
                interf_true += _tn_link_power(h_interf, true_beams[lambda_][int(other_tx)], w_r)
            true_sinr_linear = desired_true * float(tx_power) / (
                float(snr_noise_power) + interf_true * float(tx_power)
            )
            true_sinr_db[lambda_].append(float(_safe_db(true_sinr_linear, eps=eps)))

        for lambda_ in lambda_list_music_est:
            beam_est = est_beams[lambda_][int(tx_idx)]
            desired_est = _tn_link_power(h_tn, beam_est, w_r)
            interf_est = 0.0
            for other_tx in range(num_tx_total):
                if int(other_tx) == int(tx_idx):
                    continue
                h_interf = np.asarray(h_tn_all_arr[tn_idx, :, other_tx, :], dtype=np.complex128).T
                interf_est += _tn_link_power(h_interf, est_beams[lambda_][int(other_tx)], w_r)
            est_sinr_linear = desired_est * float(tx_power) / (
                float(snr_noise_power) + interf_est * float(tx_power)
            )
            est_sinr_db[lambda_].append(float(_safe_db(est_sinr_linear, eps=eps)))


        for lambda_ in lambda_list_music_real:
            beam_music_real = music_real_beams[lambda_][int(tx_idx)]
            desired_music_real = _tn_link_power(h_tn, beam_music_real, w_r)
            interf_music_real = 0.0
            for other_tx in range(num_tx_total):
                if int(other_tx) == int(tx_idx):
                    continue
                h_interf = np.asarray(h_tn_all_arr[tn_idx, :, other_tx, :], dtype=np.complex128).T
                interf_music_real += _tn_link_power(
                    h_interf,
                    music_real_beams[lambda_][int(other_tx)],
                    w_r,
                )
            music_real_sinr_linear = desired_music_real * float(tx_power) / (
                float(snr_noise_power) + interf_music_real * float(tx_power)
            )
            music_real_sinr_db[lambda_].append(float(_safe_db(music_real_sinr_linear, eps=eps)))

    inr_eval_mask = np.asarray(interfered_rx_mask, dtype=bool)
    raw_inr_db = (
        _safe_db(raw_inr_power[inr_eval_mask] * float(tx_power) / float(inr_noise_power), eps=eps)
        if np.any(inr_eval_mask)
        else np.empty((0,), dtype=np.float64)
    )
    true_inr_db = {
        lambda_: (
            _safe_db(
                true_inr_power[lambda_][inr_eval_mask] * float(tx_power) / float(inr_noise_power),
                eps=eps,
            )
            if np.any(inr_eval_mask)
            else np.empty((0,), dtype=np.float64)
        )
        for lambda_ in lambda_list
    }
    est_inr_db = {
        lambda_: (
            _safe_db(
                est_inr_power[lambda_][inr_eval_mask] * float(tx_power) / float(inr_noise_power),
                eps=eps,
            )
            if np.any(inr_eval_mask)
            else np.empty((0,), dtype=np.float64)
        )
        for lambda_ in lambda_list_music_est
    }
    music_real_inr_db = {
        lambda_: (
            _safe_db(
                music_real_inr_power[lambda_][inr_eval_mask] * float(tx_power) / float(inr_noise_power),
                eps=eps,
            )
            if np.any(inr_eval_mask)
            else np.empty((0,), dtype=np.float64)
        )
        for lambda_ in lambda_list_music_real
    }

    return {
        "raw_snr_db": np.asarray(raw_snr_db, dtype=np.float64),
        "raw_sinr_db": np.asarray(raw_sinr_db, dtype=np.float64),
        "true_snr_db": {lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in true_snr_db.items()},
        "true_sinr_db": {
            lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in true_sinr_db.items()
        },
        "est_snr_db": {lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in est_snr_db.items()},
        "est_sinr_db": {
            lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in est_sinr_db.items()
        },
        "music_real_snr_db": {
            lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in music_real_snr_db.items()
        },
        "music_real_sinr_db": {
            lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in music_real_sinr_db.items()
        },
        "raw_inr_db": np.asarray(raw_inr_db, dtype=np.float64),
        "true_inr_db": {lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in true_inr_db.items()},
        "est_inr_db": {lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in est_inr_db.items()},
        "music_real_inr_db": {
            lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in music_real_inr_db.items()
        },
        "round_pairs": {
            int(tx_idx): {
                "tn_idx": int(pair["tn_idx"]),
                "h_tn": np.asarray(pair["h_tn"], dtype=np.complex128),
                "w_r": np.asarray(pair["w_r"], dtype=np.complex128),
            }
            for tx_idx, pair in round_pairs.items()
        },
        "raw_beams": {
            int(tx_idx): np.asarray(beam, dtype=np.complex128)
            for tx_idx, beam in raw_beams.items()
        },
        "true_beams": {
            float(lambda_): {
                int(tx_idx): np.asarray(beam, dtype=np.complex128)
                for tx_idx, beam in beam_dict.items()
            }
            for lambda_, beam_dict in true_beams.items()
        },
        "est_beams": {
            float(lambda_): {
                int(tx_idx): np.asarray(beam, dtype=np.complex128)
                for tx_idx, beam in beam_dict.items()
            }
            for lambda_, beam_dict in est_beams.items()
        },
        "music_real_beams": {
            float(lambda_): {
                int(tx_idx): np.asarray(beam, dtype=np.complex128)
                for tx_idx, beam in beam_dict.items()
            }
            for lambda_, beam_dict in music_real_beams.items()
        },
        "detected_mask": detected_rx_mask,
        "eval_mask": eval_mask,
        "detected_count": int(np.count_nonzero(detected_rx_mask)),
    }


def validate_ul_frequency_percentages(values: Iterable[float]) -> list[float]:
    """Percent units: +5 means 1.05*f_DL; zero is a co-frequency control."""
    percentages = [float(v) for v in values]
    if not percentages or not all(np.isfinite(p) and p > -100 for p in percentages):
        raise ValueError("Provide a nonempty list of finite UL percentages greater than -100.")
    if len(set(percentages)) != len(percentages):
        raise ValueError("UL percentages must be unique.")
    return percentages


def transfer_music_peaks_to_dl(
    music_out: Dict[str, Any], *, dl_array_positions: np.ndarray,
    music_kwargs: Dict[str, Any], num_tx: int,
) -> Dict[str, Any]:
    """Rebuild blind UL peaks on the DL manifold, without using DL channel truth.

    Estimated UL powers/selection scores are retained. This is angular transfer,
    not instantaneous FDD CSI reciprocity or frequency-dependent gain prediction.
    """
    from ntn_music_detection import _resolve_tx_orientations

    orientations = _resolve_tx_orientations(
        num_tx=num_tx, nsect=int(music_kwargs.get("nsect", 3)),
        use_sector_orientation=bool(music_kwargs.get("use_sector_orientation", True)),
        sector_yaw_offset_rad=float(music_kwargs.get("sector_yaw_offset_rad", 0)),
        sector_pitch_rad=float(music_kwargs.get("sector_pitch_rad", 0)),
        sector_roll_rad=float(music_kwargs.get("sector_roll_rad", 0)),
        tx_orientations_rad=music_kwargs.get("tx_orientations_rad"),
    )
    tx_indices = np.asarray(music_out["peak_t_idx"], dtype=int)
    phi = np.asarray(music_out["peak_phi_hat_deg"], dtype=float)
    theta = np.asarray(music_out["peak_theta_hat_deg"], dtype=float)
    dl_vectors = np.empty((len(tx_indices), len(dl_array_positions)), dtype=np.complex128)
    phi_offset = float(np.round(float(music_kwargs.get("phi_offset_deg", 0)) % 360, 1))
    for i, tx_idx in enumerate(tx_indices):
        azimuth = (phi[i] - phi_offset) % 360
        if music_kwargs.get("phi_mirror_about_sector", False):
            azimuth = (2 * np.rad2deg(orientations[tx_idx, 0]) - azimuth) % 360
        dl_vectors[i] = array_position_steering_global(
            azimuth, theta[i], array_positions_local=dl_array_positions,
            orientation_rad=orientations[tx_idx], phase_sign=1.0,
            rotation_order=str(music_kwargs.get("rotation_order", "zyx")),
        ).ravel()
    # Only fields consumed by build_music_tx_lookup are transferred; other
    # fields remain UL diagnostics and are deliberately not passed as DL CSI.
    result = dict(music_out)
    result["peak_u_hat_raw"] = dl_vectors
    return result


def run_nulling_cdf_experiment(
    scene_config: Any,
    *,
    num_macro_sims: int,
    compute_positions_kwargs: Dict[str, Any],
    compute_paths_kwargs: Dict[str, Any],
    lambda_ranges: Iterable[float] | None = None,
    lambda_ranges_music_est: Iterable[float] | None = None,
    lambda_ranges_music_real: Iterable[float] | None = None,
    h_tn_th: float,
    tx_antennas: int,
    tx_power: float,
    snr_noise_power: float,
    inr_noise_power: float,
    music_kwargs: Dict[str, Any],
    sionna_phi_is_global: bool = True,
    theta_display_mode: str = "elevation",
    eps: float = 1e-12,
    plot_first_sim_only: bool = True,
    show_progress: bool = True,
    print_music_u_corr: bool = True,
    resample_satellite_per_macro: bool = False,
    satellite_azimuth_range_deg: Tuple[float, float] = (0.0, 360.0),
    satellite_elevation_range_deg: Tuple[float, float] = (35.0, 90.0),
    satellite_rng_seed: int | None = None,
    max_detected_b_terms: str | int | None = "all",
    channel_cache_dir: str | Path | None = None,
    position_rng_seed: int | None = None,
    ul_frequency_percentages: Iterable[float] | None = None,
    ul_to_dl_mode: str = "angle",
    multipath: int = 0,
    multipath_music_kwargs: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run macro simulations, optionally sweeping UL frequencies on fixed DL drops.

    With ul_frequency_percentages=None the legacy single-frequency API is kept.
    Otherwise return by_percentage[pct], each containing the legacy metrics for
    every lambda. DL channels and TN scheduling are shared across percentages.
    Angle transfer rebuilds DL steering with UL-estimated angles, retaining UL
    power weights (no oracle DL gain calibration). raw mode reuses UL vectors.
    """
    if multipath not in (0, 1):
        raise ValueError("multipath must be 0 or 1.")
    compute_paths_kwargs = dict(compute_paths_kwargs)
    if multipath:
        compute_paths_kwargs["multipath"] = 1
        if int(compute_paths_kwargs.get("max_depth", 0)) < 1:
            raise ValueError("multipath=1 requires a positive max_depth.")
    elif compute_paths_kwargs.get("multipath", 0):
        raise ValueError("Experiment and path generation multipath switches disagree.")
    percentages = validate_ul_frequency_percentages(
        [0.0] if ul_frequency_percentages is None else ul_frequency_percentages
    )
    if ul_to_dl_mode not in {"angle", "raw"}:
        raise ValueError("ul_to_dl_mode must be 'angle' or 'raw'.")
    if ul_frequency_percentages is not None and music_kwargs.get("pair_keys") is not None:
        raise ValueError("The FDD sweep requires blind MUSIC (pair_keys=None).")
    dl_frequency_hz = float(compute_paths_kwargs.get("fc", getattr(scene_config, "fc", 1.0)))
    if ul_frequency_percentages is not None and "fc" not in compute_paths_kwargs:
        raise ValueError("The FDD sweep requires an explicit DL fc in compute_paths_kwargs.")
    if not np.isfinite(dl_frequency_hz) or dl_frequency_hz <= 0:
        raise ValueError("DL frequency must be finite and positive.")
    if int(num_macro_sims) <= 0:
        raise ValueError("num_macro_sims must be positive.")
    resolved_b_term_limit = _resolve_b_term_limit(max_detected_b_terms)

    try:
        from tqdm.auto import trange
    except Exception:
        trange = None

    lambda_list = [] if lambda_ranges is None else [float(v) for v in lambda_ranges]
    lambda_list_music_est = (
        [float(v) for v in lambda_ranges_music_est]
        if lambda_ranges_music_est is not None
        else list(lambda_list)
    )
    lambda_list_music_real = [float(v) for v in (lambda_ranges_music_real or [])]
    case_accumulators = {}
    for percentage in percentages:
        raw_snr_all: List[float] = []
        raw_sinr_all: List[float] = []
        raw_inr_all: List[float] = []
        true_snr_all: Dict[float, List[float]] = {lambda_: [] for lambda_ in lambda_list}
        true_sinr_all: Dict[float, List[float]] = {lambda_: [] for lambda_ in lambda_list}
        est_snr_all: Dict[float, List[float]] = {lambda_: [] for lambda_ in lambda_list_music_est}
        est_sinr_all: Dict[float, List[float]] = {lambda_: [] for lambda_ in lambda_list_music_est}
        music_real_snr_all: Dict[float, List[float]] = {
            lambda_: [] for lambda_ in lambda_list_music_real
        }
        music_real_sinr_all: Dict[float, List[float]] = {
            lambda_: [] for lambda_ in lambda_list_music_real
        }
        true_inr_all: Dict[float, List[float]] = {lambda_: [] for lambda_ in lambda_list}
        est_inr_all: Dict[float, List[float]] = {lambda_: [] for lambda_ in lambda_list_music_est}
        music_real_inr_all: Dict[float, List[float]] = {
            lambda_: [] for lambda_ in lambda_list_music_real
        }
        macro_stats: List[Dict[str, Any]] = []
        case_accumulators[percentage] = {
            "raw_snr_all": raw_snr_all,
            "raw_sinr_all": raw_sinr_all,
            "raw_inr_all": raw_inr_all,
            "true_snr_all": true_snr_all,
            "true_sinr_all": true_sinr_all,
            "est_snr_all": est_snr_all,
            "est_sinr_all": est_sinr_all,
            "music_real_snr_all": music_real_snr_all,
            "music_real_sinr_all": music_real_sinr_all,
            "true_inr_all": true_inr_all,
            "est_inr_all": est_inr_all,
            "music_real_inr_all": music_real_inr_all,
            "macro_stats": macro_stats,
        }
    bs_pos_ref: np.ndarray | None = None
    sat_rng = (
        np.random.default_rng(satellite_rng_seed)
        if bool(resample_satellite_per_macro)
        else None
    )

    iterator = (
        trange(int(num_macro_sims), desc="Monte Carlo", leave=False)
        if show_progress and trange is not None
        else range(int(num_macro_sims))
    )

    for sim_idx in iterator:
        pos_kwargs = dict(compute_positions_kwargs)
        if plot_first_sim_only and sim_idx > 0:
            for key in ("plot_grid", "plot_bs", "plot_tn", "plot_ntn"):
                if key in pos_kwargs:
                    pos_kwargs[key] = False

        if sat_rng is not None:
            sat_azimuth_deg = float(
                sat_rng.uniform(
                    float(satellite_azimuth_range_deg[0]),
                    float(satellite_azimuth_range_deg[1]),
                )
            )
            sat_elevation_deg = float(
                sat_rng.uniform(
                    float(satellite_elevation_range_deg[0]),
                    float(satellite_elevation_range_deg[1]),
                )
            )
            pos_kwargs["azimuth"] = sat_azimuth_deg
            pos_kwargs["elevation"] = sat_elevation_deg
        else:
            sat_azimuth_deg = float(pos_kwargs["azimuth"])
            sat_elevation_deg = float(pos_kwargs["elevation"])

        if position_rng_seed is None:
            scene_config.compute_positions(**pos_kwargs)
        else:
            random_state = np.random.get_state()
            try:
                np.random.seed(np.random.SeedSequence([int(position_rng_seed), int(sim_idx)]).generate_state(1)[0])
                scene_config.compute_positions(**pos_kwargs)
            finally:
                np.random.set_state(random_state)
        tx_pos = np.asarray(scene_config.tx_pos, dtype=np.float64)
        sat_look_pos = np.asarray(scene_config.ntn_look_pos, dtype=np.float64).copy()
        if bs_pos_ref is None:
            bs_pos_ref = tx_pos.copy()
        elif tx_pos.shape != bs_pos_ref.shape or not np.allclose(tx_pos, bs_pos_ref):
            raise RuntimeError(
                "BS positions changed across macro simulations. "
                "The requested experiment assumes fixed BS positions."
            )

        scene_config.compute_paths(**compute_paths_kwargs)
        h_tn_all = collapse_cir_to_narrowband(scene_config.a_tn)
        h_ntn_all = collapse_cir_to_narrowband(scene_config.a_ntn)
        dl_path_catalog = None
        if multipath:
            dl_path_catalog = mps.build_path_catalog(
                scene_config.paths_ntn, scene_config.a_ntn, scene_config.tau_ntn)

        music_kwargs_sim = dict(music_kwargs)
        scene_array_positions = _scene_tx_array_positions_local(scene_config)
        if scene_array_positions is not None:
            if scene_array_positions.shape[0] != h_ntn_all.shape[3]:
                raise ValueError(
                    "Sionna TX array geometry does not match the channel tensor: "
                    f"positions={scene_array_positions.shape}, tx_ant={h_ntn_all.shape[3]}."
                )
            music_kwargs_sim["array_positions_local"] = scene_array_positions

        scene_tx_orientations = getattr(scene_config, "tx_orientation_rad", None)
        if scene_tx_orientations is not None:
            scene_tx_orientations = np.asarray(scene_tx_orientations, dtype=np.float64)
            if scene_tx_orientations.shape != (h_ntn_all.shape[2], 3):
                raise ValueError(
                    "Scene TX orientations do not match the channel tensor: "
                    f"orientations={scene_tx_orientations.shape}, num_tx={h_ntn_all.shape[2]}."
                )
            if bool(music_kwargs_sim.get("use_sector_orientation", True)):
                music_kwargs_sim["tx_orientations_rad"] = scene_tx_orientations

        if channel_cache_dir is not None:
            cache_dir = Path(channel_cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"channels_{int(sim_idx):04d}.npz"
            # Exclusive creation prevents accidentally replacing a comparison input.
            with cache_path.open("xb") as cache_file:
                np.savez_compressed(
                    cache_file, h_tn=h_tn_all, h_ntn=h_ntn_all, tx_pos=tx_pos,
                    tn_pos=np.asarray(scene_config.tn_pos),
                    ntn_pos=np.asarray(scene_config.rx_ntn_pos),
                    satellite_look_pos=sat_look_pos,
                    satellite_angles_deg=np.array([sat_azimuth_deg, sat_elevation_deg]),
                    array_positions_local=(scene_array_positions if scene_array_positions is not None else np.empty((0, 3))),
                    tx_orientations_rad=(scene_tx_orientations if scene_tx_orientations is not None else np.empty((0, 3))),
                    noise_var=float(music_kwargs_sim.get("detect_noise_var", 0.0)),
                )

        if multipath and channel_cache_dir is not None:
            with (Path(channel_cache_dir) / f"paths_dl_{sim_idx:04d}.npz").open("xb") as stream:
                np.savez_compressed(
                    stream, a_tn=scene_config.a_tn, tau_tn=scene_config.tau_tn,
                    a_ntn=scene_config.a_ntn, tau_ntn=scene_config.tau_ntn,
                    **{key: value for key, value in dl_path_catalog.items() if key != "path_channels"})

        pairing = pair_tn_to_strongest_tx(
            h_tn_all,
            h_tn_th=float(h_tn_th),
            tx_antennas=int(tx_antennas),
            tx_power=float(tx_power),
            snr_noise_power=float(snr_noise_power),
            eps=eps,
        )
        min_count = int(pairing["min_count"])
        num_ntn_rx = int(h_ntn_all.shape[0])
        num_tx_total = int(h_ntn_all.shape[2])
        num_tx_ant = int(h_ntn_all.shape[3])

        for case_index, percentage in enumerate(percentages):
            raw_snr_all = case_accumulators[percentage]["raw_snr_all"]
            raw_sinr_all = case_accumulators[percentage]["raw_sinr_all"]
            raw_inr_all = case_accumulators[percentage]["raw_inr_all"]
            true_snr_all = case_accumulators[percentage]["true_snr_all"]
            true_sinr_all = case_accumulators[percentage]["true_sinr_all"]
            est_snr_all = case_accumulators[percentage]["est_snr_all"]
            est_sinr_all = case_accumulators[percentage]["est_sinr_all"]
            music_real_snr_all = case_accumulators[percentage]["music_real_snr_all"]
            music_real_sinr_all = case_accumulators[percentage]["music_real_sinr_all"]
            true_inr_all = case_accumulators[percentage]["true_inr_all"]
            est_inr_all = case_accumulators[percentage]["est_inr_all"]
            music_real_inr_all = case_accumulators[percentage]["music_real_inr_all"]
            macro_stats = case_accumulators[percentage]["macro_stats"]
            sensing_frequency_hz = dl_frequency_hz * (1.0 + percentage / 100.0)
            if percentage == 0.0:
                sensing_paths, sensing_cir = scene_config.paths_ntn, scene_config.a_ntn
                sensing_tau = getattr(scene_config, "tau_ntn", None)
                h_ntn_sensing = h_ntn_all
            else:
                sensing_paths, sensing_cir, sensing_tau = scene_config.compute_ntn_ul_paths(
                    sensing_frequency_hz, max_depth=int(compute_paths_kwargs.get("max_depth", 3)),
                )
                h_ntn_sensing = collapse_cir_to_narrowband(sensing_cir)
            sensing_array_positions = (
                None if scene_array_positions is None else
                scene_array_positions * (sensing_frequency_hz / dl_frequency_hz)
            )
            if percentage != 0.0 and sensing_array_positions is None:
                raise ValueError("FDD sensing requires exact scene array positions.")
            if sensing_array_positions is not None:
                music_kwargs_sim["array_positions_local"] = sensing_array_positions
            case_cache_dir = None
            if channel_cache_dir is not None:
                case_cache_dir = Path(channel_cache_dir)
                if ul_frequency_percentages is not None:
                    case_cache_dir = case_cache_dir / f"ul_{case_index:03d}"
                case_cache_dir.mkdir(parents=True, exist_ok=True)
                if ul_frequency_percentages is not None:
                    with (case_cache_dir / f"sensing_{sim_idx:04d}.npz").open("xb") as stream:
                        np.savez_compressed(
                            stream, h_ntn_ul=h_ntn_sensing,
                            ul_frequency_hz=sensing_frequency_hz, dl_frequency_hz=dl_frequency_hz,
                            ul_frequency_percent=percentage,
                            array_positions_local_ul=sensing_array_positions,
                        )
            if multipath:
                ntn_music_out = mps.run_multipath_music_pipeline(
                    h_ntn_sensing, music_kwargs=music_kwargs_sim,
                    **(multipath_music_kwargs or {}))
            else:
                ntn_music_out = run_music_standard_pipeline(h_ntn_sensing, **music_kwargs_sim)
            detection_name = str(
                ntn_music_out.get(
                    "detection_name",
                    blind_music_detection_name(
                        str(music_kwargs_sim.get("detect_source_estimation", "mdl"))
                    ),
                )
            )
            path_metrics_ul, path_metrics_dl = {}, {}
            if multipath:
                ul_catalog = mps.build_path_catalog(sensing_paths, sensing_cir, sensing_tau)
                path_metrics_ul = mps.summarize_path_estimates(
                    ul_catalog, ntn_music_out, nsect=int(music_kwargs_sim["nsect"]),
                    sionna_phi_is_global=bool(sionna_phi_is_global))
                path_metrics_dl = mps.summarize_path_estimates(
                    dl_path_catalog, ntn_music_out, nsect=int(music_kwargs_sim["nsect"]),
                    sionna_phi_is_global=bool(sionna_phi_is_global))
                # UE IDs are truth-only diagnostic matches, never detector inputs.
                ntn_music_out["detected_rx_indices_unique"] = np.asarray(
                    path_metrics_ul["detected_rx_indices"], dtype=int)
                manifold_alignment = summarize_sionna_manifold_alignment(
                    ul_catalog["path_channels"],
                    mps.path_truth_map(ul_catalog, nsect=int(music_kwargs_sim["nsect"]),
                                       sionna_phi_is_global=bool(sionna_phi_is_global)),
                    array_positions_local=sensing_array_positions,
                    tx_orientations_rad=scene_tx_orientations,
                    rotation_order=str(music_kwargs_sim.get("rotation_order", "zyx")), eps=eps)
                music_quality = dict(angle_metrics=path_metrics_ul,
                                     detected_subset_metrics={}, detected_pairs_summary={})
                if case_cache_dir is not None:
                    with (case_cache_dir / f"paths_ul_{sim_idx:04d}.npz").open("xb") as stream:
                        np.savez_compressed(stream, a=sensing_cir, tau_raw=sensing_tau,
                            **{key: value for key, value in ul_catalog.items() if key != "path_channels"})
            else:
                ntn_truth = build_ntn_truth_from_paths(
                    sensing_paths,
                    sensing_cir,
                    num_tx_total=num_tx_total,
                    nsect=int(music_kwargs_sim["nsect"]),
                    sionna_phi_is_global=bool(sionna_phi_is_global),
                )
                if scene_array_positions is not None and scene_tx_orientations is not None:
                    manifold_alignment = summarize_sionna_manifold_alignment(
                        h_ntn_sensing,
                        ntn_truth["pair_map"],
                        array_positions_local=sensing_array_positions,
                        tx_orientations_rad=scene_tx_orientations,
                        rotation_order=str(music_kwargs_sim.get("rotation_order", "zyx")),
                        eps=eps,
                    )
                else:
                    manifold_alignment = {
                        "count": 0,
                        "rho_mean": float("nan"),
                        "rho_median": float("nan"),
                        "rho_min": float("nan"),
                    }
                music_quality = summarize_ntn_music_quality(
                    h_ntn_sensing,
                    ntn_music_out,
                    ntn_truth["pair_map"],
                    theta_display_mode=str(theta_display_mode),
                    eps=eps,
                )
            downlink_music_out = transfer_music_peaks_to_dl(
                ntn_music_out, dl_array_positions=scene_array_positions,
                music_kwargs=music_kwargs_sim, num_tx=num_tx_total,
            ) if percentage != 0.0 and ul_to_dl_mode == "angle" else ntn_music_out
            music_lookup = build_music_tx_lookup(
                downlink_music_out,
                num_ntn_rx=num_ntn_rx,
                num_tx_total=num_tx_total,
                num_tx_ant=num_tx_ant,
            )

            detected_rx_union = np.asarray(ntn_music_out.get("detected_rx_indices_unique", []), dtype=int)
            interfered_ntn_count = int(np.count_nonzero(np.any(np.abs(h_ntn_all) > eps, axis=(1, 2, 3))))
            pair_counts_by_tx = np.asarray(pairing["pair_counts_by_tx"], dtype=int)
            diagnostic_channels = dl_path_catalog["path_channels"] if multipath else h_ntn_all
            noncoh_metrics = summarize_music_noncoh_quality(
                diagnostic_channels,
                music_lookup,
                max_detected_b_terms=resolved_b_term_limit,
                eps=eps,
            )
            noncoh_metrics.update(summarize_music_covariance_quality(
                diagnostic_channels, music_lookup, max_detected_b_terms=resolved_b_term_limit,
            ))
            for key, source in (("fit_before", "covariance_fit_before"), ("fit_after", "covariance_fit_after"),
                                ("accepted_peak_mean", "accepted_peak_counts")):
                values = np.asarray(ntn_music_out.get(source, []), dtype=float)
                values = values[np.isfinite(values)]
                noncoh_metrics[key] = float(values.mean()) if values.size else float("nan")
            if channel_cache_dir is not None:
                with (case_cache_dir / f"music_{int(sim_idx):04d}.npz").open("xb") as diagnostics_file:
                    np.savez_compressed(diagnostics_file,
                        peak_u_used_for_dl=np.asarray(downlink_music_out.get("peak_u_hat_raw", [])),
                        ul_to_dl_mode=np.asarray(ul_to_dl_mode), **{
                        key: np.asarray(ntn_music_out[key]) for key in (
                            "peak_t_idx", "peak_phi_hat_deg", "peak_theta_hat_deg", "peak_u_hat_raw", "peak_g_hat",
                            "num_sources_record", "accepted_peak_counts", "covariance_fit_before", "covariance_fit_after",
                            "candidate_t_idx", "candidate_phi_hat_deg", "candidate_theta_hat_deg", "candidate_g_hat",
                            "candidate_counts", "correlated_source_covariance", "correlated_fit_condition",
                            "peak_phi_physical_deg",
                            "noise_power_estimate", "ul_covariance", "smoothed_ul_covariance", "raw_signal_rank",
                            "smoothing_subarray_shape", "smoothing_num_subarrays",
                        ) if key in ntn_music_out
                    })
            source_count_metrics: Dict[str, Any] = {
                "method": str(np.asarray(ntn_music_out.get("source_count_method", "unknown")).item()),
            }
            for label, key in (
                ("used", "num_sources_record"),
                ("mdl", "num_sources_mdl_record"),
                ("rank", "num_sources_rank_record"),
                ("eigengap", "num_sources_eigengap_record"),
            ):
                values = np.asarray(ntn_music_out.get(key, np.empty((0,), dtype=int)), dtype=np.float64)
                values = values[np.isfinite(values)]
                source_count_metrics[f"{label}_mean"] = (
                    float(np.mean(values)) if values.size > 0 else float("nan")
                )
                source_count_metrics[f"{label}_sum"] = (
                    int(np.sum(values)) if values.size > 0 else 0
                )
            angle_metrics = music_quality["angle_metrics"]
            if print_music_u_corr:
                steering_source = str(
                    np.asarray(ntn_music_out.get("steering_geometry_source", "ideal_upa")).item()
                )
                peak_sep_phi = float(
                    np.asarray(ntn_music_out.get("peak_min_sep_phi_deg", np.nan)).item()
                )
                peak_sep_theta = float(
                    np.asarray(ntn_music_out.get("peak_min_sep_theta_deg", np.nan)).item()
                )
                peak_corr = float(
                    np.asarray(ntn_music_out.get("peak_max_correlation", np.nan)).item()
                )
                peak_refine = bool(
                    np.asarray(ntn_music_out.get("peak_refine", False)).item()
                )
                print(
                    f"[{detection_name} sim] "
                    f"sim={int(sim_idx)} UL_offset={percentage:+g}% "
                    f"sat_az_deg={float(sat_azimuth_deg):.2f} sat_el_deg={float(sat_elevation_deg):.2f} "
                    f"steering={steering_source} peak_sep=({peak_sep_phi:.1f},{peak_sep_theta:.1f})deg "
                    f"peak_corr={peak_corr:.3f} refine={peak_refine} "
                    f"Kavg[used/mdl/rank/gap]="
                    f"{float(source_count_metrics['used_mean']):.2f}/"
                    f"{float(source_count_metrics['mdl_mean']):.2f}/"
                    f"{float(source_count_metrics['rank_mean']):.2f}/"
                    f"{float(source_count_metrics['eigengap_mean']):.2f} "
                    f"det_ntn={int(detected_rx_union.size)} interf_ntn={int(interfered_ntn_count)} "
                    f"matched_pairs={int(angle_metrics.get('matched_pairs', 0))} "
                    f"manifold_rho={float(manifold_alignment['rho_mean']):.6f} "
                    f"u_pairs={int(noncoh_metrics['pairs'])} tx_used={int(noncoh_metrics['tx_with_pairs'])} "
                    f"phi_mae_deg={float(angle_metrics.get('phi_mae_deg', np.nan)):.3f} "
                    f"elev_mae_deg={float(angle_metrics.get('elev_mae_deg', np.nan)):.3f} "
                    f"u_rho_mean={float(noncoh_metrics['u_rho_mean']):.6f} "
                    f"u_err_mean={float(noncoh_metrics['u_err_mean']):.6e} "
                    f"g_rel_err_mean={float(noncoh_metrics['g_rel_err_mean']):.6e} "
                    f"u_rho_pw={float(noncoh_metrics.get('u_rho_power_weighted', np.nan)):.6f} "
                    f"coverage95={float(noncoh_metrics['power_coverage_rho95']):.3f} "
                    f"B_nrmse={float(noncoh_metrics['covariance_nrmse']):.3e} "
                    f"cov_refine={bool(ntn_music_out.get('covariance_refine', False))} "
                    f"fit={noncoh_metrics['fit_before']:.3e}->{noncoh_metrics['fit_after']:.3e}",
                    flush=True,
                )

            macro_stats.append(
                {
                    "sim_idx": int(sim_idx),
                    "ul_frequency_percent": percentage,
                    "ul_frequency_hz": sensing_frequency_hz,
                    "min_count": int(min_count),
                    "pair_counts_by_tx": pair_counts_by_tx.copy(),
                    "detected_ntn_count": int(detected_rx_union.size),
                    "interfered_ntn_count": interfered_ntn_count,
                    "satellite_azimuth_deg": float(sat_azimuth_deg),
                    "satellite_elevation_deg": float(sat_elevation_deg),
                    "satellite_look_pos": sat_look_pos.copy(),
                    "angle_metrics": angle_metrics,
                    "path_metrics_ul": path_metrics_ul, "path_metrics_dl": path_metrics_dl,
                    "manifold_alignment": manifold_alignment,
                    "detected_subset_metrics": music_quality["detected_subset_metrics"],
                    "detected_pairs_summary": music_quality["detected_pairs_summary"],
                    "noncoh_metrics": noncoh_metrics,
                    "source_count_metrics": source_count_metrics,
                }
            )

            if min_count <= 0:
                continue

            for round_idx in range(min_count):
                round_out = run_small_round(
                    h_tn_all,
                    h_ntn_all,
                    pairs_by_tx=pairing["pairs_by_tx"],
                    music_lookup=music_lookup,
                    sim_idx=int(sim_idx),
                    round_idx=int(round_idx),
                    lambda_ranges=lambda_list,
                    lambda_ranges_music_est=lambda_list_music_est,
                    lambda_ranges_music_real=lambda_list_music_real,
                    tx_power=float(tx_power),
                    snr_noise_power=float(snr_noise_power),
                    inr_noise_power=float(inr_noise_power),
                    max_detected_b_terms=resolved_b_term_limit,
                    print_music_u_corr=bool(print_music_u_corr),
                    oracle_path_channels=(dl_path_catalog["path_channels"] if multipath else None),
                    eps=eps,
                )

                raw_snr_all.extend(np.asarray(round_out["raw_snr_db"], dtype=np.float64).tolist())
                raw_sinr_all.extend(np.asarray(round_out["raw_sinr_db"], dtype=np.float64).tolist())
                raw_inr_all.extend(np.asarray(round_out["raw_inr_db"], dtype=np.float64).tolist())
                for lambda_ in lambda_list:
                    true_snr_all[lambda_].extend(
                        np.asarray(round_out["true_snr_db"][lambda_], dtype=np.float64).tolist()
                    )
                    true_sinr_all[lambda_].extend(
                        np.asarray(round_out["true_sinr_db"][lambda_], dtype=np.float64).tolist()
                    )
                    true_inr_all[lambda_].extend(
                        np.asarray(round_out["true_inr_db"][lambda_], dtype=np.float64).tolist()
                    )
                for lambda_ in lambda_list_music_est:
                    est_snr_all[lambda_].extend(
                        np.asarray(round_out["est_snr_db"][lambda_], dtype=np.float64).tolist()
                    )
                    est_sinr_all[lambda_].extend(
                        np.asarray(round_out["est_sinr_db"][lambda_], dtype=np.float64).tolist()
                    )
                    est_inr_all[lambda_].extend(
                        np.asarray(round_out["est_inr_db"][lambda_], dtype=np.float64).tolist()
                    )
                for lambda_ in lambda_list_music_real:
                    music_real_snr_all[lambda_].extend(
                        np.asarray(round_out["music_real_snr_db"][lambda_], dtype=np.float64).tolist()
                    )
                    music_real_sinr_all[lambda_].extend(
                        np.asarray(round_out["music_real_sinr_db"][lambda_], dtype=np.float64).tolist()
                    )
                    music_real_inr_all[lambda_].extend(
                        np.asarray(round_out["music_real_inr_db"][lambda_], dtype=np.float64).tolist()
                    )

    case_results = {}
    for percentage in percentages:
        raw_snr_all = case_accumulators[percentage]["raw_snr_all"]
        raw_sinr_all = case_accumulators[percentage]["raw_sinr_all"]
        raw_inr_all = case_accumulators[percentage]["raw_inr_all"]
        true_snr_all = case_accumulators[percentage]["true_snr_all"]
        true_sinr_all = case_accumulators[percentage]["true_sinr_all"]
        est_snr_all = case_accumulators[percentage]["est_snr_all"]
        est_sinr_all = case_accumulators[percentage]["est_sinr_all"]
        music_real_snr_all = case_accumulators[percentage]["music_real_snr_all"]
        music_real_sinr_all = case_accumulators[percentage]["music_real_sinr_all"]
        true_inr_all = case_accumulators[percentage]["true_inr_all"]
        est_inr_all = case_accumulators[percentage]["est_inr_all"]
        music_real_inr_all = case_accumulators[percentage]["music_real_inr_all"]
        macro_stats = case_accumulators[percentage]["macro_stats"]
        case_results[percentage] = {
            "ul_frequency_percent": percentage,
            "ul_frequency_hz": dl_frequency_hz * (1.0 + percentage / 100.0),
            "dl_frequency_hz": dl_frequency_hz,
            "ul_to_dl_mode": ul_to_dl_mode,
            "multipath": int(multipath),
            "oracle_model": "true_dl_paths" if multipath else "true_dl_effective_channels",
            "raw_snr_db": np.asarray(raw_snr_all, dtype=np.float64),
            "raw_sinr_db": np.asarray(raw_sinr_all, dtype=np.float64),
            "raw_inr_db": np.asarray(raw_inr_all, dtype=np.float64),
            "true_snr_db": {lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in true_snr_all.items()},
            "true_sinr_db": {
                lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in true_sinr_all.items()
            },
            "est_snr_db": {lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in est_snr_all.items()},
            "est_sinr_db": {
                lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in est_sinr_all.items()
            },
            "music_real_snr_db": {
                lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in music_real_snr_all.items()
            },
            "music_real_sinr_db": {
                lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in music_real_sinr_all.items()
            },
            "true_inr_db": {lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in true_inr_all.items()},
            "est_inr_db": {lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in est_inr_all.items()},
            "music_real_inr_db": {
                lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in music_real_inr_all.items()
            },
            "macro_stats": macro_stats,
            "bs_pos_ref": bs_pos_ref,
            "lambda_ranges": np.asarray(lambda_list, dtype=np.float64),
            "lambda_ranges_music_est": np.asarray(lambda_list_music_est, dtype=np.float64),
            "lambda_ranges_music_real": np.asarray(lambda_list_music_real, dtype=np.float64),
            "max_detected_b_terms": "all" if resolved_b_term_limit is None else int(resolved_b_term_limit),
        }

    if ul_frequency_percentages is None:
        return case_results[0.0]
    return {
        "multipath": int(multipath),
        "by_percentage": case_results,
        "ul_frequency_percentages": np.asarray(percentages),
        "dl_frequency_hz": dl_frequency_hz,
        "ul_to_dl_mode": ul_to_dl_mode,
        "lambda_ranges_music_est": np.asarray(lambda_list_music_est),
        "num_estimated_curves": len(percentages) * len(lambda_list_music_est),
    }


def save_experiment_metrics(
    experiment_out: Dict[str, Any],
    *,
    result_dir: str | Path = "result",
    output_name: str = "nulling_cdf_metrics.npz",
) -> Path:
    """Save experiment arrays and macro statistics for later reuse."""
    if "by_percentage" in experiment_out:
        result_path = Path(result_dir)
        result_path.mkdir(parents=True, exist_ok=True)
        combined = {
            "ul_frequency_percentages": np.asarray(experiment_out["ul_frequency_percentages"]),
            "dl_frequency_hz": np.asarray(experiment_out["dl_frequency_hz"]),
            "ul_to_dl_mode": np.asarray(experiment_out["ul_to_dl_mode"]),
            "num_estimated_curves": np.asarray(experiment_out["num_estimated_curves"]),
            "multipath": np.asarray(experiment_out.get("multipath", 0)),
        }
        for index, case in enumerate(experiment_out["by_percentage"].values()):
            case_file = save_experiment_metrics(
                case, result_dir=result_path / f"ul_{index:03d}", output_name=output_name,
            )
            with np.load(case_file, allow_pickle=False) as archive:
                combined.update({f"ul_{index:03d}/{key}": archive[key] for key in archive.files})
        save_path = result_path / output_name
        np.savez_compressed(save_path, **combined)
        return save_path
    result_path = Path(result_dir)
    result_path.mkdir(parents=True, exist_ok=True)
    save_path = result_path / output_name

    save_dict: Dict[str, Any] = {
        "raw_snr_db": np.asarray(experiment_out["raw_snr_db"], dtype=np.float64),
        "raw_sinr_db": np.asarray(experiment_out.get("raw_sinr_db", np.empty((0,), dtype=np.float64)), dtype=np.float64),
        "raw_inr_db": np.asarray(experiment_out["raw_inr_db"], dtype=np.float64),
        "lambda_ranges": np.asarray(experiment_out["lambda_ranges"], dtype=np.float64),
        "lambda_ranges_music_est": np.asarray(
            experiment_out.get("lambda_ranges_music_est", np.empty((0,), dtype=np.float64)),
            dtype=np.float64,
        ),
        "lambda_ranges_music_real": np.asarray(
            experiment_out.get("lambda_ranges_music_real", np.empty((0,), dtype=np.float64)),
            dtype=np.float64,
        ),
        "max_detected_b_terms": np.asarray([str(experiment_out.get("max_detected_b_terms", "all"))]),
    }
    for key in ("ul_frequency_percent", "ul_frequency_hz", "dl_frequency_hz", "ul_to_dl_mode", "multipath", "oracle_model"):
        if key in experiment_out:
            save_dict[key] = np.asarray(experiment_out[key])
    # Indexed copies preserve distinct arbitrary lambdas that round to the same
    # legacy scientific-notation key; lambda_ranges_* defines their order.
    for prefix in ("true", "est", "music_real"):
        for metric in ("snr", "sinr", "inr"):
            for index, values in enumerate(experiment_out.get(f"{prefix}_{metric}_db", {}).values()):
                save_dict[f"{prefix}_{metric}_db_index_{index:03d}"] = np.asarray(values)
    if experiment_out.get("bs_pos_ref") is not None:
        save_dict["bs_pos_ref"] = np.asarray(experiment_out["bs_pos_ref"], dtype=np.float64)
    for lambda_, vals in experiment_out["true_snr_db"].items():
        save_dict[f"true_snr_db_{lambda_:.0e}"] = np.asarray(vals, dtype=np.float64)
    for lambda_, vals in experiment_out.get("true_sinr_db", {}).items():
        save_dict[f"true_sinr_db_{lambda_:.0e}"] = np.asarray(vals, dtype=np.float64)
    for lambda_, vals in experiment_out["est_snr_db"].items():
        save_dict[f"est_snr_db_{lambda_:.0e}"] = np.asarray(vals, dtype=np.float64)
    for lambda_, vals in experiment_out.get("est_sinr_db", {}).items():
        save_dict[f"est_sinr_db_{lambda_:.0e}"] = np.asarray(vals, dtype=np.float64)
    for lambda_, vals in experiment_out["true_inr_db"].items():
        save_dict[f"true_inr_db_{lambda_:.0e}"] = np.asarray(vals, dtype=np.float64)
    for lambda_, vals in experiment_out["est_inr_db"].items():
        save_dict[f"est_inr_db_{lambda_:.0e}"] = np.asarray(vals, dtype=np.float64)
    for lambda_, vals in experiment_out.get("music_real_snr_db", {}).items():
        save_dict[f"music_real_snr_db_{lambda_:.0e}"] = np.asarray(vals, dtype=np.float64)
    for lambda_, vals in experiment_out.get("music_real_sinr_db", {}).items():
        save_dict[f"music_real_sinr_db_{lambda_:.0e}"] = np.asarray(vals, dtype=np.float64)
    for lambda_, vals in experiment_out.get("music_real_inr_db", {}).items():
        save_dict[f"music_real_inr_db_{lambda_:.0e}"] = np.asarray(vals, dtype=np.float64)

    macro_stats = experiment_out.get("macro_stats", [])
    for band in ("ul", "dl"):
        save_dict[f"path_metrics_{band}_json"] = np.asarray([
            json.dumps(row.get(f"path_metrics_{band}", {})) for row in macro_stats], dtype=str)
    save_dict["macro_stats_sim_idx"] = np.asarray([row["sim_idx"] for row in macro_stats], dtype=int)
    save_dict["macro_stats_min_count"] = np.asarray([row["min_count"] for row in macro_stats], dtype=int)
    save_dict["macro_stats_detected_ntn_count"] = np.asarray(
        [row["detected_ntn_count"] for row in macro_stats],
        dtype=int,
    )
    save_dict["macro_stats_interfered_ntn_count"] = np.asarray(
        [row["interfered_ntn_count"] for row in macro_stats],
        dtype=int,
    )
    if macro_stats:
        save_dict["macro_stats_satellite_azimuth_deg"] = np.asarray(
            [row.get("satellite_azimuth_deg", np.nan) for row in macro_stats],
            dtype=np.float64,
        )
        save_dict["macro_stats_satellite_elevation_deg"] = np.asarray(
            [row.get("satellite_elevation_deg", np.nan) for row in macro_stats],
            dtype=np.float64,
        )
        save_dict["macro_stats_satellite_look_pos"] = np.stack(
            [
                np.asarray(row.get("satellite_look_pos", [np.nan, np.nan, np.nan]), dtype=np.float64)
                for row in macro_stats
            ],
            axis=0,
        )
        save_dict["macro_stats_pair_counts_by_tx"] = np.stack(
            [np.asarray(row["pair_counts_by_tx"], dtype=int) for row in macro_stats],
            axis=0,
        )
        save_dict["macro_stats_phi_mae_deg"] = np.asarray(
            [row.get("angle_metrics", {}).get("phi_mae_deg", np.nan) for row in macro_stats],
            dtype=np.float64,
        )
        save_dict["macro_stats_elev_mae_deg"] = np.asarray(
            [row.get("angle_metrics", {}).get("elev_mae_deg", np.nan) for row in macro_stats],
            dtype=np.float64,
        )
        save_dict["macro_stats_angle_match_count"] = np.asarray(
            [row.get("angle_metrics", {}).get("matched_pairs", 0) for row in macro_stats],
            dtype=int,
        )
        save_dict["macro_stats_manifold_alignment_count"] = np.asarray(
            [row.get("manifold_alignment", {}).get("count", 0) for row in macro_stats],
            dtype=int,
        )
        save_dict["macro_stats_manifold_rho_mean"] = np.asarray(
            [row.get("manifold_alignment", {}).get("rho_mean", np.nan) for row in macro_stats],
            dtype=np.float64,
        )
        save_dict["macro_stats_manifold_rho_median"] = np.asarray(
            [row.get("manifold_alignment", {}).get("rho_median", np.nan) for row in macro_stats],
            dtype=np.float64,
        )
        save_dict["macro_stats_manifold_rho_min"] = np.asarray(
            [row.get("manifold_alignment", {}).get("rho_min", np.nan) for row in macro_stats],
            dtype=np.float64,
        )
        save_dict["macro_stats_detected_subset_count"] = np.asarray(
            [row.get("detected_subset_metrics", {}).get("count", 0) for row in macro_stats],
            dtype=np.float64,
        )
        save_dict["macro_stats_detected_subset_nrmse"] = np.asarray(
            [row.get("detected_subset_metrics", {}).get("nrmse", np.nan) for row in macro_stats],
            dtype=np.float64,
        )
        save_dict["macro_stats_detected_subset_cos_sim"] = np.asarray(
            [row.get("detected_subset_metrics", {}).get("cos_sim", np.nan) for row in macro_stats],
            dtype=np.float64,
        )
        save_dict["macro_stats_detected_subset_mag_mae"] = np.asarray(
            [row.get("detected_subset_metrics", {}).get("mag_mae", np.nan) for row in macro_stats],
            dtype=np.float64,
        )
        save_dict["macro_stats_detected_subset_power_ratio_db"] = np.asarray(
            [row.get("detected_subset_metrics", {}).get("power_ratio_db", np.nan) for row in macro_stats],
            dtype=np.float64,
        )
        save_dict["macro_stats_detected_pairs_count"] = np.asarray(
            [row.get("detected_pairs_summary", {}).get("pairs", 0) for row in macro_stats],
            dtype=int,
        )
        save_dict["macro_stats_detected_pairs_nrmse_mean"] = np.asarray(
            [row.get("detected_pairs_summary", {}).get("nrmse_mean", np.nan) for row in macro_stats],
            dtype=np.float64,
        )
        save_dict["macro_stats_detected_pairs_nrmse_median"] = np.asarray(
            [row.get("detected_pairs_summary", {}).get("nrmse_median", np.nan) for row in macro_stats],
            dtype=np.float64,
        )
        save_dict["macro_stats_detected_pairs_cos_mean"] = np.asarray(
            [row.get("detected_pairs_summary", {}).get("cos_mean", np.nan) for row in macro_stats],
            dtype=np.float64,
        )
        save_dict["macro_stats_detected_pairs_cos_median"] = np.asarray(
            [row.get("detected_pairs_summary", {}).get("cos_median", np.nan) for row in macro_stats],
            dtype=np.float64,
        )
        save_dict["macro_stats_noncoh_pairs"] = np.asarray(
            [row.get("noncoh_metrics", {}).get("pairs", 0) for row in macro_stats],
            dtype=int,
        )
        save_dict["macro_stats_noncoh_tx_with_pairs"] = np.asarray(
            [row.get("noncoh_metrics", {}).get("tx_with_pairs", 0) for row in macro_stats],
            dtype=int,
        )
        save_dict["macro_stats_noncoh_u_rho_mean"] = np.asarray(
            [row.get("noncoh_metrics", {}).get("u_rho_mean", np.nan) for row in macro_stats],
            dtype=np.float64,
        )
        save_dict["macro_stats_noncoh_u_err_mean"] = np.asarray(
            [row.get("noncoh_metrics", {}).get("u_err_mean", np.nan) for row in macro_stats],
            dtype=np.float64,
        )
        save_dict["macro_stats_noncoh_g_rel_err_mean"] = np.asarray(
            [row.get("noncoh_metrics", {}).get("g_rel_err_mean", np.nan) for row in macro_stats],
            dtype=np.float64,
        )
        save_dict["macro_stats_source_count_method"] = np.asarray(
            [row.get("source_count_metrics", {}).get("method", "unknown") for row in macro_stats],
            dtype=str,
        )
        for key in ("u_rho_median", "u_rho_p10", "u_rho_power_weighted", "g_rel_err_median",
                    "power_coverage_rho95", "covariance_nrmse", "fit_before", "fit_after", "accepted_peak_mean"):
            save_dict[f"macro_stats_noncoh_{key}"] = np.asarray(
                [row.get("noncoh_metrics", {}).get(key, np.nan) for row in macro_stats], dtype=float,
            )
        for label in ("used", "mdl", "rank", "eigengap"):
            save_dict[f"macro_stats_source_count_{label}_mean"] = np.asarray(
                [
                    row.get("source_count_metrics", {}).get(f"{label}_mean", np.nan)
                    for row in macro_stats
                ],
                dtype=np.float64,
            )
            save_dict[f"macro_stats_source_count_{label}_sum"] = np.asarray(
                [
                    row.get("source_count_metrics", {}).get(f"{label}_sum", 0)
                    for row in macro_stats
                ],
                dtype=int,
            )

    np.savez(save_path, **save_dict)
    return save_path
