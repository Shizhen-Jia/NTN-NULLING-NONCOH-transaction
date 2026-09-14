"""Coherent narrowband multipath, planar smoothing, and truth-only path diagnostics.

No path truth enters the detector. Directions and powers come from the received
covariance; all physical paths remain in the DL channel used for evaluation.
"""
from __future__ import annotations

from typing import Any
import numpy as np
from scipy.optimize import linear_sum_assignment


def _numpy(value):
    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)


def _path_field(value, shape):
    """Broadcast synthetic-array [RX,TX,L] metadata to [RX,RA,TX,TA,L]."""
    array = np.asarray(_numpy(value))
    if array.ndim == 3:
        array = array[:, None, :, None, :]
    try:
        return np.broadcast_to(array, shape)
    except ValueError as exc:
        raise ValueError(f"Path metadata shape {array.shape} cannot broadcast to {shape}.") from exc


def collapse_cir_to_narrowband(
    cir: np.ndarray, *, time_index: int | None = 0,
    tau: np.ndarray | None = None, frequency_offset_hz: float = 0.0,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Coherently sum paths, never sum time samples or path powers.

    CIR axes: [RX, RX_ANT, TX, TX_ANT, PATH, TIME]. Four-dimensional already
    collapsed channels and five-dimensional single-time CIRs are also accepted.
    time_index=0 returns the legacy four-dimensional channel for static CIRs.
    time_index=None preserves TIME as the last output axis.

    cir must contain BASEBAND coefficients from paths.cir(), not paths.a.
    Carrier phase is already present. An optional offset from that carrier adds
    exp(-j*2*pi*offset*tau), as for one OFDM subcarrier; it requires path delays.
    Geometry/element responses are held at the traced carrier (no beam squint).
    Invalid/padded paths may be masked; no strongest-path truncation is applied.
    """
    h = np.asarray(cir, dtype=np.complex128)
    if h.ndim not in (4, 5, 6):
        raise ValueError("CIR must have 4, 5 or 6 axes: RX, RX_ANT, TX, TX_ANT[, PATH[, TIME]].")
    if not np.isfinite(frequency_offset_hz):
        raise ValueError("frequency_offset_hz must be finite.")
    if h.ndim == 4:
        if tau is not None or valid_mask is not None or frequency_offset_hz != 0:
            raise ValueError("A collapsed channel has no path metadata; supply a 5D/6D CIR.")
        if time_index not in (0, None):
            raise IndexError("An already collapsed channel has only one time sample.")
        return h if time_index == 0 else h[..., None]
    if h.ndim == 5:
        h = h[..., None]
    if time_index is not None:
        if isinstance(time_index, bool) or not isinstance(time_index, (int, np.integer)):
            raise ValueError("time_index must be a nonnegative integer or None.")
        if not 0 <= time_index < h.shape[-1]:
            raise IndexError("time_index is outside the CIR time axis.")
        h = h[..., time_index:time_index+1]
    path_shape = h.shape[:-1]
    if valid_mask is not None:
        mask = _path_field(valid_mask, path_shape).astype(bool)
        h = np.where(mask[..., None], h, 0.0)
    if frequency_offset_hz != 0:
        if tau is None:
            raise ValueError("Nonzero frequency offset requires tau in seconds.")
        delays = _path_field(tau, path_shape)
        valid_delay = np.isfinite(delays) & (delays >= 0)
        # Sionna marks padded delays with -1; remove those terms explicitly.
        phase = np.exp(-2j*np.pi*float(frequency_offset_hz)*np.where(valid_delay, delays, 0))
        h = np.where(valid_delay[..., None], h*phase[..., None], 0.0)
    if not np.all(np.isfinite(h)):
        raise ValueError("Valid CIR coefficients must be finite.")
    summed = np.sum(h, axis=4)
    return summed if time_index is None else summed[..., 0]


def validate_max_depth(max_depth):
    """Validate Sionna's maximum interaction depth without changing its value."""
    if (isinstance(max_depth, (bool, np.bool_))
            or not isinstance(max_depth, (int, float, np.integer, np.floating))
            or not np.isfinite(max_depth) or max_depth < 0 or int(max_depth) != max_depth):
        raise ValueError("max_depth must be a nonnegative integer.")
    return int(max_depth)


def resolve_propagation_options(max_depth, ntn_los_mode="natural", options=None):
    """Pass the configured depth to TN DL, NTN DL and reciprocal NTN UL."""
    max_depth = validate_max_depth(max_depth)
    if ntn_los_mode not in ("natural", "nlos_only"):
        raise ValueError("ntn_los_mode must be 'natural' or 'nlos_only'.")
    base = dict(los=True, specular_reflection=True, diffuse_reflection=False,
                refraction=True, synthetic_array=True, max_depth=max_depth)
    allowed = {"specular_reflection", "diffuse_reflection", "refraction", "diffraction",
               "edge_diffraction", "diffraction_lit_region", "samples_per_src",
               "max_num_paths_per_src", "seed"}
    options = {} if options is None else dict(options)
    if set(options) - allowed:
        raise ValueError(f"Unsupported propagation options: {sorted(set(options)-allowed)}")
    for name in ("samples_per_src", "max_num_paths_per_src"):
        if name in options and (int(options[name]) != options[name] or options[name] <= 0):
            raise ValueError(f"{name} must be a positive integer.")
    base.update(options)
    ntn = dict(base)
    ntn["los"] = ntn_los_mode == "natural"
    return base, ntn


def planar_subarrays(positions, rows, cols):
    """Find translated rectangular subarrays from actual local YZ coordinates.

    Element ordering is read from coordinates, not guessed from flatten_order.
    Returns element indices per subarray and centered reference coordinates.
    """
    p = np.asarray(positions, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.all(np.isfinite(p)):
        raise ValueError("Array positions must be a finite [M,3] matrix.")
    if not np.allclose(p[:, 0], p[0, 0], atol=1e-8, rtol=0):
        raise ValueError("Planar smoothing requires an array in the local YZ plane.")
    ys, zs = np.unique(p[:, 1]), np.unique(p[:, 2])
    for axis in (ys, zs):
        if len(axis) > 2 and not np.allclose(np.diff(axis), np.diff(axis)[0], rtol=2e-5, atol=1e-8):
            raise ValueError("Spatial smoothing requires uniformly spaced elements.")
    if len(ys)*len(zs) != len(p):
        raise ValueError("Spatial smoothing requires a complete rectangular array.")
    if int(rows) != rows or int(cols) != cols or not 1 <= rows <= len(zs) or not 1 <= cols <= len(ys):
        raise ValueError("Smoothing subarray rows/cols must fit inside the physical array.")
    rows, cols = int(rows), int(cols)
    if rows*cols < 2:
        raise ValueError("MUSIC subarrays require at least two elements.")
    grid = np.full((len(zs), len(ys)), -1, dtype=int)
    for i, (_, y, z) in enumerate(p):
        iz, iy = np.searchsorted(zs, z), np.searchsorted(ys, y)
        if grid[iz, iy] != -1:
            raise ValueError("Spatial smoothing does not support duplicated array positions.")
        grid[iz, iy] = i
    indices = np.array([grid[z:z+rows, y:y+cols].ravel()
                        for z in range(len(zs)-rows+1) for y in range(len(ys)-cols+1)])
    reference = p[indices[0]]
    return indices, reference - reference.mean(axis=0)


def spatially_smooth_covariance(covariance, indices, *, forward_backward=True):
    r = np.asarray(covariance, dtype=np.complex128)
    blocks = [r[np.ix_(idx, idx)] for idx in indices]
    smooth = np.mean(blocks, axis=0)
    if forward_backward:
        # Rectangular row-major coordinates: reversing both axes is centrosymmetry.
        smooth = (smooth + smooth[::-1, ::-1].conj())/2
    return (smooth + smooth.conj().T)/2


def fit_correlated_path_covariance(covariance, steering_vectors, noise_power=0.0):
    """Fit R = A Q A^H + noise I with a full positive-semidefinite Q.

    Normalized full-array steering vectors are rows of steering_vectors. The
    pseudoinverse fits Q, then a PSD projection handles finite-sample noise.
    No per-path independence assumption or path truth is used. Poorly conditioned
    directions are diagnosed; a small numerical singular-value cutoff is used.
    """
    r = np.asarray(covariance, dtype=complex)
    u = np.asarray(steering_vectors, dtype=complex)
    signal = (r+r.conj().T)/2 - float(noise_power)*np.eye(len(r))
    if u.size == 0:
        return np.empty((0, 0), complex), np.empty(0), 1.0 if np.linalg.norm(signal) else 0.0, float("nan")
    a = u.T
    inverse = np.linalg.pinv(a, rcond=1e-8)
    q = inverse @ signal @ inverse.conj().T
    eigenvalues, eigenvectors = np.linalg.eigh((q+q.conj().T)/2)
    q = (eigenvectors*np.maximum(eigenvalues, 0)) @ eigenvectors.conj().T
    fit = a @ q @ a.conj().T
    scale = np.linalg.norm(signal)
    residual = float(np.linalg.norm(signal-fit)/scale) if scale else 0.0
    return q, np.maximum(np.diag(q).real, 0), residual, float(np.linalg.cond(a))


def select_estimated_paths(powers, *, top_k=None, energy_fraction=1.0):
    """Select anonymous estimated paths per BS, never truncate the true channel."""
    if not np.isfinite(energy_fraction) or not 0 < energy_fraction <= 1:
        raise ValueError("energy_fraction must be in (0,1].")
    if top_k is not None and (int(top_k) != top_k or top_k <= 0):
        raise ValueError("top_k must be a positive integer or None.")
    powers = np.asarray(powers, dtype=float)
    order = np.argsort(-powers, kind="stable")
    order = order[powers[order] > 0]
    if order.size and energy_fraction < 1:
        count = np.searchsorted(np.cumsum(powers[order]), energy_fraction*powers[order].sum()) + 1
        order = order[:count]
    if top_k is not None:
        order = order[:int(top_k)]
    return order


def run_multipath_music_pipeline(
    h_all, *, music_kwargs, subarray_rows=6, subarray_cols=6,
    spatial_smoothing=True, forward_backward=True, top_k=None, energy_fraction=1.0,
):
    """Estimate multiple coherent path directions from UL covariance only.

    Spatial smoothing decorrelates paths using translated subarrays; full-array
    correlated covariance fitting estimates powers. The full array is retained
    for UL->DL transfer. In sample mode there is one random waveform per UE/RX
    antenna, shared by its paths. Smoothing creates no independent ray signals.
    Analytic source-count MDL remains a diagnostic, not an iid-sample guarantee.
    """
    import ntn_music_detection as nmd

    h = np.asarray(h_all, dtype=complex)
    kw = dict(music_kwargs)
    if h.ndim != 4 or kw.get("pair_keys") is not None:
        raise ValueError("Multipath MUSIC requires [RX,RA,TX,M] channels and blind pair_keys=None.")
    if kw.get("covariance_refine", False):
        raise ValueError("Set covariance_refine=False for multipath; full correlated power fitting replaces it.")
    if kw.get("scan_mode", "complex") != "complex":
        raise ValueError("Multipath MUSIC requires scan_mode='complex'.")
    nrx, nra, ntx, m = h.shape
    p = np.asarray(kw.get("array_positions_local"), dtype=float)
    if p.shape != (m, 3):
        raise ValueError("Multipath MUSIC requires exact full-array positions [M,3].")
    select_estimated_paths([], top_k=top_k, energy_fraction=energy_fraction)
    mode = kw.get("channel_mode", "conj")
    if mode not in ("raw", "conj"):
        raise ValueError("channel_mode must be 'raw' or 'conj'.")
    if spatial_smoothing:
        indices, p_sub = planar_subarrays(p, subarray_rows, subarray_cols)
    else:
        indices, p_sub = np.arange(m)[None, :], p
        subarray_rows, subarray_cols = int(kw["tx_rows"]), int(kw["tx_cols"])
    orientation = nmd._resolve_tx_orientations(
        num_tx=ntx, nsect=int(kw.get("nsect", 3)),
        use_sector_orientation=bool(kw.get("use_sector_orientation", True)),
        sector_yaw_offset_rad=float(kw.get("sector_yaw_offset_rad", 0)),
        sector_pitch_rad=float(kw.get("sector_pitch_rad", 0)),
        sector_roll_rad=float(kw.get("sector_roll_rad", 0)),
        tx_orientations_rad=kw.get("tx_orientations_rad"),
    )
    covariance_mode = kw.get("detect_covariance_mode", "analytic")
    noise = float(kw.get("detect_noise_var", 0))
    if covariance_mode not in ("analytic", "sample") or noise < 0 or not np.isfinite(noise):
        raise ValueError("Use analytic/sample covariance with finite nonnegative noise.")
    snapshots = int(kw.get("detect_num_snapshots", 800))
    if covariance_mode == "sample" and snapshots < 2:
        raise ValueError("Sample covariance requires at least two snapshots.")
    powers = nmd._broadcast_powers(kw.get("detect_user_powers"), nrx, nra)
    if not np.all(np.isfinite(powers)) or np.any(powers < 0):
        raise ValueError("User powers must be finite and nonnegative.")
    _, plane, sign = nmd._parse_manifold_label(kw.get("manifold_label", "yz:+1"))
    records, candidate_records, q_records = [], [], []
    counts, source_counts, mdl_counts, rank_counts, gap_counts = [], [], [], [], []
    fits, conditions, noise_estimates, raw_ranks = [], [], [], []
    covariances, smoothed_covariances = [], []
    for tx in range(ntx):
        hi = h[:, :, tx, :] if mode == "raw" else h[:, :, tx, :].conj()
        if covariance_mode == "analytic":
            covariance = nmd._covariance_from_static_channels(hi, powers, noise)
        else:
            covariance, _ = nmd._sample_covariance_from_snapshots(
                hi=hi, user_powers_2d=powers, noise_var=noise, num_snapshots=snapshots,
                rng=np.random.default_rng(kw.get("detect_rng_seed")),
            )
        raw_covariance = covariance if mode == "raw" else covariance.conj()
        smooth = spatially_smooth_covariance(
            covariance, indices, forward_backward=bool(forward_backward and spatial_smoothing))
        detection = nmd.detect_ntn_music_from_hi(
            hi[..., indices[0]], covariance_override=smooth,
            num_sources=kw.get("detect_num_sources"), noise_var=noise,
            covariance_mode=covariance_mode, num_snapshots=snapshots,
            source_estimation=kw.get("detect_source_estimation", "rank"),
            energy_ratio=kw.get("detect_energy_ratio", .98),
            rank_relative_threshold=kw.get("detect_rank_relative_threshold", 1e-5),
            rank_noise_margin=kw.get("detect_rank_noise_margin", 1e-3), compute_user_scores=False,
        )
        k = int(detection["num_sources_est"])
        _, en = nmd.signal_noise_subspaces_from_music_out(detection)
        peaks = nmd.music_top_peaks(
            en, num_rows=int(subarray_rows), num_cols=int(subarray_cols),
            phi_grid_deg=kw.get("phi_grid_deg", np.arange(0, 360, 2.)),
            theta_grid_deg=kw.get("theta_grid_deg", np.arange(0, 181, 2.)),
            orientation_rad=tuple(orientation[tx]), panel_plane=plane, phase_sign=sign,
            horizontal_sign=int(kw.get("steering_horizontal_sign", -1)),
            flatten_order=kw.get("flatten_order", "F"),
            forward_only=bool(kw.get("sector_forward_only", True)),
            forward_cos_min=float(kw.get("sector_forward_cos_min", 0)),
            rotation_order=kw.get("rotation_order", "zyx"), array_positions_local=p_sub,
            array_phase_sign=1 if mode == "raw" else -1, top_n=k,
            local_maxima_only=kw.get("peak_local_maxima", True),
            max_noise_projection=kw.get("peak_max_noise_projection", .2),
            min_sep_phi_deg=kw.get("peak_min_sep_phi_deg", 0),
            min_sep_theta_deg=kw.get("peak_min_sep_theta_deg", 0),
            max_peak_correlation=kw.get("peak_max_correlation", .98),
            refine_peaks=kw.get("peak_refine", True),
            refine_half_width_deg=kw.get("peak_refine_half_width_deg", 1),
            refine_maxiter=kw.get("peak_refine_maxiter", 40),
        )
        full_vectors = np.asarray([
            nmd.array_position_steering_global(phi, theta, p, orientation_rad=tuple(orientation[tx]),
                                               rotation_order=kw.get("rotation_order", "zyx")).ravel()
            for _, phi, theta, _ in peaks], dtype=complex).reshape(-1, m)
        noise_hat = noise if covariance_mode == "analytic" else nmd.estimate_noise_power_from_music_out(detection)
        q, gains, fit, condition = fit_correlated_path_covariance(raw_covariance, full_vectors, noise_hat)
        chosen = select_estimated_paths(gains, top_k=top_k, energy_fraction=energy_fraction)
        for i, (score, phi, theta, _) in enumerate(peaks):
            phi_store = phi
            if kw.get("phi_mirror_about_sector", False):
                phi_store = (2*np.rad2deg(orientation[tx, 0])-phi_store) % 360
            offset = float(np.round(float(kw.get("phi_offset_deg", 0)) % 360, 1))
            row = (tx, (phi_store+offset) % 360, theta, gains[i], score, full_vectors[i], phi)
            candidate_records.append(row)
        for i in chosen:
            records.append(candidate_records[len(candidate_records)-len(peaks)+i])
        q_records.append(q)
        counts.append(len(chosen)); source_counts.append(k)
        mdl_counts.append(int(detection["num_sources_mdl"]))
        rank_counts.append(int(detection["num_sources_rank"]))
        gap_counts.append(int(detection["num_sources_eigengap"]))
        fits.append(fit); conditions.append(condition); noise_estimates.append(noise_hat)
        raw_ranks.append(nmd._estimate_num_sources_rank(
            np.linalg.eigvalsh(covariance)[::-1], noise_power=noise,
            relative_threshold=kw.get("detect_rank_relative_threshold", 1e-5),
            noise_margin=kw.get("detect_rank_noise_margin", 1e-3)))
        covariances.append(raw_covariance)
        smoothed_covariances.append(smooth if mode == "raw" else smooth.conj())
    max_candidates = max((len(q) for q in q_records), default=0)
    q_padded = np.zeros((ntx, max_candidates, max_candidates), dtype=complex)
    for tx, q in enumerate(q_records):
        q_padded[tx, :len(q), :len(q)] = q
    def column(rows, index, dtype=float):
        return np.asarray([row[index] for row in rows], dtype=dtype)
    output = {
        "peak_t_idx": column(records, 0, int), "peak_phi_hat_deg": column(records, 1),
        "peak_theta_hat_deg": column(records, 2), "peak_g_hat": column(records, 3),
        "peak_phi_physical_deg": column(records, 6),
        "peak_score": column(records, 4), "peak_selection_score": column(records, 3),
        "peak_u_hat_raw": column(records, 5, complex).reshape(-1, m),
        "candidate_t_idx": column(candidate_records, 0, int),
        "candidate_phi_hat_deg": column(candidate_records, 1),
        "candidate_theta_hat_deg": column(candidate_records, 2),
        "candidate_g_hat": column(candidate_records, 3),
        "candidate_counts": np.array([len(q) for q in q_records]),
        "correlated_source_covariance": q_padded,
        "covariance_fit_before": np.asarray(fits), "covariance_fit_after": np.asarray(fits),
        "correlated_fit_condition": np.asarray(conditions), "noise_power_estimate": np.asarray(noise_estimates),
        "ul_covariance": np.asarray(covariances), "smoothed_ul_covariance": np.asarray(smoothed_covariances),
        "raw_signal_rank": np.asarray(raw_ranks),
        "num_sources_record": np.asarray(source_counts), "num_sources_mdl_record": np.asarray(mdl_counts),
        "num_sources_rank_record": np.asarray(rank_counts), "num_sources_eigengap_record": np.asarray(gap_counts),
        "accepted_peak_counts": np.asarray(counts), "source_count_method": kw.get("detect_source_estimation", "rank"),
        "steering_geometry_source": "sionna_array_positions",
        "detection_name": ("Spatially smoothed" if spatial_smoothing else "Unsmoothed") + " MUSIC + correlated power fit",
        "covariance_refine": False, "spatial_smoothing": bool(spatial_smoothing),
        "smoothing_subarray_shape": np.array([subarray_rows, subarray_cols]),
        "smoothing_num_subarrays": len(indices), "peak_refine": kw.get("peak_refine", True),
        "peak_min_sep_phi_deg": kw.get("peak_min_sep_phi_deg", 0),
        "peak_min_sep_theta_deg": kw.get("peak_min_sep_theta_deg", 0),
        "peak_max_correlation": kw.get("peak_max_correlation", .98),
        "detected_rx_indices_unique": np.empty(0, int), "detected_rx_indices_by_tx": {},
        # Deliberately empty paired fields: anonymous paths have no inferred UE IDs.
        "pair_rx_idx": np.empty(0, int), "pair_t_idx": np.empty(0, int),
        "pair_rx_ant_idx": np.empty(0, int), "pair_u_hat_raw": np.empty((0, m), complex),
        "pair_alpha_hat_raw": np.empty(0, complex),
    }
    return output


def build_path_catalog(paths, cir, tau=None, *, time_index=0):
    """Extract ALL valid paths for storage/evaluation, never for peak estimation."""
    a = np.asarray(cir, dtype=complex)
    if a.ndim == 6:
        a = a[..., time_index]
    if a.ndim != 5:
        raise ValueError("Path catalog requires a 5D/6D CIR.")
    valid = _path_field(paths.valid, a.shape).astype(bool)
    a = np.where(valid, a, 0)
    power = np.sum(np.abs(a)**2, axis=(1, 3))
    link_valid = np.any(valid, axis=(1, 3)) & (power > 0)
    def scalar(field):
        return _path_field(field, a.shape)[:, 0, :, 0, :]
    interactions = np.asarray(_numpy(paths.interactions))
    if interactions.ndim == 6:
        interactions = interactions[:, :, 0, :, 0, :]
    if interactions.ndim != 4 or interactions.shape[1:] != link_valid.shape:
        raise ValueError("Unexpected path interaction axes; expected DEPTH,RX,TX,PATH.")
    is_los = link_valid & ~np.any(interactions != 0, axis=0)
    nrx, nra, ntx, m, num_paths = a.shape
    path_channels = a.transpose(0, 4, 1, 2, 3).reshape(nrx*num_paths, nra, ntx, m)
    catalog = dict(
        valid=link_valid, is_los=is_los, path_power=power, interactions=interactions,
        tau=scalar(paths.tau if tau is None else tau),
        phi_t=scalar(paths.phi_t), theta_t=scalar(paths.theta_t),
        path_channels=path_channels,
    )
    return catalog


def path_truth_map(catalog, *, nsect=3, sionna_phi_is_global=True):
    """Legacy-shaped diagnostic map with one virtual row per true ray, not UE."""
    import ntn_music_detection as nmd
    num_paths = catalog["valid"].shape[-1]
    mapping = {}
    for rx, tx, path in np.argwhere(catalog["valid"]):
        phi = np.rad2deg(catalog["phi_t"][rx, tx, path]) % 360
        if not sionna_phi_is_global:
            phi = float(nmd.sector_local_aod_to_global(phi, sector_index=int(tx % nsect), nsect=nsect))
        mapping[(int(rx*num_paths+path), int(tx))] = (
            phi, np.rad2deg(catalog["theta_t"][rx, tx, path]), int(tx//nsect), int(tx % nsect))
    return mapping


def summarize_path_estimates(catalog, estimated, *, max_angle_error_deg=5.0, nsect=3,
                             sionna_phi_is_global=True):
    """One-to-one angular matching across all rays; LOS/NLOS are per BS-UE link."""
    valid, los = catalog["valid"], catalog["is_los"]
    has_path, has_los = np.any(valid, axis=-1), np.any(los, axis=-1)
    matched, phi_errors, theta_errors, angular_errors = [], [], [], []
    truth_map = path_truth_map(catalog, nsect=nsect, sionna_phi_is_global=sionna_phi_is_global)
    def unit(phi, theta):
        ph, th = np.deg2rad(phi), np.deg2rad(theta)
        return np.column_stack((np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)))
    powers_matched = 0.0
    detected = set()
    for tx in range(valid.shape[1]):
        keys = [key for key in truth_map if key[1] == tx]
        est = np.flatnonzero(np.asarray(estimated["peak_t_idx"]) == tx)
        if not keys or not len(est):
            continue
        phi = np.array([truth_map[key][0] for key in keys])
        theta = np.array([truth_map[key][1] for key in keys])
        ph_hat = np.asarray(estimated.get("peak_phi_physical_deg", estimated["peak_phi_hat_deg"]))[est]
        th_hat = np.asarray(estimated["peak_theta_hat_deg"])[est]
        distance = np.rad2deg(np.arccos(np.clip(unit(phi, theta) @ unit(ph_hat, th_hat).T, -1, 1)))
        rr, cc = linear_sum_assignment(np.where(distance <= max_angle_error_deg, distance, 1e6))
        for r, c in zip(rr, cc):
            if distance[r, c] > max_angle_error_deg:
                continue
            rx, ray = divmod(keys[r][0], valid.shape[-1])
            detected.add(rx)
            matched.append((rx, tx, ray, int(est[c])))
            powers_matched += catalog["path_power"][rx, tx, ray]
            phi_errors.append(abs((ph_hat[c]-phi[r]+180) % 360-180))
            theta_errors.append(abs(th_hat[c]-theta[r]))
            angular_errors.append(distance[r, c])
    total_power = catalog["path_power"][valid].sum()
    return {
        "valid_paths": int(valid.sum()), "los_paths": int(los.sum()),
        "nlos_paths": int((valid & ~los).sum()),
        "los_links": int(has_los.sum()), "nlos_links": int((has_path & ~has_los).sum()),
        "no_path_links": int((~has_path).sum()), "matched_paths": len(matched),
        "matched_pairs": len(matched), "estimated_paths": len(estimated["peak_t_idx"]),
        "angular_mae_deg": float(np.mean(angular_errors)) if angular_errors else float("nan"),
        "phi_mae_deg": float(np.mean(phi_errors)) if phi_errors else float("nan"),
        "elev_mae_deg": float(np.mean(theta_errors)) if theta_errors else float("nan"),
        "matched_path_power_fraction": float(powers_matched/total_power) if total_power else 0.,
        "detected_rx_indices": sorted(detected),
    }
