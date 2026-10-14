"""Blind MUSIC with selectable MDL/rank source counts; no NTN channel truth."""
import numpy as np
import ntn_music_detection as nmd
import multipath_support as mps


def detect_directions(covariances, *, positions_dl, orientations, f_ul, f_dl,
                      snapshots, noise_normalized, config, power_correction=True):
    """Same smoothing/peak/correlated-power primitives as the multipath notebook.

    Input covariance uses conjugated-channel convention; fitted powers and output
    steering vectors use the raw channel convention consumed by BeamformingCalc.
    """
    config.validate()
    margin = (config.rank_margin_for_snapshots(snapshots) if config.source_estimation == "rank"
              else config.rank_noise_margin)
    covariances = np.asarray(covariances, dtype=np.complex128)
    positions_ul = np.asarray(positions_dl) * f_ul / f_dl
    ntx, m, other_m = covariances.shape
    if m != other_m or positions_ul.shape != (m, 3) or np.asarray(orientations).shape != (ntx, 3):
        raise ValueError("Covariances and array geometries have incompatible dimensions")
    if config.spatial_smoothing:
        indices, sub_positions = mps.planar_subarrays(positions_ul, config.subarray_rows, config.subarray_cols)
        rows, cols = config.subarray_rows, config.subarray_cols
    else:
        indices, sub_positions = np.arange(m)[None, :], positions_ul
        # Exact geometry is used; rows*cols only specifies the array size.
        rows, cols = 1, m
    directions, diagnostics = [], []
    for b, covariance in enumerate(covariances):
        smooth = mps.spatially_smooth_covariance(
            covariance, indices, forward_backward=config.forward_backward and config.spatial_smoothing)
        detection = nmd.detect_ntn_music_from_hi(
            np.zeros((1, 1, len(sub_positions)), dtype=complex), covariance_override=smooth,
            noise_var=noise_normalized, covariance_mode="sample", num_snapshots=snapshots,
            source_estimation=config.source_estimation,
            rank_relative_threshold=config.rank_relative_threshold,
            rank_noise_margin=margin, compute_user_scores=False)
        _, en = nmd.signal_noise_subspaces_from_music_out(detection)
        k = int(detection["num_sources_est"])
        peaks = nmd.music_top_peaks(
            en, num_rows=rows, num_cols=cols,
            phi_grid_deg=np.arange(0, 360, config.grid_step_deg),
            theta_grid_deg=np.arange(0, 180 + 1e-9, config.grid_step_deg),
            orientation_rad=tuple(orientations[b]), array_positions_local=sub_positions,
            array_phase_sign=-1, rotation_order="zyx", top_n=k,
            forward_only=config.forward_only, forward_cos_min=0.0,
            min_sep_phi_deg=0.0, min_sep_theta_deg=0.0, local_maxima_only=True,
            max_peak_correlation=config.max_peak_correlation,
            max_noise_projection=config.max_noise_projection,
            refine_peaks=config.peak_refine,
            refine_half_width_deg=config.peak_refine_half_width_deg,
            refine_maxiter=config.peak_refine_maxiter)
        ul_vectors = np.asarray([
            nmd.array_position_steering_global(phi, theta, positions_ul,
                                               orientation_rad=tuple(orientations[b])).ravel()
            for _, phi, theta, _ in peaks], dtype=complex).reshape(-1, m)
        noise_hat = nmd.estimate_noise_power_from_music_out(detection)
        _, gains, fit, _ = mps.fit_correlated_path_covariance(covariance.conj(), ul_vectors, noise_hat)
        chosen = mps.select_estimated_paths(gains, top_k=config.top_k, energy_fraction=config.energy_fraction)
        dl_vectors = np.asarray([
            nmd.array_position_steering_global(peaks[i][1], peaks[i][2], positions_dl,
                                               orientation_rad=tuple(orientations[b])).ravel()
            for i in chosen], dtype=complex).reshape(-1, m)
        scale = (f_ul / f_dl) ** 2 if power_correction else 1.0
        directions.append(dict(u=dl_vectors, g=np.asarray(gains)[chosen] * scale,
                               phi=np.asarray([peaks[i][1] for i in chosen]),
                               theta=np.asarray([peaks[i][2] for i in chosen])))
        diagnostics.append(dict(num_sources=k, accepted_peaks=len(chosen),
                                empty=int(len(chosen) == 0), noise_estimate=float(noise_hat),
                                covariance_fit=float(fit), rank_noise_margin_effective=float(margin)))
    return directions, diagnostics
