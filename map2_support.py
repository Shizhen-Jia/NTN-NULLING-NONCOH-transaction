"""Single-scene rank-MUSIC maps using the SectorDrop experiment's public pipeline.

Plotting helpers do not select victims or change beams. Terminal CDFs use actual
VSAT channels. Sionna radio maps show spatially averaged power with an
isotropic receiver, as in map.ipynb. Radio-map code lives in map2.ipynb so it
can be edited interactively. Truth matching only labels plot markers.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import numpy as np


BS_COLORS = ("#b2182b", "#e08214", "#762a83", "#2166ac")


def save_json(path, value):
    def clean(item):
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple, np.ndarray)):
            return [clean(v) for v in item]
        if isinstance(item, np.generic):
            return clean(item.item())
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def aggregate_inr(channels, beams, tx_power, noise_power):
    """Match ncu._interference_power_per_rx: sum_b,r |v_b^H h_irb|^2 P/N."""
    h, v = np.asarray(channels), np.asarray(beams)
    if h.ndim != 4 or v.shape != h.shape[2:]:
        raise ValueError("Expected channels [RX,RA,TX,M] and beams [TX,M].")
    received = np.einsum("nrtm,tm->nrt", h, v.conj(), optimize=True)
    return np.sum(np.abs(received)**2, axis=(1, 2)) * float(tx_power) / float(noise_power)


def run_snapshot(config, *, positions_kwargs, paths_kwargs, music_kwargs,
                 multipath_kwargs, position_seed, satellite_seed,
                 satellite_azimuth_range, satellite_elevation_range,
                 ul_percentage, ul_to_dl_mode, ul_dl_power_correction,
                 lambda_value, h_tn_th, tx_power, snr_noise_power,
                 inr_noise_power, output_dir):
    """One macro, one UL carrier and one scheduling round; no favorable-drop search."""
    import multipath_support as mps
    import nulling_cdf_utils as ncu

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    sat_rng = np.random.default_rng(satellite_seed)
    azimuth = float(sat_rng.uniform(*satellite_azimuth_range))
    elevation = float(sat_rng.uniform(*satellite_elevation_range))
    placement = dict(positions_kwargs, azimuth=azimuth, elevation=elevation)
    if position_seed is None:
        config.compute_positions(**placement)
    else:
        random_state = np.random.get_state()
        try:
            # Same seed derivation as macro 0 of run_nulling_cdf_experiment.
            np.random.seed(np.random.SeedSequence([int(position_seed), 0]).generate_state(1)[0])
            config.compute_positions(**placement)
        finally:
            np.random.set_state(random_state)
    # Native Sionna requires an integer; share one fresh seed across TN/DL/UL paths.
    paths_kwargs = dict(paths_kwargs)
    propagation = paths_kwargs.get("propagation_options")
    if propagation is not None and propagation.get("seed", 42) is None:
        paths_kwargs["propagation_options"] = dict(
            propagation, seed=int(np.random.SeedSequence().generate_state(1)[0]))
    print("Tracing one SectorDrop scene (including its normal TN acceptance retries)...", flush=True)
    config.compute_paths(**paths_kwargs)
    h_tn = mps.collapse_cir_to_narrowband(config.a_tn)
    h_dl = mps.collapse_cir_to_narrowband(config.a_ntn)
    array_dl = ncu._scene_tx_array_positions_local(config)
    orientations = np.asarray(config.tx_orientation_rad, dtype=float)
    pairing = ncu.pair_tn_to_strongest_tx(
        h_tn, h_tn_th=h_tn_th, tx_antennas=h_dl.shape[-1],
        tx_power=tx_power, snr_noise_power=snr_noise_power)
    if int(pairing["min_count"]) != 1 or not np.all(pairing["pair_counts_by_tx"] == 1):
        raise RuntimeError("SectorDrop must retain exactly one TN per sector.")
    for tx, pairs in pairing["pairs_by_tx"].items():
        if len(pairs) != 1 or int(pairs[0]["tn_idx"]) != int(tx):
            raise RuntimeError("The retained TN order does not match the target sectors.")

    frequency_dl = float(paths_kwargs["fc"])
    frequency_ul = frequency_dl * (1 + float(ul_percentage) / 100)
    if ul_percentage == 0:
        sensing_paths, sensing_cir, sensing_tau = config.paths_ntn, config.a_ntn, config.tau_ntn
        h_ul = h_dl
    else:
        sensing_paths, sensing_cir, sensing_tau = config.compute_ntn_ul_paths(
            frequency_ul, max_depth=config.ntn_solver_options["max_depth"])
        h_ul = mps.collapse_cir_to_narrowband(sensing_cir)
    kw = dict(music_kwargs, array_positions_local=array_dl * (frequency_ul / frequency_dl),
              tx_orientations_rad=orientations, covariance_refine=False)
    print("Estimating anonymous NTN path directions with rank-MUSIC...", flush=True)
    detected = mps.run_multipath_music_pipeline(h_ul, music_kwargs=kw, **multipath_kwargs)
    # Exactly the main notebook's path matching: used only to label detected UEs.
    catalog_ul = mps.build_path_catalog(sensing_paths, sensing_cir, sensing_tau)
    path_metrics = mps.summarize_path_estimates(catalog_ul, detected, nsect=kw["nsect"],
                                               sionna_phi_is_global=True)
    detected_idx = np.asarray(path_metrics["detected_rx_indices"], dtype=int)
    detected["detected_rx_indices_unique"] = detected_idx
    dl_detected = (ncu.transfer_music_peaks_to_dl(
        detected, dl_array_positions=array_dl, music_kwargs=kw, num_tx=h_dl.shape[2])
        if ul_percentage != 0 and ul_to_dl_mode == "angle" else detected)
    power_scale = (frequency_ul / frequency_dl)**2 if ul_dl_power_correction else 1.
    if power_scale != 1:
        dl_detected = dict(dl_detected)
        dl_detected["peak_g_hat"] = np.asarray(detected["peak_g_hat"]) * power_scale
    lookup = ncu.build_music_tx_lookup(dl_detected, num_ntn_rx=len(h_dl),
                                      num_tx_total=h_dl.shape[2], num_tx_ant=h_dl.shape[3])
    round_out = ncu.run_small_round(
        h_tn, h_dl, pairs_by_tx=pairing["pairs_by_tx"], music_lookup=lookup, round_idx=0,
        lambda_ranges=None, lambda_ranges_music_est=[lambda_value], lambda_ranges_music_real=[],
        tx_power=tx_power, snr_noise_power=snr_noise_power, inr_noise_power=inr_noise_power,
        max_detected_b_terms="all")
    beams_before = np.stack([round_out["raw_beams"][t].ravel() for t in range(h_dl.shape[2])])
    beams_after = np.stack([round_out["est_beams"][lambda_value][t].ravel()
                            for t in range(h_dl.shape[2])])
    np.testing.assert_allclose(np.linalg.norm(beams_before, axis=1), 1, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(beams_after, axis=1), 1, atol=1e-7)
    before = aggregate_inr(h_dl, beams_before, tx_power, inr_noise_power)
    after = aggregate_inr(h_dl, beams_after, tx_power, inr_noise_power)
    before_db = 10*np.log10(np.maximum(before, 1e-30))
    after_db = 10*np.log10(np.maximum(after, 1e-30))
    valid = np.any(np.abs(h_dl) > 1e-12, axis=(1, 2, 3))
    np.testing.assert_allclose(before_db[valid], round_out["raw_inr_db"], atol=1e-7)
    np.testing.assert_allclose(after_db[valid], round_out["est_inr_db"][lambda_value], atol=1e-7)
    summary = dict(
        ntn_deployed=len(h_dl), ntn_detected=len(detected_idx), tn_scheduled=len(h_tn),
        lambda_value=lambda_value, ul_frequency_hz=frequency_ul, dl_frequency_hz=frequency_dl,
        satellite_angles_deg=[azimuth, elevation], source_count_method="rank",
        path_solver_seed=(paths_kwargs.get("propagation_options") or {}).get("seed", 42),
        num_sources_by_sector=detected["num_sources_record"],
        accepted_peaks_by_sector=detected["accepted_peak_counts"], path_matching=path_metrics,
        detected_median_inr_before_db=(float(np.median(before_db[detected_idx])) if len(detected_idx) else None),
        detected_median_inr_after_db=(float(np.median(after_db[detected_idx])) if len(detected_idx) else None),
        detected_median_suppression_db=(float(np.median((before_db-after_db)[detected_idx])) if len(detected_idx) else None),
        tn_median_sinr_before_db=float(np.median(round_out["raw_sinr_db"])),
        tn_median_sinr_after_db=float(np.median(round_out["est_sinr_db"][lambda_value])),
        tn_median_sinr_change_db=float(np.median(round_out["est_sinr_db"][lambda_value]-round_out["raw_sinr_db"])),
        detection_label="Main-pipeline truth-to-peak angular matching, for visualization only",
        beam_source="Estimated anonymous rank-MUSIC peaks; no oracle channels in nulling")
    save_json(directory / "snapshot_summary.json", summary)
    np.savez_compressed(directory / "snapshot.npz", h_tn=h_tn, h_ntn=h_dl,
        detected_ntn_idx=detected_idx, bs_pos=config.tx_pos, tn_pos=config.tn_pos,
        ntn_pos=config.rx_ntn_pos, rectangles=config.tn_sampler.rectangles,
        beams_before=beams_before, beams_after=beams_after, inr_before=before, inr_after=after,
        satellite_look_pos=config.ntn_look_pos,
        peak_t_idx=detected["peak_t_idx"], peak_u=dl_detected["peak_u_hat_raw"],
        peak_g=dl_detected["peak_g_hat"])
    print(json.dumps({k: v for k, v in summary.items() if k.startswith(("ntn_", "tn_", "detected_"))}, indent=2), flush=True)
    return dict(detected_idx=detected_idx, beams_before=beams_before, beams_after=beams_after,
                inr_before=before, inr_after=after, round_out=round_out, summary=summary)


def sector_boundary_segments(bs_positions, rectangles, yaw_offset_rad=0.):
    """Three true wedge boundaries per BS, clipped to its own drop rectangle."""
    segments = []
    for b, (position, rect) in enumerate(zip(bs_positions, rectangles)):
        x, y = np.asarray(position, float)[:2]
        xmin, xmax, ymin, ymax = map(float, rect)
        if not (xmin < x < xmax and ymin < y < ymax):
            raise ValueError("The BS must lie inside its drop rectangle.")
        # Boresights are yaw + {0,120,240}; boundaries are halfway between.
        for angle in yaw_offset_rad + np.deg2rad([60., 180., 300.]):
            dx, dy = np.cos(angle), np.sin(angle)
            distances = []
            if abs(dx) > 1e-12:
                distances.append(((xmax if dx > 0 else xmin)-x)/dx)
            if abs(dy) > 1e-12:
                distances.append(((ymax if dy > 0 else ymin)-y)/dy)
            distance = min(t for t in distances if t > 0)
            segments.append((b, np.array([[x, y], [x+distance*dx, y+distance*dy]])))
    return segments


def _map_overlays(ax, config, detected_idx, *, dark_background=False):
    extent = np.asarray(config.extent, float) / 1000
    rectangles = np.asarray(config.tn_sampler.rectangles, float)
    partition_color = "white" if dark_background else "0.25"
    # Include the four BS-cell boundaries, so sector rays never imply a
    # partition that extends into a neighbouring BS's drop rectangle.
    edges = set()
    for xmin, xmax, ymin, ymax in rectangles:
        for a, b in [((xmin,ymin),(xmax,ymin)), ((xmax,ymin),(xmax,ymax)),
                     ((xmax,ymax),(xmin,ymax)), ((xmin,ymax),(xmin,ymin))]:
            edges.add(tuple(sorted((a, b))))
    for edge in sorted(edges):
        xy = np.asarray(edge) / 1000
        ax.plot(xy[:,0], xy[:,1], color=partition_color, lw=.65, alpha=.75, zorder=4)
    for b, segment in sector_boundary_segments(config.tx_pos, rectangles, config.tn_drop_yaw_rad):
        line, = ax.plot(segment[:,0]/1000, segment[:,1]/1000,
                        color=BS_COLORS[b], lw=1., linestyle=(0,(5,3)), zorder=5)
        line.set_path_effects([pe.Stroke(linewidth=1.8, foreground="white", alpha=.8), pe.Normal()])
    for b, position in enumerate(config.tx_pos):
        ax.scatter(position[0]/1000, position[1]/1000, s=74, marker="^",
                   color=BS_COLORS[b], edgecolors="white", linewidths=.7, zorder=8)
        ax.annotate(f"BS {b+1}", position[:2]/1000, xytext=(5,5), textcoords="offset points",
                    fontsize=7, color=BS_COLORS[b], zorder=9,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=.6))
    # All 12 TNs are scheduled; only detected NTN terminals are displayed.
    for i, position in enumerate(config.tn_pos):
        color = BS_COLORS[int(config.tn_bs_index[i])]
        ax.scatter(position[0]/1000, position[1]/1000, s=37, marker="o",
                   facecolors="#83d18d", edgecolors=color, linewidths=1.2, zorder=7)
    selected = np.asarray(config.rx_ntn_pos)[np.asarray(detected_idx, dtype=int)]
    if len(selected):
        points = ax.scatter(selected[:,0]/1000, selected[:,1]/1000, s=26, marker="x",
                            color="#002b80", linewidths=1.2, zorder=10)
        points.set_path_effects([pe.Stroke(linewidth=2.4, foreground="white"), pe.Normal()])
    ax.set(xlim=extent[:2], ylim=extent[2:], xlabel="x (km)", ylabel="y (km)")
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)


def _legend_handles():
    return [Line2D([],[],marker="^",color="none",markerfacecolor="0.25",label="TN-BS"),
            Line2D([],[],marker="o",color="none",markerfacecolor="#83d18d",markeredgecolor="0.25",label="Scheduled TN-UE"),
            Line2D([],[],marker="x",color="#002b80",linestyle="none",label="Detected NTN-UE"),
            Line2D([],[],color="0.35",linestyle="--",label="Sector boundary")]


def plot_layout(config, snapshot, output_dir, *, background_path=None):
    fig, ax = plt.subplots(figsize=(5.4,5.5), layout="constrained")
    extent = np.asarray(config.extent)/1000
    if background_path is not None and Path(background_path).is_file():
        ax.imshow(plt.imread(background_path), extent=extent, origin="upper", alpha=.85)
    else:
        ax.imshow(config.cm.bldg_grid, extent=extent, origin="lower",
                  cmap=ListedColormap(["#f4f3ef", "#c8c5be"]), interpolation="nearest")
    _map_overlays(ax, config, snapshot["detected_idx"])
    ax.set_title("Sector-constrained deployment", fontsize=10)
    ax.legend(handles=_legend_handles(), loc="upper center", bbox_to_anchor=(.5,-.12),
              ncol=2, fontsize=7, frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(Path(output_dir)/f"map2_layout.{ext}", dpi=350, bbox_inches="tight")
    return fig


def plot_ntn_inr_cdf(snapshot, output_dir):
    """Compare identical UE sets using the actual coherent VSAT channel powers."""
    before = np.asarray(snapshot["inr_before"], dtype=float)
    after = np.asarray(snapshot["inr_after"], dtype=float)
    if before.ndim != 1 or before.shape != after.shape:
        raise ValueError("Before/after INR must contain the same ordered UE set.")
    if not np.all(np.isfinite([before, after])) or np.any(np.array([before, after]) < 0):
        raise ValueError("INR samples must be finite and nonnegative.")
    detected = np.asarray(snapshot["detected_idx"], dtype=int)
    positive = np.r_[before[before > 0], after[after > 0]]
    floor_db = float(5*np.floor(10*np.log10(positive.min())/5)-5) if len(positive) else -100.
    before_db = 10*np.log10(np.maximum(before, 10**(floor_db/10)))
    after_db = 10*np.log10(np.maximum(after, 10**(floor_db/10)))
    groups = [("detected", detected), ("all", np.arange(len(before)))]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True,
                             layout="constrained")
    stats = {}
    for ax, (name, indices), title in zip(axes, groups, ("Detected NTN-UEs", "All deployed NTN-UEs")):
        ax.set_title(f"{title} (N = {len(indices)})", fontsize=12)
        if not len(indices):
            ax.text(.5, .5, "No detected terminals", transform=ax.transAxes, ha="center")
            continue
        for values, label, color, style in (
                (before_db, "Before nulling", "#d95f02", "--"),
                (after_db, "After rank-MUSIC nulling", "#0072b2", "-")):
            x = np.sort(values[indices])
            # Retain every UE, including zero-power samples, in both ECDFs.
            ax.step(np.r_[x[0], x], np.r_[0., np.arange(1, len(x)+1)/len(x)],
                    where="post", label=label, color=color, linestyle=style, lw=2)
        nonzero = indices[(before[indices] > 0) & (after[indices] > 0)]
        suppression = before_db[nonzero]-after_db[nonzero]
        median = float(np.median(suppression)) if len(suppression) else None
        stats[name] = dict(count=len(indices),
            median_inr_before_db=float(np.median(before_db[indices])),
            median_inr_after_db=float(np.median(after_db[indices])),
            median_paired_suppression_db=median,
            finite_paired_suppression_count=len(nonzero),
            improved_count=int(np.count_nonzero(after[indices] < before[indices])),
            worsened_count=int(np.count_nonzero(after[indices] > before[indices])),
            unchanged_count=int(np.count_nonzero(after[indices] == before[indices])),
            zero_before_count=int(np.count_nonzero(before[indices] == 0)),
            zero_after_count=int(np.count_nonzero(after[indices] == 0)))
        if median is not None:
            ax.text(.97, .06, f"Median paired reduction: {median:.1f} dB",
                    ha="right", transform=ax.transAxes, fontsize=10,
                    bbox=dict(facecolor="white", edgecolor="0.85", alpha=.9))
        if np.any(before[indices] == 0) or np.any(after[indices] == 0):
            ax.text(.03, .02, "Zero interference shown at the left boundary",
                    transform=ax.transAxes, fontsize=8)
        ax.legend(loc="upper left", fontsize=9)
    for ax in axes:
        ax.set(xlabel="Aggregate NTN INR (dB)", ylim=(0, 1.02))
        ax.grid(True, alpha=.25)
        ax.tick_params(labelsize=10)
    axes[0].set_ylabel("Empirical CDF", fontsize=11)
    directory = Path(output_dir)
    for ext in ("png", "pdf"):
        fig.savefig(directory/f"map2_ntn_inr_cdf.{ext}", dpi=300, bbox_inches="tight")
    flags = np.zeros(len(before), dtype=int)
    flags[detected] = 1
    # Store exact linear powers too; the dB floor is only a display convention.
    np.savetxt(directory/"map2_ntn_inr_cdf.csv",
               np.column_stack([np.arange(len(before)), flags, before, after, before_db, after_db]),
               delimiter=",", header="ue_index,detected,inr_before_linear,inr_after_linear,inr_before_db_display,inr_after_db_display",
               comments="", fmt=["%d", "%d", "%.12e", "%.12e", "%.8f", "%.8f"])
    save_json(directory/"map2_ntn_inr_cdf_summary.json", dict(display_floor_db=floor_db, **stats))
    print(json.dumps(stats, indent=2), flush=True)
    return fig
