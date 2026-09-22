"""Fresh Sionna SectorDrop channels for *static* Appendix-D validation.

Every macro redraws TN/NTN UEs and satellite direction and retraces DL and FDD
UL channels. This supplies the existing cache adapter; it does not manufacture
motion, repeated-listening histories, or deadline observations for E7/E8.
Sionna is imported lazily so existing cache/finite-model runs need no RT install.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False,
                               default=lambda x: np.asarray(x).tolist()) + "\n")


def _save_npz(path: Path, **values: Any) -> None:
    with path.open("xb") as stream:
        np.savez_compressed(stream, **values)


def fresh_rt_config(*, quick: bool = True, seed: int = 20260921,
                    num_macros: int | None = None, max_depth: int | None = None) -> dict:
    """Resolve and validate physical settings without importing Sionna.

    Six drops permit disjoint 2/2/2 development splits, not risk certification.
    Even the 20-drop full profile requires a larger sample for statistical claims.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    count = (6 if quick else 20) if num_macros is None else num_macros
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 6:
        raise ValueError("Fresh RT needs at least 6 whole drops for disjoint train/calibration/test")
    depth = 3 if max_depth is None else max_depth
    if isinstance(depth, bool) or not isinstance(depth, (int, np.integer)) or depth < 0:
        raise ValueError("max_depth must be a nonnegative integer")
    # Preserve the original notebook's 7 GHz DL, array, geometry, deployment,
    # TN acceptance rules, and physical powers. Quick reduces ray/angle budgets.
    bandwidth, dl, tx_power = 200e6, 7e9, 10 ** ((35 - 30) / 10)
    ul_power = 10 ** ((23 - 30) / 10)
    noise = lambda nf: 10 ** ((-174 + 10 * np.log10(bandwidth) + nf - 30) / 10)
    ntrain, ncal = max(2, int(.4 * count)), max(2, int(.3 * count))
    return dict(
        source="fresh_sionna_rt", profile="quick" if quick else "full", seed=int(seed),
        scene_path=str(_ROOT / "Denver_scene/10kmwithfigure/10km.xml"),
        scene_variant="sector_drop_fresh_appendix_d", grid_size_m=10,
        f_dl=dl, ul_frequency_percentages=[-10.0], ul_frequencies_hz=[.9 * dl],
        ul_to_dl_mode="angle", ul_dl_power_correction=False,
        bs_element_spacing_m=299792458 / (2 * dl),
        B_ul=bandwidth, bandwidth=bandwidth, enforce_disjoint_bands=True,
        tx_power=tx_power, ul_tx_power=ul_power,
        snr_noise_power=noise(7), inr_noise_power=noise(3), bs_noise_power=noise(2),
        num_macro_sims=int(count), max_depth=int(depth),
        channel_model="coherent narrowband at carrier; all traced rays summed",
        channel_time_index=0, channel_frequency_offset_hz=0,
        temporal_data_available=False,
        statistical_scope="independent UE/satellite drops conditional on one fixed city geometry",
        development_only=bool(quick),
        certificate="empirical whole-drop calibration only; no finite-sample risk certificate",
        expected_split_counts=dict(train=ntrain, calibration=ncal, test=count-ntrain-ncal),
        seed_derivation="SeedSequence([root_seed, macro_id]).spawn(4): position, satellite, propagation, MUSIC",
        positions=dict(ntn_rx=20, tn_rx=12, centerBS=False, bs_grid=[2, 2],
                       bs_boundary=2500, tn_outdoor_probability=.5,
                       tn_association_margin_db=3.0, tn_max_attempts=256,
                       tn_candidates_per_batch=16, tn_sector_yaw_offset_rad=0.,
                       ntn_building_ratio=.8, plot_grid=False, plot_bs=False,
                       plot_tn=False, plot_ntn=False),
        # At depth zero indoor TN sampling is deliberately preserved: a drop
        # can fail instead of silently changing its indoor/outdoor population.
        paths=dict(nsect=3, fc=dl, tx_rows=8, tx_cols=8, tn_rx_rows=1, tn_rx_cols=1,
                   max_depth=int(depth), ntn_los_mode="natural", bandwidth=bandwidth,
                   tx_power_dbm=35, sector_yaw_offset_rad=0.,
                   sector_pitch_rad=float(np.deg2rad(6)), sector_roll_rad=0.,
                   propagation_options=dict(specular_reflection=True, refraction=True,
                       diffuse_reflection=False, diffraction=False,
                       samples_per_src=10_000 if quick else 100_000,
                       max_num_paths_per_src=10_000 if quick else 100_000)),
        satellite_azimuth_range=[0., 360.], satellite_elevation_range=[25., 90.],
        music=dict(tx_rows=8, tx_cols=8, nsect=3, pair_keys=None,
                   detect_num_sources=None, detect_user_powers=None,
                   detect_noise_var=noise(2)/ul_power, detect_covariance_mode="sample",
                   detect_num_snapshots=800, detect_source_estimation="mdl",
                   detect_energy_ratio=.98, detect_rank_relative_threshold=1e-5,
                   detect_rank_noise_margin=1e-3, channel_mode="conj",
                   manifold_label="yz:+1", flatten_order="F", scan_mode="complex",
                   steering_horizontal_sign=-1, use_sector_orientation=True,
                   sector_yaw_offset_rad=0., sector_pitch_rad=float(np.deg2rad(6)),
                   sector_roll_rad=0., rotation_order="zyx", sector_forward_only=True,
                   sector_forward_cos_min=0., peak_max_correlation=.98,
                   peak_refine=True, peak_refine_half_width_deg=1.,
                   peak_refine_maxiter=40, peak_local_maxima=True,
                   peak_max_noise_projection=.2, covariance_refine=False,
                   phi_grid_deg=np.arange(0., 360., 2. if quick else .5).tolist(),
                   theta_grid_deg=np.arange(0., 180.01, 2. if quick else .5).tolist()),
        multipath_music=dict(subarray_rows=6, subarray_cols=6, spatial_smoothing=True,
                             forward_backward=True, top_k=None, energy_fraction=1.),
        sensing_model="800 finite complex-Gaussian UE snapshots per sector; independent sensing windows across sectors; noise normalized by 23 dBm UL power",
    )


def _macro_seeds(seed: int, macro_id: int) -> dict[str, int]:
    children = np.random.SeedSequence([seed, macro_id]).spawn(4)
    return {name: int(child.generate_state(1)[0]) for name, child in zip(
        ("position", "satellite", "propagation", "music"), children)}


def _load_backend():
    try:
        from sionna.rt import load_scene
        from SceneConfigSionnaSectorDrop import SceneConfigSionna
        import multipath_support as mps
        import nulling_cdf_utils as ncu
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Fresh RT requires the existing Sionna RT environment. Run with the "
            "sionna20 notebook kernel or its Python interpreter; cached channels "
            "are never substituted for a requested fresh run."
        ) from exc
    return load_scene, SceneConfigSionna, mps, ncu


def generate_fresh_rt_cache(output: str | Path, *, quick: bool = True,
                            seed: int = 20260921, num_macros: int | None = None,
                            max_depth: int | None = None) -> dict:
    """Create a new timestamped cache below ``output`` and return its metadata.

    Call ``run_cached_spatial`` with result["cache_dir"] and an output directory
    outside that cache. A failed run remains explicitly failed and incomplete;
    existing archived channels are never overwritten or reused as fresh data.
    """
    config = fresh_rt_config(quick=quick, seed=seed, num_macros=num_macros,
                             max_depth=max_depth)
    scene_file = Path(config["scene_path"])
    import sector_drop_reporting as reporting

    root = Path(output).expanduser().resolve()
    directory = root / ("fresh_rt_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    directory.mkdir(parents=True, exist_ok=False)
    reporting.update_status(directory, "running", completed_macros=0)
    try:
        channels = directory / "channels"
        observations = channels / "ul_000"
        observations.mkdir(parents=True)
        config["positions"]["tn_drop_output_dir"] = str(directory / "tn_drop")
        config["positions"]["tn_min_channel_norm"] = float(np.sqrt(
            10 ** (-6 / 10) * config["bs_noise_power"] * 64 / config["tx_power"]))
        seeds = [dict(macro_id=i, **_macro_seeds(int(seed), i))
                 for i in range(config["num_macro_sims"])]
        config["macro_seed_file"] = "macro_seeds.json"
        reporting.save_run_metadata(directory, config)
        _write_json(directory / "macro_seeds.json", seeds)
        source = Path(__file__)
        (directory / "sources" / source.name).write_bytes(source.read_bytes())
        _write_json(directory / "fresh_rt_source.json", dict(
            path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        reporting.update_status(directory, "running", completed_macros=0)
        macro_rows = []
        with reporting.capture_run_log(directory):
            print(f"Fresh Sionna RT cache: {directory}", flush=True)
            print("Static independent drops; this is not a physical dynamic E7/E8 trajectory.", flush=True)
            if not scene_file.is_file():
                raise FileNotFoundError(f"Sionna scene not found: {scene_file}")
            load_scene, scene_type, mps, ncu = _load_backend()
            scene = scene_type(load_scene(str(scene_file)))
            scene.build_coverage_map(grid_size=config["grid_size_m"], show_xy=False, plot=False)
            bs_reference = None
            for row in seeds:
                macro = row["macro_id"]
                print(f"Fresh RT macro {macro + 1}/{len(seeds)}: redraw TN/NTN and trace DL/UL", flush=True)
                satellite_rng = np.random.default_rng(row["satellite"])
                azimuth = float(satellite_rng.uniform(*config["satellite_azimuth_range"]))
                elevation = float(satellite_rng.uniform(*config["satellite_elevation_range"]))
                positions = dict(config["positions"], azimuth=azimuth, elevation=elevation)
                paths = dict(config["paths"])
                paths["propagation_options"] = dict(paths["propagation_options"], seed=row["propagation"])
                # The legacy sector sampler draws from NumPy's global RNG. Preserve
                # the caller's state, including all TN channel-validation retries.
                random_state = np.random.get_state()
                try:
                    np.random.seed(row["position"])
                    scene.compute_positions(**positions)
                    scene.compute_paths(**paths)
                finally:
                    np.random.set_state(random_state)
                tx_pos = np.asarray(scene.tx_pos)
                if bs_reference is None:
                    bs_reference = tx_pos.copy()
                elif not np.allclose(tx_pos, bs_reference):
                    raise RuntimeError("BS positions changed across fresh drops")
                h_tn = mps.collapse_cir_to_narrowband(scene.a_tn)
                h_ntn = mps.collapse_cir_to_narrowband(scene.a_ntn)
                dl_positions = ncu._scene_tx_array_positions_local(scene)
                orientations = np.asarray(scene.tx_orientation_rad)
                if dl_positions is None:
                    raise RuntimeError("Cannot transfer FDD peaks without the actual array geometry")
                _save_npz(channels / f"channels_{macro:04d}.npz", h_tn=h_tn, h_ntn=h_ntn,
                          tx_pos=tx_pos, tn_pos=np.asarray(scene.tn_pos),
                          ntn_pos=np.asarray(scene.rx_ntn_pos),
                          satellite_look_pos=np.asarray(scene.ntn_look_pos),
                          satellite_angles_deg=np.array([azimuth, elevation]),
                          array_positions_local=dl_positions, tx_orientations_rad=orientations,
                          noise_var=config["music"]["detect_noise_var"])
                _save_npz(channels / f"paths_dl_{macro:04d}.npz", a_tn=scene.a_tn,
                          tau_tn=scene.tau_tn, a_ntn=scene.a_ntn, tau_ntn=scene.tau_ntn)
                frequency = config["ul_frequencies_hz"][0]
                _, ul_cir, ul_tau = scene.compute_ntn_ul_paths(frequency, max_depth=config["max_depth"])
                h_ul = mps.collapse_cir_to_narrowband(ul_cir)
                ul_positions = dl_positions * (frequency / config["f_dl"])
                _save_npz(observations / f"sensing_{macro:04d}.npz", h_ntn_ul=h_ul,
                          ul_frequency_hz=frequency, dl_frequency_hz=config["f_dl"],
                          ul_frequency_percent=-10., array_positions_local_ul=ul_positions)
                _save_npz(observations / f"paths_ul_{macro:04d}.npz", a=ul_cir, tau_raw=ul_tau)
                music_kw = dict(config["music"], array_positions_local=ul_positions,
                                tx_orientations_rad=orientations,
                                # Passing a Generator advances independent sector
                                # windows; passing an int would reset inside each TX.
                                detect_rng_seed=np.random.default_rng(row["music"]))
                detected = mps.run_multipath_music_pipeline(
                    h_ul, music_kwargs=music_kw, **config["multipath_music"])
                transferred = ncu.transfer_music_peaks_to_dl(
                    detected, dl_array_positions=dl_positions,
                    music_kwargs=music_kw, num_tx=h_ul.shape[2])
                # No DL channel, NTN UE IDs, or per-path truth is supplied to MUSIC
                # or angular transfer. DL truth is kept separately for offline tests.
                numeric = {k: np.asarray(v) for k, v in detected.items()
                           if np.asarray(v).dtype.kind != "O"}
                _save_npz(observations / f"music_{macro:04d}.npz",
                          peak_u_used_for_dl=transferred["peak_u_hat_raw"],
                          peak_g_used_for_dl=transferred["peak_g_hat"],
                          ul_to_dl_mode=np.asarray("angle"), **numeric)
                macro_rows.append(dict(macro_id=macro, peaks=int(len(detected["peak_t_idx"])),
                                       zero_ntn_dl_links=int(np.sum(np.linalg.norm(h_ntn, axis=(1, 3)) == 0)),
                                       satellite_azimuth_deg=azimuth, satellite_elevation_deg=elevation))
                _write_json(directory / "macro_summary.json", macro_rows)
                reporting.update_status(directory, "running", completed_macros=macro + 1)
        from .cache_adapter import inspect_cache
        info = inspect_cache(directory)
        if (info["num_macros"] != config["num_macro_sims"] or not info["ul_cases"]
                or len(info["ul_cases"][0]["macro_ids"]) != config["num_macro_sims"]):
            raise RuntimeError("Fresh RT cache is incomplete")
        summary = dict(cache_dir=str(directory), profile=config["profile"],
                       num_macros=info["num_macros"], max_depth=config["max_depth"], seed=int(seed),
                       temporal_data_available=False, development_only=config["development_only"],
                       expected_split_counts=config["expected_split_counts"],
                       statistical_scope=config["statistical_scope"], certificate=config["certificate"])
        _write_json(directory / "fresh_rt_summary.json", summary)
        reporting.update_status(directory, "complete", completed_macros=info["num_macros"])
        reporting.write_artifact_manifest(directory)
        return summary
    except BaseException as exc:
        # Include dependency/scene loading and final cache validation failures,
        # not only exceptions inside the tracing log context. Keep every partial
        # artifact for diagnosis and never turn an incomplete run into a cache.
        try:
            reporting.update_status(directory, "failed", error=f"{type(exc).__name__}: {exc}")
            reporting.write_artifact_manifest(directory)
        except OSError:
            # A full/unwritable disk must not hide the original failure.
            pass
        raise
