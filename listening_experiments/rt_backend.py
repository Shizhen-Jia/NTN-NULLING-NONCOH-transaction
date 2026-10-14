"""The only module that imports Sionna; used solely for missing RT cache entries."""
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
import numpy as np

from .config import RTConfig


class RTBackend:
    def __init__(self, config: RTConfig, project_root, cache_dir):
        from sionna.rt import load_scene
        from SceneConfigSionnaSectorDrop import SceneConfigSionna

        self.config = config
        self.cache_dir = Path(cache_dir)
        self.scene_config = SceneConfigSionna(load_scene(str((Path(project_root) / config.scene_path).resolve())))
        self.scene_config.build_coverage_map(grid_size=10, show_xy=True, plot=False)

    def trace(self, macro_id):
        from multipath_support import collapse_cir_to_narrowband
        from nulling_cdf_utils import _scene_tx_array_positions_local

        c, scene = self.config, self.scene_config
        satellite_rng = np.random.default_rng(np.random.SeedSequence([c.satellite_seed, macro_id]))
        azimuth = float(satellite_rng.uniform(*c.satellite_azimuth))
        elevation = float(satellite_rng.uniform(*c.satellite_elevation))
        state = np.random.get_state()
        try:
            np.random.seed(np.random.SeedSequence([c.position_seed, macro_id]).generate_state(1)[0])
            scene.compute_positions(
                ntn_rx=c.ntn_users, tn_rx=12, azimuth=azimuth, elevation=elevation,
                centerBS=False, bs_grid=(2, 2), bs_boundary=c.bs_boundary,
                tn_outdoor_probability=c.tn_outdoor_probability,
                tn_association_margin_db=c.tn_association_margin_db,
                tn_min_channel_norm=c.tn_min_channel_norm,
                tn_max_attempts=c.tn_max_attempts, tn_candidates_per_batch=c.tn_candidates_per_batch,
                tn_sector_yaw_offset_rad=0.0, ntn_building_ratio=c.ntn_building_ratio,
                tn_drop_output_dir=str(self.cache_dir / "tn_drop" / f"macro_{macro_id:04d}"),
                plot_grid=False, plot_bs=False, plot_tn=False, plot_ntn=False)
            scene.compute_paths(
                nsect=3, fc=c.f_dl, tx_rows=c.tx_rows, tx_cols=c.tx_cols,
                tn_rx_rows=1, tn_rx_cols=1, max_depth=c.max_depth, ntn_los_mode=c.ntn_los_mode,
                bandwidth=c.bandwidth, tx_power_dbm=c.bs_power_dbm,
                sector_yaw_offset_rad=0.0, sector_pitch_rad=np.deg2rad(c.sector_pitch_deg),
                sector_roll_rad=0.0,
                propagation_options=dict(specular_reflection=True, refraction=True,
                                         diffuse_reflection=False, diffraction=False,
                                         samples_per_src=c.samples_per_src,
                                         max_num_paths_per_src=c.max_num_paths_per_src,
                                         seed=c.rt_seed))
        finally:
            np.random.set_state(state)
        h_tn, h_dl = collapse_cir_to_narrowband(scene.a_tn), collapse_cir_to_narrowband(scene.a_ntn)
        _, cir_ul, _ = scene.compute_ntn_ul_paths(c.f_ul, max_depth=c.max_depth)
        geometry = _scene_tx_array_positions_local(scene)
        if geometry is None:
            raise ValueError("Sionna did not provide exact array positions")
        arrays = dict(macro_id=np.array(macro_id), h_tn_dl=h_tn, h_ntn_dl=h_dl,
                      h_ntn_ul=collapse_cir_to_narrowband(cir_ul), array_positions_dl=geometry,
                      orientations=np.asarray(scene.tx_orientation_rad), f_dl=np.array(c.f_dl), f_ul=np.array(c.f_ul),
                      tn_positions=np.asarray(scene.tn_pos), ntn_positions=np.asarray(scene.rx_ntn_pos),
                      bs_positions=np.asarray(scene.tx_pos), satellite_angles=np.array([azimuth, elevation]))
        packages = {}
        for package in ("sionna", "sionna-rt", "mitsuba", "numpy"):
            try:
                packages[package] = version(package)
            except PackageNotFoundError:
                pass
        return arrays, dict(origin="fresh_sector_drop_rt", packages=packages,
                            channel_model="coherent narrowband CIR sum at time_index=0",
                            frequency_units="Hz", array_position_units="DL wavelengths")
