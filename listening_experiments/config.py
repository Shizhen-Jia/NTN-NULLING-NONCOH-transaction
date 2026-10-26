"""Explicit physical, sensing, and service-time configurations."""
from dataclasses import dataclass, asdict
import math


def watts(dbm):
    return 10.0 ** ((dbm - 30.0) / 10.0)


def noise_watts(bandwidth, noise_figure):
    return watts(-174.0 + 10.0 * math.log10(bandwidth) + noise_figure)


@dataclass(frozen=True)
class RTConfig:
    # scene_path: str = "Denver_scene/10kmwithfigure/10km.xml"
    scene_path: str = "Denver_scene/10kmwithfigure/scenes/flat/scene.xml"
    ntn_users: int = 20
    tx_rows: int = 8
    tx_cols: int = 8
    f_dl: float = 7e9
    f_ul: float = 6.3e9
    bandwidth: float = 200e6
    max_depth: int = 3
    ntn_los_mode: str = "natural"
    position_seed: int = 20260908
    satellite_seed: int = 20260909
    rt_seed: int = 42
    satellite_elevation: tuple = (25.0, 90.0)
    satellite_azimuth: tuple = (0.0, 360.0)
    bs_power_dbm: float = 35.0
    bs_noise_figure_db: float = 2.0
    tn_outdoor_probability: float = 0.5
    tn_association_margin_db: float = 3.0
    tn_link_threshold_db: float = -6.0
    tn_max_attempts: int = 256
    tn_candidates_per_batch: int = 16
    ntn_building_ratio: float = 0.8
    bs_boundary: float = 2500.0
    sector_pitch_deg: float = 6.0
    samples_per_src: int = 100_000
    max_num_paths_per_src: int = 100_000

    def validate(self):
        for name in ("ntn_users", "tx_rows", "tx_cols", "samples_per_src",
                     "max_num_paths_per_src", "tn_max_attempts", "tn_candidates_per_batch"):
            positive_int(getattr(self, name), name)
        if isinstance(self.max_depth, bool) or int(self.max_depth) != self.max_depth or self.max_depth < 0:
            raise ValueError("max_depth must be a nonnegative integer")
        if min(self.f_dl, self.f_ul, self.bandwidth) <= 0:
            raise ValueError("Frequencies and bandwidth must be positive")
        if self.f_dl != self.f_ul and abs(self.f_dl - self.f_ul) <= self.bandwidth:
            raise ValueError("UL/DL bands overlap with the configured equal bandwidths")
        if not 0 <= self.tn_outdoor_probability <= 1 or not 0 <= self.ntn_building_ratio <= 1:
            raise ValueError("Deployment probabilities must be in [0, 1]")

    @property
    def tn_min_channel_norm(self):
        return math.sqrt(10 ** (self.tn_link_threshold_db / 10) *
                         noise_watts(self.bandwidth, self.bs_noise_figure_db) *
                         self.tx_rows * self.tx_cols / watts(self.bs_power_dbm))


@dataclass(frozen=True)
class ExperimentConfig:
    p_values: tuple = (0.1, 0.3, 1.0)
    listening_windows: tuple = (1, 4, 16, 64)
    snapshots_per_window: int = 128
    episodes_per_scene: int = 5
    seed: int = 20261013
    lambda_weight: float = 1e11
    ul_power_dbm: float = 23.0
    bs_power_dbm: float = 35.0
    bandwidth_ul: float = 200e6
    bandwidth_dl: float = 200e6
    bs_noise_figure_db: float = 2.0
    tn_noise_figure_db: float = 7.0
    ntn_noise_figure_db: float = 3.0
    inr_threshold_db: float = -3.0
    channel_epsilon: float = 1e-12
    ul_dl_power_correction: bool = True

    def validate(self):
        for name in ("snapshots_per_window", "episodes_per_scene"):
            positive_int(getattr(self, name), name)
        if self.snapshots_per_window < 2:
            raise ValueError("At least two snapshots per window are required")
        if not self.p_values or len(set(self.p_values)) != len(self.p_values):
            raise ValueError("p_values must be nonempty and unique")
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in self.p_values):
            raise ValueError("Activity probabilities must be in [0, 1]")
        if not self.listening_windows or tuple(sorted(set(self.listening_windows))) != tuple(self.listening_windows):
            raise ValueError("listening_windows must be unique and increasing")
        for L in self.listening_windows:
            positive_int(L, "listening window count")
        if not math.isfinite(self.lambda_weight) or self.lambda_weight < 0:
            raise ValueError("lambda_weight must be finite and nonnegative")
        if min(self.bandwidth_ul, self.bandwidth_dl, self.channel_epsilon) <= 0:
            raise ValueError("Bandwidths and channel_epsilon must be positive")
        for name, value in asdict(self).items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")

    @property
    def sensing_noise_normalized(self):
        # Work with Y / sqrt(P_UL): fitted powers retain channel-gain units.
        return noise_watts(self.bandwidth_ul, self.bs_noise_figure_db) / watts(self.ul_power_dbm)


@dataclass(frozen=True)
class MusicConfig:
    spatial_smoothing: bool = True
    subarray_rows: int = 6
    subarray_cols: int = 6
    forward_backward: bool = True
    grid_step_deg: float = 2.0  # Pilot; use 0.5 to match the reference notebook.
    peak_refine: bool = True
    peak_refine_half_width_deg: float = 1.0
    peak_refine_maxiter: int = 40
    max_noise_projection: float = 0.2
    max_peak_correlation: float = 0.98
    forward_only: bool = True
    top_k: int | None = None
    energy_fraction: float = 1.0
    source_estimation: str = "rank"
    rank_relative_threshold: float = 1e-5
    rank_noise_margin: float = 0.45
    # None retains a constant margin, including when loading old run configs.
    # The notebook enables the calibrated 128-snapshot reference explicitly.
    rank_noise_reference_snapshots: int | None = None

    def validate(self):
        if self.source_estimation not in {"mdl", "rank"}:
            raise ValueError("source_estimation must be 'mdl' or 'rank'")
        for name in ("rank_relative_threshold", "rank_noise_margin"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.rank_noise_reference_snapshots is not None:
            positive_int(self.rank_noise_reference_snapshots, "rank_noise_reference_snapshots")
        positive_int(self.subarray_rows, "subarray_rows")
        positive_int(self.subarray_cols, "subarray_cols")
        positive_int(self.peak_refine_maxiter, "peak_refine_maxiter")
        if not 0 < self.grid_step_deg <= 30:
            raise ValueError("grid_step_deg must lie in (0, 30]")
        if not 0 <= self.max_noise_projection <= 1 or not 0 < self.max_peak_correlation <= 1:
            raise ValueError("Invalid peak acceptance settings")
        if not 0 < self.energy_fraction <= 1 or self.peak_refine_half_width_deg < 0:
            raise ValueError("Invalid path selection/refinement settings")
        if self.top_k is not None:
            positive_int(self.top_k, "top_k")

    def rank_margin_for_snapshots(self, snapshots):
        """Fixed noise-calibrated rule, independent of activity, channels and INR."""
        positive_int(snapshots, "snapshots")
        reference = self.rank_noise_reference_snapshots
        if reference is None:
            return self.rank_noise_margin
        if snapshots < reference:
            raise ValueError("Snapshot count is below the rank calibration reference; recalibrate or use a constant margin")
        return self.rank_noise_margin * math.sqrt(reference / snapshots)


@dataclass(frozen=True)
class TimingConfig:
    """All times are in units of one listening window tau_0; not measured hardware."""
    horizon: float = 256.0
    listen: float = 1.0
    switch: float = 0.1  # BOTH retunings per window combined.
    gap: float = 1.0
    processing: float = 1.0

    def durations(self, windows):
        positive_int(windows, "windows")
        values = asdict(self)
        if any(not math.isfinite(v) or v < 0 for v in values.values()) or self.listen <= 0:
            raise ValueError("Timing parameters must be finite and nonnegative, listen > 0")
        rf = windows * (self.listen + self.switch)
        mrt = (windows - 1) * self.gap + self.processing
        ready = rf + mrt
        if ready >= self.horizon:
            raise ValueError(f"L={windows}: t_ready={ready:g} must be below H={self.horizon:g}")
        return dict(rf=rf, ready=ready, mrt=mrt, final=self.horizon - ready,
                    data=self.horizon - rf, horizon=self.horizon)


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 1 or int(value) != value:
        raise ValueError(f"{name} must be a positive integer")
