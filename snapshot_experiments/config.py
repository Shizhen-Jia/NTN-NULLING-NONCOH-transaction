"""Reproducible settings for the two-figure snapshot experiment."""
from dataclasses import dataclass
from pathlib import Path
import math
import json

from listening_experiments.config import MusicConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SnapshotConfig:
    source_run: str = "result/CDF/20261026_220019_877592"
    rank_calibration_file: str = "result/MUSIC_SNAPSHOT/rank_calibration/6e6154c169eb6bf3/calibration.json"
    output_root: str = "result/MUSIC_SNAPSHOT"
    num_scenes: int = 20
    num_ntn_ues: int = 50
    repeats: int = 5
    snapshots: tuple = (64, 128, 256, 512, 1024, 2048, 4096, 8192)
    cdf_snapshots: tuple = (64, 1024, 8192)
    block_size: int = 64
    frequency_hz: float = 8e9
    lambda_weight: float = 1e13
    seed: int = 2026102803
    workers: int = 4
    match_angle_deg: float = 5.0
    high_inr_threshold_db: float = -3.0
    channel_epsilon: float = 1e-12
    noise_trials: int = 1000
    noise_wilson_limit: float = 0.01
    bootstrap_draws: int = 2000
    display_floor_db: float = -120.0
    font_directory: str = "result/CDF/fonts"

    def validate(self):
        for name in ("num_scenes", "num_ntn_ues", "repeats", "block_size", "workers", "noise_trials", "bootstrap_draws"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not self.snapshots or tuple(sorted(set(self.snapshots))) != tuple(self.snapshots):
            raise ValueError("snapshots must be nonempty, unique and increasing")
        if any(isinstance(t, bool) or not isinstance(t, int) or t < 64 or t % self.block_size
               for t in self.snapshots):
            raise ValueError("Every budget must be >=64 and divisible by block_size")
        if not self.cdf_snapshots or not set(self.cdf_snapshots).issubset(self.snapshots):
            raise ValueError("CDF budgets must belong to the computed snapshot grid")
        if self.seed < 0 or not 0 < self.noise_wilson_limit < 1:
            raise ValueError("Invalid seed or noise validation limit")
        if not 0 < self.match_angle_deg < 90 or self.channel_epsilon <= 0:
            raise ValueError("Invalid matching angle or channel threshold")
        if not math.isfinite(self.lambda_weight) or self.lambda_weight < 0 or self.frequency_hz <= 0:
            raise ValueError("Invalid lambda or frequency")
        if not math.isfinite(self.high_inr_threshold_db):
            raise ValueError("The fixed pre-nulling INR threshold must be finite")
        return self

    def path(self, value):
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def reference_music_config(calibration_file=None):
    """Read the independently trained and validated 64-snapshot rank rule."""
    config = SnapshotConfig()
    path = config.path(calibration_file or config.rank_calibration_file)
    report = json.loads(path.read_text())
    if report["reference_snapshots"] != 64 or any(r["wilson_upper"] > .01 for r in report["rows"]):
        raise ValueError("The 64-snapshot calibration has not passed independent noise validation")
    music = MusicConfig(**report["music"])
    music.validate()
    return music

