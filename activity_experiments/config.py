"""Matched sensing conditions and explicit activity/timing assumptions."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ActivityConfig:
    source_run: str = "result/CDF/20261026_220019_877592"
    continuous_run: str = "result/MUSIC_SNAPSHOT/20261026_220701_036881"
    output_root: str = "result/ACTIVITY_LISTENING"
    num_scenes: int = 20
    num_ntn_ues: int = 50
    repeats: int = 5
    p_values: tuple = (0.1, 0.3, 1.0)
    windows: tuple = (1, 4, 8, 16, 32, 64, 66, 96)
    snapshots_per_window: int = 64
    lambdas: tuple = (1e11, 1e13)
    seed: int = 2026102803
    validation_seed: int = 2026102901
    synthetic_seed: int = 2026102902
    workers: int = 8
    match_angle_deg: float = 5.0
    threshold_db: float = -3.0
    bootstrap_draws: int = 2000
    coverage_trials: int = 100000
    conditional_trials: int = 2000

    @property
    def snapshots(self):
        return tuple(x * self.snapshots_per_window for x in self.windows)

    def validate(self):
        if self.num_scenes < 1 or self.repeats < 1 or self.workers < 1:
            raise ValueError("Scene, repeat and worker counts must be positive")
        if self.num_ntn_ues != 50 or self.snapshots_per_window != 64:
            raise ValueError("This study uses the matched 50-UE, 64-sample-window setup")
        if tuple(sorted(set(self.windows))) != tuple(self.windows) or min(self.windows) < 1:
            raise ValueError("Window budgets must be positive, unique and increasing")
        if any(not 0 <= p <= 1 for p in self.p_values) or len(set(self.p_values)) != len(self.p_values):
            raise ValueError("Activity probabilities must be unique and in [0, 1]")
        if 1.0 not in self.p_values or any(x <= 0 for x in self.lambdas):
            raise ValueError("Include continuous activity and positive penalty weights")
        return self

    def path(self, value):
        path = Path(value)
        return path.resolve() if path.is_absolute() else (ROOT / path).resolve()
