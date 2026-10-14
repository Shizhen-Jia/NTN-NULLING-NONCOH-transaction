"""Independent Fig. 10/11 experiments; ray tracing is imported only on demand."""

from .config import RTConfig, ExperimentConfig, MusicConfig, TimingConfig
from .cache import prepare_channels, load_legacy_channels, load_cached_channels
from .runner import run_experiment
from .analysis import analyze_run
from .plotting import plot_figures

__all__ = ["RTConfig", "ExperimentConfig", "MusicConfig", "TimingConfig",
           "prepare_channels", "load_legacy_channels", "load_cached_channels", "run_experiment",
           "analyze_run", "plot_figures"]
