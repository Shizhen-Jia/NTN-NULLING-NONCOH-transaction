"""Atomic checkpoints, provenance archives, and notebook-visible run logs."""
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from pathlib import Path
import shutil
import sys

from listening_experiments.storage import (atomic_json, atomic_npz, canonical_json,
                                           file_hash, fingerprint, read_npz)
from .config import PROJECT_ROOT

SHARED_FILES = ["BeamformingCalc.py", "ntn_music_detection.py", "multipath_support.py",
                "nulling_cdf_utils.py"] + [f"listening_experiments/{name}.py" for name in
                ("__init__", "cache", "config", "storage", "observations", "detection", "evaluation",
                 "calibrate_rank", "analysis", "runner")]


def computation_signature():
    files = SHARED_FILES + [f"snapshot_experiments/{name}.py"
                           for name in ("config", "cache", "metrics", "runner", "storage")]
    return {name: file_hash(PROJECT_ROOT / name) for name in files}


def archive_sources(directory, config):
    files = [PROJECT_ROOT / name for name in SHARED_FILES]
    files += list((PROJECT_ROOT / "snapshot_experiments").glob("*.py"))
    for path in files:
        target = directory / "sources" / path.relative_to(PROJECT_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    notebook = PROJECT_ROOT / "MUSIC_Snapshot_Experiment.ipynb"
    if notebook.exists():
        shutil.copy2(notebook, directory / "sources" / notebook.name)
    font_dir = config.path(config.font_directory)
    (directory / "sources/fonts").mkdir(parents=True, exist_ok=True)
    for path in font_dir.glob("*.TTF"):
        shutil.copy2(path, directory / "sources/fonts" / path.name)
    calibration = config.path(config.rank_calibration_file)
    target = directory / "sources/rank_calibration"
    target.mkdir(parents=True, exist_ok=True)
    for path in calibration.parent.iterdir():
        if path.is_file():
            shutil.copy2(path, target / path.name)


class _Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.stream.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self):
        self.stream.flush()
        self.log.flush()


@contextmanager
def run_log(directory):
    with (Path(directory) / "run.log").open("a", buffering=1) as log:
        with redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
            yield

