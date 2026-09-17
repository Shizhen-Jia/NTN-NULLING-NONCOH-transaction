"""Run archives for the sector-drop notebook; no changes to the physical model."""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import csv
import hashlib
import json
import platform
import shutil
import sys
import traceback

import numpy as np


def _now():
    return datetime.now().astimezone().isoformat()


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=lambda x: np.asarray(x).tolist()) + "\n")


def update_status(result_dir, status, **details):
    path = Path(result_dir) / "run_status.json"
    record = json.loads(path.read_text()) if path.exists() else {"started_at": _now()}
    record.update(status=status, updated_at=_now(), **details)
    if status in ("complete", "failed"):
        record["finished_at"] = _now()
    _write_json(path, record)


def create_run_directory(parent="result"):
    directory = Path(parent) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory.mkdir(parents=True, exist_ok=False)
    update_status(directory, "running")
    return directory


def save_run_metadata(result_dir, config):
    directory = Path(result_dir)
    _write_json(directory / "run_config.json", config)
    packages = {}
    for name in ("numpy", "scipy", "matplotlib", "sionna", "sionna-rt", "mitsuba",
                 "drjit", "nbclient", "nbformat", "ipykernel"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    _write_json(directory / "environment.json", dict(
        recorded_at=_now(), executable=sys.executable, python=sys.version,
        platform=platform.platform(), packages=packages,
    ))
    sources = directory / "sources"
    sources.mkdir(exist_ok=True)
    manifest = {}
    names = ["Nulling_CDF_SectorDrop.ipynb", "SceneConfigSionnaSectorDrop.py",
             "tn_sector_drop.py", "multipath_support.py", "ntn_music_detection.py",
             "nulling_cdf_utils.py", "BeamformingCalc.py", "sector_drop_reporting.py",
             "run_sector_drop.py", "TN_SECTOR_DROP.md", "FDD_MULTIPATH_ANALYSIS.md"]
    for name in names:
        source = Path(name)
        if source.is_file():
            shutil.copy2(source, sources / name)
            manifest[name] = dict(sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                                  size_bytes=source.stat().st_size)
    scene = Path(config.get("scene_path", ""))
    if scene.is_file():
        manifest[str(scene)] = dict(sha256=hashlib.sha256(scene.read_bytes()).hexdigest(),
                                    size_bytes=scene.stat().st_size, copied=False)
    _write_json(directory / "source_manifest.json", manifest)


class _Tee:
    def __init__(self, original, log):
        self.original, self.log = original, log

    def write(self, text):
        self.original.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self):
        self.original.flush()
        self.log.flush()

    def __getattr__(self, name):
        return getattr(self.original, name)


@contextmanager
def capture_run_log(result_dir):
    """Keep notebook output visible while persisting Python stdout/stderr."""
    with (Path(result_dir) / "run.log").open("a", buffering=1) as log:
        with redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
            try:
                yield
            except BaseException as exc:
                traceback.print_exc()
                update_status(result_dir, "failed", error=f"{type(exc).__name__}: {exc}")
                raise


def save_summary(result_dir, experiment):
    rows = []
    for percentage, case in experiment["by_percentage"].items():
        for metric in ("inr", "snr", "sinr"):
            groups = [("no_nulling", "", case[f"raw_{metric}_db"])]
            for prefix in ("est", "music_real"):
                groups.extend((prefix, float(lam), values)
                              for lam, values in case.get(f"{prefix}_{metric}_db", {}).items())
            for method, lam, values in groups:
                data = np.asarray(values).ravel()
                finite = data[np.isfinite(data)]
                quantiles = np.percentile(finite, [10, 50, 90]) if finite.size else (None,)*3
                rows.append(dict(ul_percentage=float(percentage),
                                 ul_frequency_hz=float(case["ul_frequency_hz"]),
                                 ul_dl_power_correction=bool(case.get("ul_dl_power_correction", False)),
                                 ul_dl_power_scale=float(case.get("ul_dl_power_scale", 1.0)),
                                 metric=metric, method=method, lambda_value=lam,
                                 macro_count=len(case["macro_stats"]), count=int(data.size),
                                 finite_count=int(finite.size), p10_db=quantiles[0],
                                 median_db=quantiles[1], p90_db=quantiles[2]))
    with (Path(result_dir) / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_artifact_manifest(result_dir):
    directory = Path(result_dir)
    _write_json(directory / "artifacts.json", [
        dict(path=str(path.relative_to(directory)), size_bytes=path.stat().st_size)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "artifacts.json"
    ])


def finish_run(result_dir):
    directory = Path(result_dir)
    required = ["run_config.json", "nulling_cdf_metrics.npz", "summary.csv",
                "cdf_style_map.json", "nulling_inr_cdf.png", "nulling_inr_cdf.pdf",
                "nulling_tn_sinr_cdf.png", "nulling_tn_sinr_cdf.pdf"]
    config = json.loads((directory / "run_config.json").read_text())
    if config.get("plot_snr"):
        required += ["nulling_tn_snr_cdf.png", "nulling_tn_snr_cdf.pdf"]
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Run archive is incomplete: {missing}")
    update_status(directory, "complete")
    write_artifact_manifest(directory)
