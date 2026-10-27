"""Small smoke run, final consistency checks, and artifact provenance."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np

from .cache import load_inputs
from .runner import create_run, run_sweep, read_completed, utc_now
from .storage import atomic_json, file_hash, read_npz, archive_sources, run_log


def run_smoke(directory, config, music):
    directory = Path(directory)
    smoke_config = replace(config, num_scenes=min(3, config.num_scenes), repeats=min(2, config.repeats),
                           snapshots=tuple(config.cdf_snapshots),
                           output_root=str(directory / "checks/smoke"))
    cases, physical, observation, manifest = load_inputs(smoke_config)
    pointer = directory / "checks/smoke_directory.json"
    resume = json.loads(pointer.read_text())["directory"] if pointer.exists() else None
    smoke_dir = create_run(smoke_config, music, manifest, resume_directory=resume)
    atomic_json(pointer, dict(directory=str(smoke_dir)))
    run_sweep(cases, smoke_config, music, physical, observation, smoke_dir)
    atomic_json(directory / "smoke_validation.json", dict(passed=True, directory=str(smoke_dir),
                scenes=len(cases), repeats=smoke_config.repeats, snapshots=smoke_config.snapshots))
    return smoke_dir


def compare_smoke_prefixes(directory):
    directory = Path(directory)
    path = directory / "checks/smoke_directory.json"
    if not path.exists():
        return dict(performed=False)
    smoke = Path(json.loads(path.read_text())["directory"])
    _, records = read_completed(smoke)
    checked = 0
    for r in records:
        full = read_npz(directory / "episodes" / f"macro_{int(r['macro_id']):04d}_repeat_{int(r['repeat_id']):03d}.npz")
        indices = [int(np.flatnonzero(full["snapshots"] == t)[0]) for t in r["snapshots"]]
        for name in ("inr_linear", "sinr_linear", "tp_fp_fn", "detected_ue_mask", "num_sources", "accepted_peaks",
                     "path_power_coverage", "high_inr_ue_counts"):
            np.testing.assert_allclose(full[name][indices], r[name], rtol=1e-12, atol=1e-20)
        checked += len(indices)
    result = dict(performed=True, passed=True, network_detection_comparisons=checked,
                  meaning="Adding intermediate T budgets leaves shared-prefix outputs unchanged")
    atomic_json(directory / "prefix_replay_validation.json", result)
    return result


def finalize_run(directory, config):
    directory = Path(directory)
    _, records = read_completed(directory)
    required = ["snapshot_detection_ab.pdf", "snapshot_detection_ab.png", "snapshot_inr_cdf.pdf",
                "snapshot_inr_cdf.png", "detection_summary.csv", "nulling_summary.csv", "render_validation.json"]
    for name in required:
        if not (directory / name).is_file():
            raise FileNotFoundError(directory / name)
    archive_sources(directory, config)
    compare_smoke_prefixes(directory)
    status = dict(status="complete", completed_episodes=len(records),
                  total_episodes=config.num_scenes*config.repeats,
                  completed_network_detections=len(records)*len(config.snapshots),
                  total_network_detections=config.num_scenes*config.repeats*len(config.snapshots), updated=utc_now())
    atomic_json(directory / "status.json", status)
    # Archived sources include any execution-time plotting refinements.
    artifacts = [dict(path=str(p.relative_to(directory)), bytes=p.stat().st_size, sha256=file_hash(p))
                 for p in sorted(directory.rglob("*")) if p.is_file() and p.name != "artifacts.json"]
    atomic_json(directory / "artifacts.json", artifacts)
    return status


def execute(config, music, *, resume_directory=None, smoke=True):
    from .validation import check_observations, validate_noise
    from .analysis import analyze_run
    from .plotting import plot_results
    cases, physical, observation, manifest = load_inputs(config)
    directory = create_run(config, music, manifest, resume_directory=resume_directory)
    print(f"Output directory: {directory}", flush=True)
    with run_log(directory):
        atomic_json(directory / "observation_validation.json", check_observations(cases[0], observation))
        validate_noise(directory, cases[0], config, music)
        if smoke:
            run_smoke(directory, config, music)
        run_sweep(cases, config, music, physical, observation, directory)
        analyze_run(directory)
        plot_results(directory)
    finalize_run(directory, config)
    return directory

