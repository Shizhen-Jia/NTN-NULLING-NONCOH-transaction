"""Finite-snapshot detection and nulling with per-episode atomic checkpoints."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import json
import multiprocessing as mp
import platform
import time
import numpy as np

from listening_experiments.detection import detect_directions
from listening_experiments.evaluation import evaluate_beams, final_beams
from listening_experiments.observations import pooled_observations
from .metrics import match_directions, coverage_metrics
from .storage import (atomic_json, atomic_npz, fingerprint, read_npz,
                      computation_signature, archive_sources)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def create_run(config, music, manifest, *, resume_directory=None):
    values = asdict(config)
    simulation = {k: v for k, v in values.items() if k not in
                  ("workers", "output_root", "font_directory", "bootstrap_draws", "display_floor_db", "cdf_snapshots")}
    identity = dict(experiment=simulation, music=asdict(music), channels=manifest,
                    code=computation_signature(), numpy_version=np.__version__)
    key = fingerprint(identity)
    if resume_directory is None:
        stamp = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d_%H%M%S_%f")
        directory = config.path(config.output_root) / stamp
        directory.mkdir(parents=True, exist_ok=False)
    else:
        directory = config.path(resume_directory)
        saved = json.loads((directory / "run_config.json").read_text())
        if saved["run_id"] != key:
            raise ValueError("Resume configuration, source data or computation code changed")
    atomic_json(directory / "run_config.json", dict(run_id=key, config=values, identity=identity,
                python_version=platform.python_version(), created_or_resumed=utc_now(),
                estimator="Finite-snapshot rank-MUSIC with spatial smoothing and correlated-power fitting",
                snapshot_semantics=f"Nested {config.block_size}-sample blocks; continuous UE activity; shared UE symbols and independent sector noise"))
    atomic_json(directory / "channel_manifest.json", manifest)
    if not (directory / "status.json").exists():
        atomic_json(directory / "status.json", dict(status="prepared", updated=utc_now(),
                    total_episodes=config.num_scenes*config.repeats,
                    total_network_detections=config.num_scenes*config.repeats*len(config.snapshots)))
    atomic_json(directory.parent / "latest_run.json", dict(directory=str(directory.resolve()), run_id=key))
    archive_sources(directory, config)
    return directory


def _episode(case, config, music, physical, observation, directory, key, repeat):
    start = time.perf_counter()
    macro, nt = case.channel.macro_id, len(config.snapshots)
    ntx, m = case.channel.h_ntn_ul.shape[2:]
    h = case.channel.h_ntn_ul[:, 0, :, :].conj().transpose(1, 2, 0)
    population = h @ h.swapaxes(-1, -2).conj() + observation.sensing_noise_normalized*np.eye(m)
    inr = np.empty((nt, len(case.ue_ids)))
    sinr = np.empty((nt, len(case.pairs)))
    counts = np.empty((nt, ntx, 3), np.int64)
    detected = np.zeros((nt, case.channel.h_ntn_ul.shape[0]), bool)
    sources = np.empty((nt, ntx), np.int64)
    accepted = np.empty_like(sources)
    noise, fit = np.empty((nt, ntx)), np.empty((nt, ntx))
    covariance_error = np.empty(nt)
    path_power = np.empty((nt, ntx, 2))
    high_counts = np.empty((nt, 2), np.int64)
    peak_rows, match_rows = [], []
    progress = Path(directory) / "progress" / f"macro_{macro:04d}_repeat_{repeat:03d}.json"
    for ti, (blocks, covariance, activity) in enumerate(pooled_observations(case.channel, observation, repeat)):
        budget = blocks * config.block_size
        if budget != config.snapshots[ti] or not np.all(activity == blocks):
            raise ValueError("Observation prefix or continuous-activity invariant failed")
        estimates, diag = detect_directions(covariance[0], positions_dl=case.channel.array_positions_dl,
            orientations=case.channel.orientations, f_ul=case.channel.f_ul, f_dl=case.channel.f_dl,
            snapshots=budget, noise_normalized=observation.sensing_noise_normalized,
            config=music, power_correction=True)
        # The deployed beam designer receives anonymous estimates only.
        beams = final_beams(case.pairs, estimates, physical)
        inr[ti], sinr[ti] = evaluate_beams(case.channel, case.pairs, beams, case.ue_ids, physical)
        evaluation = match_directions(case.truth, estimates, config.match_angle_deg)
        counts[ti], detected[ti] = evaluation["counts"], evaluation["detected_mask"]
        coverage = coverage_metrics(case.truth, evaluation, case.ue_ids, case.baseline_inr,
                                    config.high_inr_threshold_db)
        path_power[ti], high_counts[ti] = coverage["path_power_by_sector"], coverage["high_inr_ue_counts"]
        peak_rows.extend((budget, *row) for row in evaluation["peaks"])
        match_rows.extend((budget, *row) for row in evaluation["matches"])
        sources[ti] = [d["num_sources"] for d in diag]
        accepted[ti] = [d["accepted_peaks"] for d in diag]
        noise[ti] = [d["noise_estimate"] for d in diag]
        fit[ti] = [d["covariance_fit"] for d in diag]
        covariance_error[ti] = np.linalg.norm(covariance[0]-population) / np.linalg.norm(population)
        atomic_json(progress, dict(macro_id=macro, repeat_id=repeat, finished_T=budget,
                                  matched_ues=int(detected[ti].sum()), updated=utc_now()))
    if not np.all(np.isfinite(inr)) or not np.all(np.isfinite(sinr)) or np.any(inr < 0) or np.any(sinr < 0):
        raise ValueError("Nonfinite or negative physical INR/SINR")
    elapsed = time.perf_counter() - start
    path = Path(directory) / "episodes" / f"macro_{macro:04d}_repeat_{repeat:03d}.npz"
    atomic_npz(path, run_id=np.array(key), macro_id=np.array(macro), repeat_id=np.array(repeat),
        snapshots=np.asarray(config.snapshots), seed=np.array(config.seed), ue_ids=case.ue_ids,
        tn_user_ids=np.array([p["tn_idx"] for p in case.pairs]),
        raw_inr_linear=case.baseline_inr, raw_sinr_linear=case.baseline_sinr,
        inr_linear=inr, sinr_linear=sinr, tp_fp_fn=counts, detected_ue_mask=detected,
        num_sources=sources, accepted_peaks=accepted, noise_estimate=noise, covariance_fit=fit,
        covariance_relative_error=covariance_error,
        path_power_coverage=path_power, high_inr_ue_counts=high_counts,
        high_inr_ue_ids=coverage["high_inr_ue_ids"],
        peaks=np.asarray(peak_rows, float).reshape(-1, 5),
        peak_columns=np.array(["T", "sector_id", "phi_deg", "theta_deg", "gain_dl"]),
        matches=np.asarray(match_rows, float).reshape(-1, 6),
        match_columns=np.array(["T", "sector_id", "ue_id", "path_id", "peak_id_within_T", "angle_deg"]),
        ul_linked_ue_ids=np.flatnonzero(np.any(case.truth["valid"], axis=(1, 2))),
        runtime_seconds=np.array(elapsed))
    return macro, repeat, elapsed, int(detected[-1].sum())


def _validate_record(path, key, case, repeat, config):
    record = read_npz(path)
    if str(record["run_id"]) != key or int(record["macro_id"]) != case.channel.macro_id or int(record["repeat_id"]) != repeat:
        raise ValueError(f"Invalid checkpoint identity: {path}")
    np.testing.assert_array_equal(record["snapshots"], config.snapshots)
    np.testing.assert_array_equal(record["ue_ids"], case.ue_ids)
    np.testing.assert_array_equal(record["raw_inr_linear"], case.baseline_inr)
    np.testing.assert_array_equal(record["raw_sinr_linear"], case.baseline_sinr)
    return record


def run_sweep(cases, config, music, physical, observation, directory):
    directory = Path(directory)
    key = json.loads((directory / "run_config.json").read_text())["run_id"]
    total, completed = len(cases)*config.repeats, 0
    todo = []
    for case in cases:
        for repeat in range(config.repeats):
            path = directory / "episodes" / f"macro_{case.channel.macro_id:04d}_repeat_{repeat:03d}.npz"
            if path.exists():
                _validate_record(path, key, case, repeat, config)
                completed += 1
            else:
                todo.append((case, repeat))
    def status(state, **extra):
        atomic_json(directory / "status.json", dict(status=state, completed_episodes=completed,
                    total_episodes=total, completed_network_detections=completed*len(config.snapshots),
                    total_network_detections=total*len(config.snapshots), updated=utc_now(), **extra))
    status("running")
    print(f"Snapshot sweep: {len(cases)} scenes x {config.repeats} repeats x {len(config.snapshots)} budgets; "
          f"{completed} cached episodes, {config.workers} workers", flush=True)
    try:
        if todo:
            with ProcessPoolExecutor(max_workers=config.workers, mp_context=mp.get_context("spawn")) as pool:
                futures = [pool.submit(_episode, case, config, music, physical, observation, directory, key, repeat)
                           for case, repeat in todo]
                for future in as_completed(futures):
                    macro, repeat, seconds, matched = future.result()
                    completed += 1
                    status("running")
                    print(f"[{completed:3d}/{total}] scene={macro:02d}, repeat={repeat}, "
                          f"{seconds:.1f}s, matched UEs at max T={matched}", flush=True)
        status("simulation_complete")
    except BaseException as error:
        status("failed", error=f"{type(error).__name__}: {error}")
        raise
    return directory


def read_completed(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "run_config.json").read_text())
    config = manifest["config"]
    records = []
    for macro in range(config["num_scenes"]):
        for repeat in range(config["repeats"]):
            path = directory / "episodes" / f"macro_{macro:04d}_repeat_{repeat:03d}.npz"
            if not path.exists():
                raise FileNotFoundError(f"Incomplete run: {path}")
            record = read_npz(path)
            if str(record["run_id"]) != manifest["run_id"] or int(record["macro_id"]) != macro or int(record["repeat_id"]) != repeat:
                raise ValueError(f"Wrong checkpoint identity: {path}")
            np.testing.assert_array_equal(record["snapshots"], config["snapshots"])
            records.append(record)
    return manifest, records
