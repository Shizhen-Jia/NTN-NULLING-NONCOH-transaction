"""Resumable activity experiments on frozen, reusable RT channels."""
from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np

from .config import ExperimentConfig, MusicConfig
from .storage import SCHEMA_VERSION, fingerprint, code_signature, atomic_json, atomic_npz, read_npz
from .observations import pooled_observations
from .detection import detect_directions
from .evaluation import prepare_links, final_beams, evaluate_beams


def _utc():
    return datetime.now(timezone.utc).isoformat()


def run_experiment(channels, config: ExperimentConfig, music: MusicConfig, output_root):
    """Checkpoint each full macro/episode. Re-running the same config resumes it."""
    config.validate()
    music.validate()
    rank_margins = [music.rank_margin_for_snapshots(L * config.snapshots_per_window)
                    if music.source_estimation == "rank" else music.rank_noise_margin
                    for L in config.listening_windows]
    channels = sorted(channels, key=lambda c: c.macro_id)
    if not channels or len({c.macro_id for c in channels}) != len(channels):
        raise ValueError("Require nonempty channels with unique macro IDs")
    for c in channels:
        c.validate()
    if len({(c.f_dl, c.f_ul) for c in channels}) != 1:
        raise ValueError("All macros must share the same frequency pair")
    dependencies = ["BeamformingCalc.py", "ntn_music_detection.py", "multipath_support.py", "nulling_cdf_utils.py"]
    dependencies += [f"listening_experiments/{x}.py" for x in
                     ("config", "storage", "runner", "observations", "detection", "evaluation")]
    identity = dict(schema=SCHEMA_VERSION, experiment=config, music=music,
                    channels=[dict(macro_id=c.macro_id, source_id=c.source_id,
                                   f_dl=c.f_dl, f_ul=c.f_ul, shape=list(c.h_ntn_dl.shape),
                                   provenance=c.metadata) for c in channels],
                    code=code_signature(dependencies),
                    estimator=f"finite-snapshot {music.source_estimation.upper()}-MUSIC; spatial smoothing; correlated power fit",
                    rank_noise_margins_by_budget=rank_margins,
                    normalization="Y/sqrt(P_UL); fitted powers are record-averaged channel gains",
                    aggregation="equal macro, equal episode, equal protected user or served link")
    key = fingerprint(identity)
    directory = Path(output_root) / key[:20]
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "ac_config.json", dict(run_id=key, **identity))
    print(f"Run outputs: {directory.resolve()}", flush=True)
    total = len(channels) * config.episodes_per_scene
    completed = 0
    status_path = directory / "status.json"
    atomic_json(status_path, dict(status="running", completed=0, total=total, updated=_utc()))
    try:
        for c in channels:
            pairs, protected = prepare_links(c, config)
            mrt_inr, mrt_sinr = evaluate_beams(c, pairs, [pair["w_t"] for pair in pairs], protected, config)
            for episode in range(config.episodes_per_scene):
                path = directory / "episodes" / f"macro_{c.macro_id:04d}_episode_{episode:04d}.npz"
                if path.exists():
                    saved = read_npz(path)
                    if str(saved["run_id"]) != key or int(saved["macro_id"]) != c.macro_id or int(saved["episode_id"]) != episode:
                        raise ValueError(f"Checkpoint identity mismatch: {path}")
                    print(f"[{completed + 1}/{total}] macro={c.macro_id} episode={episode}: checkpoint hit", flush=True)
                else:
                    P, K = len(config.p_values), len(config.listening_windows)
                    N, B = len(protected), len(pairs)
                    inr = np.empty((P, K, N))
                    sinr = np.empty((P, K, B))
                    counts = np.empty((P, K, N), dtype=np.int64)
                    source_counts = np.empty((P, K, B), dtype=np.int64)
                    accepted = np.empty_like(source_counts)
                    noise = np.empty((P, K, B))
                    fit = np.empty_like(noise)
                    peak_rows = []
                    for l_index, (L, covariance, J) in enumerate(pooled_observations(c, config, episode)):
                        for p_index, p in enumerate(config.p_values):
                            estimates, diag = detect_directions(
                                covariance[p_index], positions_dl=c.array_positions_dl,
                                orientations=c.orientations, f_ul=c.f_ul, f_dl=c.f_dl,
                                snapshots=L * config.snapshots_per_window,
                                noise_normalized=config.sensing_noise_normalized, config=music,
                                power_correction=config.ul_dl_power_correction)
                            beams = final_beams(pairs, estimates, config)
                            inr[p_index, l_index], sinr[p_index, l_index] = evaluate_beams(c, pairs, beams, protected, config)
                            counts[p_index, l_index] = J[p_index, protected]
                            source_counts[p_index, l_index] = [d["num_sources"] for d in diag]
                            accepted[p_index, l_index] = [d["accepted_peaks"] for d in diag]
                            noise[p_index, l_index] = [d["noise_estimate"] for d in diag]
                            fit[p_index, l_index] = [d["covariance_fit"] for d in diag]
                            for b, estimate in enumerate(estimates):
                                peak_rows.extend((p, L, b, ph, th, g) for ph, th, g in
                                                 zip(estimate["phi"], estimate["theta"], estimate["g"]))
                        print(f"  macro={c.macro_id}, episode={episode}, L={L}, T={L * config.snapshots_per_window}: done", flush=True)
                    if not np.all(np.isfinite(inr)) or not np.all(np.isfinite(sinr)):
                        raise ValueError("Nonfinite physical INR/SINR; refusing to checkpoint")
                    atomic_npz(path, run_id=np.array(key), macro_id=np.array(c.macro_id), episode_id=np.array(episode),
                               p_values=np.asarray(config.p_values), listening_windows=np.asarray(config.listening_windows),
                               ue_ids=protected, tn_user_ids=np.array([p["tn_idx"] for p in pairs]),
                               tx_ids=np.arange(B), round_id=np.array(0), J=counts,
                               inr_mrt_linear=mrt_inr, inr_final_linear=inr,
                               sinr_mrt_linear=mrt_sinr, sinr_final_linear=sinr,
                               estimated_sources=source_counts, accepted_peaks=accepted,
                               rank_noise_margins_by_budget=np.asarray(rank_margins),
                               noise_estimate=noise, covariance_fit=fit,
                               peak_columns=np.array(["p", "L", "tx_id", "phi_deg", "theta_deg", "gain_dl"]),
                               peaks=np.asarray(peak_rows, dtype=float).reshape(-1, 6))
                    print(f"[{completed + 1}/{total}] saved {path.name}", flush=True)
                completed += 1
                atomic_json(status_path, dict(status="running", completed=completed, total=total, updated=_utc()))
        atomic_json(status_path, dict(status="complete", completed=completed, total=total, updated=_utc()))
    except BaseException as error:
        atomic_json(status_path, dict(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                     completed=completed, total=total, updated=_utc(),
                                     error=f"{type(error).__name__}: {error}"))
        raise
    return directory


def load_completed_run(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "ac_config.json").read_text())
    config = ExperimentConfig(**manifest["experiment"])
    config.validate()
    records = []
    for channel in manifest["channels"]:
        for episode in range(config.episodes_per_scene):
            path = directory / "episodes" / f'macro_{channel["macro_id"]:04d}_episode_{episode:04d}.npz'
            if not path.exists():
                raise FileNotFoundError(f"Incomplete run: {path}; resume run_experiment before plotting")
            record = read_npz(path)
            if str(record["run_id"]) != manifest["run_id"] or int(record["macro_id"]) != channel["macro_id"] or int(record["episode_id"]) != episode:
                raise ValueError(f"Checkpoint metadata mismatch: {path}")
            if not np.array_equal(record["p_values"], config.p_values) or not np.array_equal(record["listening_windows"], config.listening_windows):
                raise ValueError(f"Checkpoint grid mismatch: {path}")
            records.append(record)
    return manifest, config, records
