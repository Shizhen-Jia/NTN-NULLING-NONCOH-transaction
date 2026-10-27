"""Read-only source import, fixed scheduling, and baseline agreement checks."""
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np

from listening_experiments.cache import load_legacy_channels
from listening_experiments.config import ExperimentConfig, noise_watts
from listening_experiments.evaluation import evaluate_beams
from listening_experiments.storage import file_hash, read_npz
from nulling_cdf_utils import pair_tn_to_strongest_tx


@dataclass
class SceneCase:
    channel: object
    truth: dict
    pairs: list
    ue_ids: np.ndarray
    baseline_inr: np.ndarray
    baseline_sinr: np.ndarray


def load_inputs(config):
    config.validate()
    source = config.path(config.source_run)
    settings = json.loads((source / "run_config.json").read_text())
    if settings["positions"]["ntn_rx"] != config.num_ntn_ues:
        raise ValueError("The cached NTN deployment does not match num_ntn_ues")
    frequencies = np.asarray(settings["ul_frequencies_hz"], float)
    indices = np.flatnonzero(np.isclose(frequencies, config.frequency_hz, rtol=1e-12, atol=0))
    if len(indices) != 1 or settings["f_dl"] != config.frequency_hz:
        raise ValueError("Select a source archive with exactly one requested co-frequency UL case")
    if config.num_scenes > settings["num_macro_sims"]:
        raise ValueError("The source archive does not contain enough frozen scenes")
    band = f"ul_{int(indices[0]):03d}"
    channels = load_legacy_channels(source / "channels", ul_subdir=band,
                                    num_macros=config.num_scenes,
                                    f_dl=config.frequency_hz, f_ul=config.frequency_hz)
    calibration = config.path(config.rank_calibration_file)
    calibration_report = json.loads(calibration.read_text())
    for channel in channels:
        np.testing.assert_allclose(channel.array_positions_dl, calibration_report["array_positions"],
                                   rtol=0, atol=1e-12)
    bandwidth = float(settings["paths"]["bandwidth"])
    thermal = noise_watts(bandwidth, 0.0)
    physical = ExperimentConfig(
        p_values=(1.0,), listening_windows=tuple(t // config.block_size for t in config.snapshots),
        snapshots_per_window=config.block_size, episodes_per_scene=config.repeats,
        seed=config.seed, lambda_weight=config.lambda_weight, ul_power_dbm=35.0,
        bs_power_dbm=float(10*np.log10(settings["tx_power"]) + 30),
        bandwidth_ul=bandwidth, bandwidth_dl=bandwidth,
        tn_noise_figure_db=float(10*np.log10(settings["snr_noise_power"] / thermal)),
        ntn_noise_figure_db=float(10*np.log10(settings["inr_noise_power"] / thermal)),
        channel_epsilon=config.channel_epsilon)
    physical.validate()
    # Explicit normalized variance avoids silently adopting the listening runner's 23 dBm setting.
    observation = SimpleNamespace(seed=config.seed, p_values=(1.0,),
        snapshots_per_window=config.block_size,
        listening_windows=tuple(t // config.block_size for t in config.snapshots),
        sensing_noise_normalized=float(settings["music"]["detect_noise_var"]))
    if observation.sensing_noise_normalized <= 0:
        raise ValueError("Finite-snapshot validation requires a positive known noise variance")
    paths = [source / "run_config.json", source / band / "nulling_cdf_metrics.npz", calibration]
    cases = []
    for channel in channels:
        if channel.h_ntn_ul.shape[0] != config.num_ntn_ues:
            raise ValueError("The sensing archive has the wrong NTN population")
        macro = channel.macro_id
        truth_file = source / "channels" / band / f"paths_ul_{macro:04d}.npz"
        truth = read_npz(truth_file)
        truth = {k: truth[k] for k in ("valid", "is_los", "path_power", "phi_t", "theta_t")}
        if truth["valid"].shape[:2] != (channel.h_ntn_ul.shape[0], channel.h_ntn_ul.shape[2]):
            raise ValueError(f"Truth/channel dimensions disagree for macro {macro}")
        pairing = pair_tn_to_strongest_tx(channel.h_tn_dl, h_tn_th=settings["h_tn_th"],
            tx_antennas=channel.h_tn_dl.shape[-1], tx_power=settings["tx_power"],
            snr_noise_power=settings["snr_noise_power"], eps=config.channel_epsilon)
        if not np.all(pairing["pair_counts_by_tx"] == 1):
            raise ValueError(f"Macro {macro} does not have exactly one fixed TN pair per sector")
        pairs = [pairing["pairs_by_tx"][b][0] for b in range(channel.h_tn_dl.shape[2])]
        users = np.flatnonzero(np.any(np.abs(channel.h_ntn_dl) > config.channel_epsilon, axis=(1, 2, 3)))
        if not len(users):
            raise ValueError(f"Macro {macro} has no evaluable DL-linked NTN users")
        raw_inr, raw_sinr = evaluate_beams(channel, pairs, [p["w_t"] for p in pairs], users, physical)
        cases.append(SceneCase(channel, truth, pairs, users, raw_inr, raw_sinr))
        paths.extend([Path(channel.metadata["dl_archive"]), Path(channel.metadata["ul_archive"]), truth_file])
    reference = read_npz(source / band / "nulling_cdf_metrics.npz")
    count_i = sum(len(case.ue_ids) for case in cases)
    count_s = sum(len(case.pairs) for case in cases)
    # Same masks, user order and fixed serving pairs as the existing CDF experiment.
    np.testing.assert_allclose(10*np.log10(np.maximum(np.concatenate([c.baseline_inr for c in cases]), 1e-12)),
                               reference["raw_inr_db"][:count_i], rtol=0, atol=1e-8)
    np.testing.assert_allclose(10*np.log10(np.maximum(np.concatenate([c.baseline_sinr for c in cases]), 1e-12)),
                               reference["raw_sinr_db"][:count_s], rtol=0, atol=1e-8)
    manifest = dict(source_run=str(source), frequency_hz=config.frequency_hz, ul_subdirectory=band,
        num_ntn_ues_per_scene=config.num_ntn_ues,
        sensing_noise_normalized=observation.sensing_noise_normalized,
        baseline_agreement_atol_db=1e-8, baseline_agrees_with_cdf=True,
        baseline_inr_count=count_i, baseline_sinr_count=count_s,
        scenes=[dict(macro_id=c.channel.macro_id, ue_ids=c.ue_ids.tolist(),
                     tn_ids=[p["tn_idx"] for p in c.pairs], source_id=c.channel.source_id) for c in cases],
        files=[dict(path=str(p), sha256=file_hash(p)) for p in paths])
    return cases, physical, observation, manifest

