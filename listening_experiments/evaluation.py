"""Fixed TN scheduling and linear-domain INR/SINR evaluation."""
import numpy as np
from BeamformingCalc import nulling_bf_music_noncoh
from nulling_cdf_utils import pair_tn_to_strongest_tx, _interference_power_per_rx, _tn_link_power
from .config import noise_watts, watts


def prepare_links(channels, config):
    pairing = pair_tn_to_strongest_tx(
        channels.h_tn_dl, h_tn_th=0.0, tx_antennas=channels.h_tn_dl.shape[-1],
        tx_power=watts(config.bs_power_dbm),
        snr_noise_power=noise_watts(config.bandwidth_dl, config.tn_noise_figure_db),
        eps=config.channel_epsilon)
    # One fixed round per scene, as specified in Section 9. All cached sector
    # drops must support it; do not silently discard failing sectors/scenes.
    if pairing["min_count"] < 1:
        raise ValueError(f"Macro {channels.macro_id}: at least one sector has no TN pair; inspect the RT drop")
    pairs = [pairing["pairs_by_tx"][b][0] for b in range(channels.h_tn_dl.shape[2])]
    protected = np.flatnonzero(np.any(np.abs(channels.h_ntn_dl) > config.channel_epsilon, axis=(1, 2, 3)))
    if not len(protected):
        raise ValueError(f"Macro {channels.macro_id} has no DL-coupled protected users")
    return pairs, protected


def final_beams(pairs, directions, config):
    return [nulling_bf_music_noncoh(pair["h_tn"], pair["w_r"], estimates["u"],
                                  estimates["g"], config.lambda_weight,
                                  eps=config.channel_epsilon)[0]
            for pair, estimates in zip(pairs, directions)]


def evaluate_beams(channels, pairs, beams, protected, config):
    """Sum independent BS interference powers; retain coherent paths within each link."""
    bs_power = watts(config.bs_power_dbm)
    ntn_noise = noise_watts(config.bandwidth_dl, config.ntn_noise_figure_db)
    tn_noise = noise_watts(config.bandwidth_dl, config.tn_noise_figure_db)
    inr = sum(_interference_power_per_rx(channels.h_ntn_dl[:, :, b, :], beam)
              for b, beam in enumerate(beams)) * bs_power / ntn_noise
    sinr = []
    for b, pair in enumerate(pairs):
        desired = _tn_link_power(pair["h_tn"], beams[b], pair["w_r"]) * bs_power
        interference = sum(
            _tn_link_power(channels.h_tn_dl[pair["tn_idx"], :, other, :].T,
                           beams[other], pair["w_r"]) * bs_power
            for other in range(len(beams)) if other != b)
        sinr.append(desired / (tn_noise + interference))
    return np.asarray(inr)[protected], np.asarray(sinr)
