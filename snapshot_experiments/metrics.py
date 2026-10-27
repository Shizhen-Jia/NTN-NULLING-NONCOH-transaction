"""Offline one-to-one angular truth matching; never used by the beam designer."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def unit_vectors(phi_degrees, theta_degrees):
    phi, theta = np.deg2rad(phi_degrees), np.deg2rad(theta_degrees)
    return np.column_stack((np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)))


def match_directions(truth, directions, max_angle_deg=5.0):
    """Return per-sector TP/FP/FN, globally deduplicated UE coverage and match rows."""
    valid = np.asarray(truth["valid"], bool)
    detected = np.zeros(valid.shape[0], dtype=bool)
    counts = np.zeros((valid.shape[1], 3), dtype=np.int64)
    matches, peaks = [], []
    peak_offset = 0
    for tx, estimates in enumerate(directions):
        keys = np.argwhere(valid[:, tx, :])
        phi, theta = np.asarray(estimates["phi"]), np.asarray(estimates["theta"])
        peaks.extend((tx, ph, th, gain) for ph, th, gain in zip(phi, theta, estimates["g"]))
        counts[tx, 1:] = (len(phi), len(keys))
        if len(keys) and len(phi):
            rays = unit_vectors(np.rad2deg(truth["phi_t"][keys[:, 0], tx, keys[:, 1]]),
                                np.rad2deg(truth["theta_t"][keys[:, 0], tx, keys[:, 1]]))
            distance = np.rad2deg(np.arccos(np.clip(rays @ unit_vectors(phi, theta).T, -1, 1)))
            rr, cc = linear_sum_assignment(np.where(distance <= max_angle_deg, distance, 1e6))
            for r, c in zip(rr, cc):
                if distance[r, c] <= max_angle_deg:
                    rx, path = keys[r]
                    detected[rx] = True
                    counts[tx] += (1, -1, -1)
                    matches.append((tx, rx, path, peak_offset+c, distance[r, c]))
        peak_offset += len(phi)
    return dict(counts=counts, detected_mask=detected,
                matches=np.asarray(matches, float).reshape(-1, 5),
                peaks=np.asarray(peaks, float).reshape(-1, 4))


def to_db(values, floor_db=-120.0):
    """Display conversion only; all threshold decisions use retained linear power."""
    values = np.asarray(values, float)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Power samples must be finite and nonnegative")
    return 10*np.log10(np.maximum(values, 10**(floor_db/10)))


def coverage_metrics(truth, evaluation, ue_ids, raw_inr, high_inr_threshold_db):
    """Power-weighted ray recall and fixed high-INR UE recall for offline evaluation.

    Ray powers are summed separately, without coherent cross terms. Their recall
    is a coverage diagnostic, not the fraction of physical interference removed.
    The high-INR population is selected once from the no-nulling network INR.
    """
    valid = np.asarray(truth["valid"], bool)
    powers = np.asarray(truth["path_power"], float)
    if powers.shape != valid.shape or not np.all(np.isfinite(powers)) or np.any(powers < 0):
        raise ValueError("Invalid truth path powers")
    matched = np.zeros_like(valid)
    for sector, user, ray, _, _ in evaluation["matches"]:
        matched[int(user), int(sector), int(ray)] = True
    if np.any(matched & ~valid):
        raise ValueError("A matched path is absent from the truth catalog")
    power_by_sector = np.column_stack(((powers*matched).sum(axis=(0, 2)),
                                      (powers*valid).sum(axis=(0, 2))))
    ue_ids, raw_inr = np.asarray(ue_ids, int), np.asarray(raw_inr, float)
    if ue_ids.shape != raw_inr.shape or np.any(raw_inr < 0) or not np.all(np.isfinite(raw_inr)):
        raise ValueError("Invalid fixed no-nulling UE population")
    high_ids = ue_ids[raw_inr > 10**(high_inr_threshold_db/10)]
    high_counts = np.array([evaluation["detected_mask"][high_ids].sum(), len(high_ids)], dtype=np.int64)
    return dict(path_power_by_sector=power_by_sector, high_inr_ue_counts=high_counts,
                high_inr_ue_ids=high_ids)

