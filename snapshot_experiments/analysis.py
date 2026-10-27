"""Scene-cluster bootstrap, fixed-population CDFs, and concise paper summaries."""
from pathlib import Path
import csv
import numpy as np

from .metrics import to_db
from .runner import read_completed
from .storage import atomic_json, atomic_npz


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def safe_ratio(numerator, denominator):
    a, b = np.broadcast_arrays(np.asarray(numerator, float), np.asarray(denominator, float))
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=b > 0)


def analyze_run(directory):
    directory = Path(directory)
    manifest, records = read_completed(directory)
    config = manifest["config"]
    budgets = np.asarray(config["snapshots"], int)
    scenes, repeats = config["num_scenes"], config["repeats"]
    grouped = [[r for r in records if int(r["macro_id"]) == macro] for macro in range(scenes)]
    for group in grouped:
        if len(group) != repeats:
            raise ValueError("Incomplete repeats for a scene")
        for r in group:
            for name in ("ue_ids", "tn_user_ids", "raw_inr_linear", "raw_sinr_linear", "high_inr_ue_ids"):
                np.testing.assert_array_equal(r[name], group[0][name])
    # Deduplicate deterministic baselines across repeats and snapshot budgets.
    raw_inr = np.concatenate([group[0]["raw_inr_linear"] for group in grouped])
    raw_sinr = np.concatenate([group[0]["raw_sinr_linear"] for group in grouped])
    by_scene_count = np.stack([np.mean([r["detected_ue_mask"].sum(axis=1) for r in group], axis=0)
                               for group in grouped])
    by_scene_confusion = np.stack([np.sum([r["tp_fp_fn"].sum(axis=1) for r in group], axis=0)
                                   for group in grouped])
    by_scene_power = np.stack([np.sum([r["path_power_coverage"].sum(axis=1) for r in group], axis=0)
                               for group in grouped])
    by_scene_high = np.stack([np.sum([r["high_inr_ue_counts"] for r in group], axis=0)
                              for group in grouped])
    total = by_scene_confusion.sum(axis=0)
    precision = safe_ratio(total[:, 0], total[:, 0]+total[:, 1])
    recall = safe_ratio(total[:, 0], total[:, 0]+total[:, 2])
    rng = np.random.default_rng(np.random.SeedSequence([config["seed"], 880001]))
    draws = rng.integers(0, scenes, size=(config["bootstrap_draws"], scenes))
    boot_counts = by_scene_count[draws].mean(axis=1)
    boot_confusion = by_scene_confusion[draws].sum(axis=1)
    boot_p = safe_ratio(boot_confusion[..., 0], boot_confusion[..., 0]+boot_confusion[..., 1])
    boot_r = safe_ratio(boot_confusion[..., 0], boot_confusion[..., 0]+boot_confusion[..., 2])
    def intervals(values):
        return np.stack([np.nanquantile(values[:, t], [.025, .975])
                         if np.isfinite(values[:, t]).any() else [np.nan, np.nan]
                         for t in range(len(budgets))])
    count_ci, precision_ci, recall_ci = map(intervals, (boot_counts, boot_p, boot_r))
    archive = dict(snapshots=budgets, matched_ue_mean=by_scene_count.mean(axis=0),
        matched_ue_lower=count_ci[:, 0], matched_ue_upper=count_ci[:, 1],
        precision=precision, precision_lower=precision_ci[:, 0], precision_upper=precision_ci[:, 1],
        recall=recall, recall_lower=recall_ci[:, 0], recall_upper=recall_ci[:, 1],
        tp_fp_fn=total, raw_inr_linear=raw_inr, raw_sinr_linear=raw_sinr,
        matched_ue_by_scene=by_scene_count, tp_fp_fn_by_scene=by_scene_confusion)
    for name, values in (("power_recall", by_scene_power), ("high_inr_ue_recall", by_scene_high)):
        sums = values.sum(axis=0)
        boot_sums = values[draws].sum(axis=1)
        ci = intervals(safe_ratio(boot_sums[..., 0], boot_sums[..., 1]))
        archive[name] = safe_ratio(sums[:, 0], sums[:, 1])
        archive[name+"_lower"], archive[name+"_upper"] = ci[:, 0], ci[:, 1]
        archive[name+"_by_scene"] = values
        archive[name+"_numerator"], archive[name+"_denominator"] = sums[:, 0], sums[:, 1]
    detection_rows, quality_rows, per_episode = [], [], []
    for ti, budget in enumerate(budgets):
        inr = np.concatenate([r["inr_linear"][ti] for r in records])
        sinr = np.concatenate([r["sinr_linear"][ti] for r in records])
        paired_loss = np.concatenate([to_db(r["raw_sinr_linear"], config["display_floor_db"])
                                     - to_db(r["sinr_linear"][ti], config["display_floor_db"]) for r in records])
        angles = np.concatenate([r["matches"][r["matches"][:, 0] == budget, 5] for r in records])
        archive[f"inr_linear_T{budget}"] = inr
        archive[f"sinr_linear_T{budget}"] = sinr
        detection_rows.append(dict(T=int(budget), scenes=scenes, repeats=repeats,
            deployed_ntn_ues_per_scene=config["num_ntn_ues"],
            matched_ue_mean=float(archive["matched_ue_mean"][ti]),
            matched_ue_lower=float(count_ci[ti, 0]), matched_ue_upper=float(count_ci[ti, 1]),
            precision=float(precision[ti]), precision_lower=float(precision_ci[ti, 0]),
            precision_upper=float(precision_ci[ti, 1]), recall=float(recall[ti]),
            recall_lower=float(recall_ci[ti, 0]), recall_upper=float(recall_ci[ti, 1]),
            TP=int(total[ti, 0]), FP=int(total[ti, 1]), FN=int(total[ti, 2]),
            angular_mae_deg=float(angles.mean()) if len(angles) else None,
            angular_p90_deg=float(np.quantile(angles, .9)) if len(angles) else None,
            covariance_relative_error=float(np.mean([r["covariance_relative_error"][ti] for r in records]))))
        for name in ("power_recall", "high_inr_ue_recall"):
            for suffix in ("", "_lower", "_upper", "_numerator", "_denominator"):
                detection_rows[-1][name+suffix] = float(archive[name+suffix][ti])
        quality_rows.append(dict(T=int(budget), lambda_value=config["lambda_weight"],
            inr_samples=len(inr), sinr_samples=len(sinr),
            inr_median_db=float(to_db(np.median(inr), config["display_floor_db"])),
            inr_p90_db=float(to_db(np.quantile(inr, .9), config["display_floor_db"])),
            inr_fraction_le_minus3=float(np.mean(inr <= 10**(-3/10))),
            inr_zero_count=int(np.sum(inr == 0)),
            tn_sinr_median_db=float(to_db(np.median(sinr), config["display_floor_db"])),
            tn_sinr_p05_db=float(to_db(np.quantile(sinr, .05), config["display_floor_db"])),
            tn_loss_median_db=float(np.median(paired_loss)),
            tn_fraction_loss_gt3=float(np.mean(paired_loss > 3))))
        for r in records:
            tp, fp, fn = r["tp_fp_fn"][ti].sum(axis=0)
            per_episode.append(dict(macro_id=int(r["macro_id"]), repeat_id=int(r["repeat_id"]),
                                    T=int(budget), matched_ues=int(r["detected_ue_mask"][ti].sum()),
                                    TP=int(tp), FP=int(fp), FN=int(fn),
                                    matched_path_power=float(r["path_power_coverage"][ti, :, 0].sum()),
                                    total_path_power=float(r["path_power_coverage"][ti, :, 1].sum()),
                                    high_inr_ues_matched=int(r["high_inr_ue_counts"][ti, 0]),
                                    high_inr_ues_total=int(r["high_inr_ue_counts"][ti, 1])))
    baseline = dict(inr_samples=len(raw_inr), tn_samples=len(raw_sinr),
        inr_p90_db=float(to_db(np.quantile(raw_inr, .9), config["display_floor_db"])),
        inr_fraction_le_minus3=float(np.mean(raw_inr <= 10**(-3/10))),
        tn_sinr_p05_db=float(to_db(np.quantile(raw_sinr, .05), config["display_floor_db"])),
        tn_sinr_median_db=float(to_db(np.median(raw_sinr), config["display_floor_db"])))
    write_csv(directory / "detection_summary.csv", detection_rows)
    write_csv(directory / "nulling_summary.csv", quality_rows)
    write_csv(directory / "episode_detection_metrics.csv", per_episode)
    atomic_npz(directory / "snapshot_metrics.npz", **archive)
    atomic_json(directory / "baseline_summary.json", baseline)
    atomic_json(directory / "analysis_config.json", dict(bootstrap_draws=config["bootstrap_draws"],
        bootstrap_unit="Independent scene; all repeats and T values remain paired",
        detection_count="Equal scene weights; equal repeat weights within each scene",
        precision_recall="Micro ratios from summed one-to-one path matching counts",
        paper_detection_quality=["power_recall", "high_inr_ue_recall"],
        power_recall="Matched ray power divided by total valid ray power, pooled across sectors/scenes/repeats; excludes coherent cross terms",
        high_inr_ue_recall="Detected distinct UEs divided by UEs above a fixed pre-nulling network INR threshold",
        high_inr_threshold_db=config["high_inr_threshold_db"],
        high_inr_population="Fixed across all snapshot budgets; any sector match counts as UE detection, not proof of successful protection",
        cdf="Pooled fixed DL-linked population; zero post-nulling INR retained",
        display_floor_db=config["display_floor_db"], cdf_snapshots=config["cdf_snapshots"]))
    lines = ["# MUSIC snapshot experiment results", "",
             f"{scenes} frozen scenes with {config['num_ntn_ues']} NTN UEs each, {repeats} repeats per scene, 8 GHz UL/DL, lambda={config['lambda_weight']:.0e}.",
             "", "| T | Matched UEs | Power-weighted recall | High-INR UE recall | INR p90 (dB) | INR <= -3 dB | TN SINR p05 (dB) |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for d, q in zip(detection_rows, quality_rows):
        lines.append(f"| {d['T']} | {d['matched_ue_mean']:.2f} | {d['power_recall']:.3f} | {d['high_inr_ue_recall']:.3f} | "
                     f"{q['inr_p90_db']:.2f} | {100*q['inr_fraction_le_minus3']:.1f}% | {q['tn_sinr_p05_db']:.2f} |")
    lines += ["", "Power-weighted recall uses the existing 5-degree one-to-one path matches and sums individual ray powers. It is not the fraction of coherent interference removed.",
              f"High-INR UE recall uses the fixed UE set with no-nulling INR > {config['high_inr_threshold_db']:g} dB; a match at any sector counts once.",
              "", "Original path metrics remain available for reporting false detections and missed weak paths:",
              "", "| T | Path precision | Path recall | Matched angular MAE (deg) |",
              "|---:|---:|---:|---:|"]
    for d in detection_rows:
        error = f"{d['angular_mae_deg']:.3f}" if d['angular_mae_deg'] is not None else "N/A"
        lines.append(f"| {d['T']} | {d['precision']:.3f} | {d['recall']:.3f} | {error} |")
    lines += ["", f"No-nulling baseline: {len(raw_inr)} NTN samples and {len(raw_sinr)} TN samples, each counted once.",
              "", "The source-count threshold follows a pre-calibrated T-dependent rule. The experiment measures this full detection protocol, not a fixed-threshold covariance-only effect.",
              "Direction matching uses offline truth for evaluation only; it does not decode UE identities or feed truth into beamforming.",
              "The source's normalized sensing noise is preserved. These results must not be relabeled as a 23 dBm uplink experiment."]
    (directory / "RESULTS.md").write_text("\n".join(lines)+"\n")
    return detection_rows, quality_rows

