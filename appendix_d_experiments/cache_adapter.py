"""Read-only bridge from SectorDrop caches to Appendix-D spatial validation.

This module does not infer temporal sensing kernels or risk certificates from
static snapshots.  All fitted channel sets are *empirical* calibration sets.
True NTN DL channels appear only in the offline fit and held-out evaluation.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _csv_write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _db(value: Any, floor_db: float = -120.0) -> np.ndarray:
    """Only plotting is floored; all raw powers, including exact zero, are saved."""
    return 10 * np.log10(np.maximum(np.asarray(value, dtype=float), 10 ** (floor_db / 10)))


def inspect_cache(cache_dir: str | Path) -> dict:
    """Inspect saved channels without Sionna or a conic solver."""
    root = Path(cache_dir).expanduser().resolve()
    config_file = root / "run_config.json"
    if not config_file.is_file():
        raise FileNotFoundError(f"Missing SectorDrop configuration: {config_file}")
    config = json.loads(config_file.read_text())
    channels = sorted((root / "channels").glob("channels_*.npz"))
    if not channels:
        raise ValueError(f"No coherent DL channel caches under {root / 'channels'}")
    ids = [int(p.stem.split("_")[-1]) for p in channels]
    with np.load(channels[0], allow_pickle=False) as data:
        shape_tn, shape_ntn = list(data["h_tn"].shape), list(data["h_ntn"].shape)
    cases = []
    percentages = config.get("ul_frequency_percentages", [])
    for index, pct in enumerate(percentages):
        case_path = root / "channels" / f"ul_{index:03d}"
        present = sorted(int(p.stem.split("_")[-1]) for p in case_path.glob("music_*.npz"))
        if present:
            cases.append({"case": index, "offset_percent": float(pct), "macro_ids": present})
    return {
        "cache_dir": str(root), "macro_ids": ids, "num_macros": len(ids),
        "h_tn_shape": shape_tn, "h_ntn_shape": shape_ntn, "ul_cases": cases,
        "f_dl_hz": float(config["f_dl"]), "tx_power_w": float(config["tx_power"]),
        "tn_noise_w": float(config["snr_noise_power"]),
        "ntn_noise_w": float(config["inr_noise_power"]),
        "channel_model": config.get("channel_model", "unspecified in source cache"),
        "temporal_data_available": False,
    }


def _basis(peaks: np.ndarray, antennas: int) -> np.ndarray:
    """One anonymous joint span, not separately constrained individual paths."""
    if peaks.size == 0:
        return np.empty((antennas, 0), dtype=complex)
    if peaks.ndim != 2 or peaks.shape[1] != antennas:
        raise ValueError(f"Unexpected MUSIC array shape {peaks.shape}; expected (*,{antennas})")
    if not np.all(np.isfinite(peaks)):
        raise ValueError("MUSIC cache contains nonfinite steering vectors")
    left, singular, _ = np.linalg.svd(peaks.T, full_matrices=False)
    keep = singular > max(float(singular[0]) * 1e-8, 1e-14)
    return left[:, keep]


def _load_macro(root: Path, macro_id: int, ul_case: int, config: dict) -> list[dict]:
    with np.load(root / "channels" / f"channels_{macro_id:04d}.npz", allow_pickle=False) as data:
        h_tn = np.asarray(data["h_tn"], dtype=complex)
        h_ntn = np.asarray(data["h_ntn"], dtype=complex)
    if h_tn.ndim != 4 or h_ntn.ndim != 4 or h_ntn.shape[2:] != h_tn.shape[2:]:
        raise ValueError("Channel arrays must have compatible [UE,RX,TX,TX_ANT] shapes")
    if h_ntn.shape[1] != 1:
        raise ValueError("This adapter requires a single NTN receive branch; supply fixed-combiner effective channels for multi-antenna NTN receivers")
    if not np.all(np.isfinite(h_tn)) or not np.all(np.isfinite(h_ntn)):
        raise ValueError(f"Nonfinite channel values in macro {macro_id}")
    with np.load(root / "channels" / f"ul_{ul_case:03d}" / f"music_{macro_id:04d}.npz", allow_pickle=False) as data:
        if "peak_u_used_for_dl" not in data.files:
            raise ValueError("Cache has no UL-inferred DL manifold; raw UL vectors cannot silently substitute in FDD")
        peaks = np.asarray(data["peak_u_used_for_dl"], dtype=complex)
        peak_tx = np.asarray(data["peak_t_idx"], dtype=int)
    if len(peaks) != len(peak_tx):
        raise ValueError("MUSIC peak direction/sector index length mismatch")
    antennas = h_tn.shape[-1]
    powers = np.sum(np.abs(h_tn) ** 2, axis=(1, 3))
    associations = np.argmax(powers, axis=1)
    sectors = []
    pmax, noise = float(config["tx_power"]), float(config["inr_noise_power"])
    for tx in range(h_tn.shape[2]):
        candidates = np.flatnonzero(associations == tx)
        # The fixed single-user calendar chooses lowest UE index if a sector
        # has multiple associations.  It never depends on NTN truth or Gamma.
        tn_id = int(candidates[0]) if len(candidates) else None
        if tn_id is not None:
            matrix = h_tn[tn_id, :, tx, :].T
            _, _, vh = np.linalg.svd(matrix, full_matrices=False)
            combiner = vh.conj().T[:, 0]
            effective_tn = matrix @ combiner
        else:
            effective_tn = None
        # Existing repository convention is |v^H H w_r|^2, so no additional
        # conjugation of stored channel vectors is applied here.
        full_ntn = h_ntn[:, 0, tx, :] * np.sqrt(pmax / noise)
        A = _basis(peaks[peak_tx == tx], antennas)
        coeff = full_ntn @ A.conj()
        remainder = full_ntn - coeff @ A.T
        sectors.append({"macro_id": macro_id, "sector_id": tx, "tn_id": tn_id,
                        "h": effective_tn, "f": full_ntn, "A": A,
                        "coefficient_norms": np.linalg.norm(coeff, axis=1),
                        "residual_norms": np.linalg.norm(remainder, axis=1)})
    return sectors


def _empirical_quantile(values: Iterable[float], probability: float) -> float:
    return float(np.quantile(list(values), probability, method="higher"))


def _plot_cdfs(out: Path, rows: list[dict], key: str, xlabel: str, basename: str,
               gamma_values: list[float], floor_db: float, db: bool = True) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 4.1), constrained_layout=True)
    for gamma in [None] + gamma_values:
        subset = [r[key] for r in rows if r["gamma_db"] == gamma]
        if not subset:
            continue
        x = np.sort(_db(subset, floor_db) if db else np.asarray(subset))
        label = "No nulling" if gamma is None else rf"$\Gamma={gamma:g}$ dB"
        ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", label=label)
    ax.set(xlabel=xlabel, ylabel="Empirical CDF", ylim=(0, 1.02))
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    ax.set_title("Held-out ray-tracing cache: static, one controlled sector\n"
                 "Includes solutions failing the fixed TN SNR floor", fontsize=9)
    for extension in ("pdf", "png"):
        fig.savefig(out / f"{basename}.{extension}", dpi=200)
    plt.close(fig)


def run_cached_spatial(output_dir: str | Path, cache_dir: str | Path,
                       gamma_db: Iterable[float] = (-15, -10, -5, 0), seed: int = 20260921,
                       max_sectors: int | None = None, ul_case: int | None = None,
                       tn_snr_floor_db: float = -6.0, calibration_quantile: float = .95,
                       solver: str | None = None) -> dict:
    """Calibrate on whole drops, then sweep Gamma with a fixed TN SNR floor.

    The SNR floor is a *static feasibility diagnostic*, not a replacement for
    Appendix D's deadline probability constraints. Every maximum-TN-service
    robust beam remains in the CDF, including those unable to meet the floor;
    these settings are clearly counted as infeasible. No QoS is auto-relaxed.
    """
    from .robust import ProtectionSet, solve_robust_beam, support

    info = inspect_cache(cache_dir)
    root = Path(info["cache_dir"])
    out = Path(output_dir).expanduser().resolve()
    if out == root or root in out.parents:
        raise ValueError("Write new results outside the preserved source-cache directory")
    gamma_values = [float(g) for g in gamma_db]
    if not gamma_values or not np.all(np.isfinite(gamma_values)):
        raise ValueError("gamma_db must contain finite thresholds")
    if not 0 < calibration_quantile <= 1:
        raise ValueError("calibration_quantile must be in (0,1]")
    if max_sectors is not None and max_sectors < 1:
        raise ValueError("max_sectors must be positive")
    cases = info["ul_cases"]
    if not cases:
        raise ValueError("No cached FDD MUSIC observations found")
    if ul_case is None:
        nonzero = [c for c in cases if c["offset_percent"] != 0]
        choice = min(nonzero or cases, key=lambda c: abs(c["offset_percent"] + 10))
    else:
        matching = [c for c in cases if c["case"] == ul_case]
        if not matching:
            raise ValueError(f"No cached MUSIC case {ul_case}; available: {[c['case'] for c in cases]}")
        choice = matching[0]
    ul_case = int(choice["case"])
    complete_ids = sorted(set(info["macro_ids"]) & set(choice["macro_ids"]))
    if len(complete_ids) < 6:
        raise ValueError(f"Need at least 6 complete whole-macro drops for disjoint train/calibration/test; found {len(complete_ids)}")
    config = json.loads((root / "run_config.json").read_text())
    for name in ("tx_power", "snr_noise_power", "inr_noise_power"):
        if not np.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f"Positive finite {name} is required")
    permutation = np.random.default_rng(seed).permutation(complete_ids)
    ntrain, ncal = max(2, int(.4 * len(permutation))), max(2, int(.3 * len(permutation)))
    split_ids = {"train": permutation[:ntrain], "calibration": permutation[ntrain:ntrain+ncal],
                 "test": permutation[ntrain+ncal:]}
    splits = {name: [sector for mid in ids for sector in _load_macro(root, int(mid), ul_case, config)]
              for name, ids in split_ids.items()}
    # Train sets only set relative coefficient/residual scales. Calibration
    # then takes a whole-macro maximum score, so correlated sectors and UEs
    # are never advertised as independent calibration samples.
    C0 = max(max(float(np.max(s["coefficient_norms"])) for s in splits["train"]), 1e-12)
    rho0 = max(max(float(np.max(s["residual_norms"])) for s in splits["train"]), 1e-12)
    macro_scores = {}
    for sector in splits["calibration"]:
        score = max(float(np.max(sector["coefficient_norms"])) / C0,
                    float(np.max(sector["residual_norms"])) / rho0)
        mid = sector["macro_id"]
        macro_scores[mid] = max(macro_scores.get(mid, 0.0), score)
    inflation = max(1.0, _empirical_quantile(macro_scores.values(), calibration_quantile))
    C, rho = C0 * inflation, rho0 * inflation
    coverage = []
    for name, sectors in splits.items():
        for s in sectors:
            covered = (s["coefficient_norms"] <= C * (1 + 1e-10)) & (s["residual_norms"] <= rho * (1 + 1e-10))
            for ue, inside in enumerate(covered):
                coverage.append({"split": name, "macro_id": s["macro_id"], "sector_id": s["sector_id"],
                                 "ntn_id": ue, "coefficient_norm": s["coefficient_norms"][ue],
                                 "residual_norm": s["residual_norms"][ue], "C": C, "rho": rho,
                                 "decomposition_inside_set": int(inside), "ul_subspace_rank": s["A"].shape[1],
                                 "zero_dl_channel": int(np.linalg.norm(s["f"][ue]) == 0)})
    out.mkdir(parents=True, exist_ok=True)
    _csv_write(out / "calibration_coverage.csv", coverage)
    tn_rows, ntn_rows = [], []
    test = [s for s in splits["test"] if s["tn_id"] is not None]
    if max_sectors is not None:
        test = test[:max_sectors]
    if not test:
        raise ValueError("No held-out sectors have a scheduled TN user")
    snr_scale = float(config["tx_power"]) / float(config["snr_noise_power"])
    threshold_tn = 10 ** (tn_snr_floor_db / 10)
    for sector in test:
        h = sector["h"]
        norm_h = float(np.linalg.norm(h))
        direction = h / norm_h if norm_h else np.zeros_like(h)
        protection = ProtectionSet(A=sector["A"], C=C, rho=rho, name="anonymous_UL_union_plus_background")
        for threshold_db in [None] + gamma_values:
            if threshold_db is None:
                v, status, violation = direction, "baseline_unconstrained", 0.0
            else:
                # Unit h gives good conditioning; scaling back occurs in SNR.
                solution = solve_robust_beam(direction, [protection], 10 ** (threshold_db / 10), solver=solver)
                v, status, violation = solution.v, solution.status, float(solution.max_violation)
                if v is None or not np.all(np.isfinite(v)):
                    raise RuntimeError(f"Robust solve failed for macro {sector['macro_id']}, sector {sector['sector_id']}: {status}")
            v = np.asarray(v).reshape(-1)
            snr = float(abs(np.vdot(h, v)) ** 2 * snr_scale)
            power = float(np.vdot(v, v).real)
            inr = np.abs(sector["f"].conj() @ v) ** 2
            bound = float(support(v, protection) ** 2)
            tn_rows.append({"macro_id": sector["macro_id"], "sector_id": sector["sector_id"], "tn_id": sector["tn_id"],
                            "gamma_db": threshold_db, "snr_linear": snr, "snr_db_plot": float(_db(snr)),
                            "tn_snr_floor_db": tn_snr_floor_db, "static_tn_feasible": int(snr >= threshold_tn * (1 - 1e-6)),
                            "power_fraction": power, "tx_power_w": power * config["tx_power"],
                            "robust_inr_bound": bound, "solver_status": status, "max_solver_violation": violation,
                            "ul_subspace_rank": sector["A"].shape[1]})
            for ue, value in enumerate(inr):
                ntn_rows.append({"macro_id": sector["macro_id"], "sector_id": sector["sector_id"], "ntn_id": ue,
                                 "gamma_db": threshold_db, "inr_linear": float(value), "inr_db_plot": float(_db(value)),
                                 "zero_dl_channel": int(np.linalg.norm(sector["f"][ue]) == 0),
                                 "exceeds_gamma": None if threshold_db is None else int(value > 10 ** (threshold_db / 10) * (1 + 1e-6))})
    summary_rows = []
    for threshold_db in [None] + gamma_values:
        tn = [r for r in tn_rows if r["gamma_db"] == threshold_db]
        ntn = [r for r in ntn_rows if r["gamma_db"] == threshold_db]
        summary_rows.append({"gamma_db": threshold_db, "sectors": len(tn), "ntn_sector_pairs": len(ntn),
                             "tn_static_feasible_fraction": float(np.mean([r["static_tn_feasible"] for r in tn])),
                             "median_tn_snr_db": float(np.median([r["snr_db_plot"] for r in tn])),
                             "median_power_fraction": float(np.median([r["power_fraction"] for r in tn])),
                             "ntn_inr_p95_db": float(_db(np.quantile([r["inr_linear"] for r in ntn], .95))),
                             "spatial_exceedance_fraction": None if threshold_db is None else float(np.mean([r["exceeds_gamma"] for r in ntn]))})
    _csv_write(out / "tn_samples.csv", tn_rows)
    _csv_write(out / "ntn_samples.csv", ntn_rows)
    _csv_write(out / "gamma_summary.csv", summary_rows)
    _plot_cdfs(out, ntn_rows, "inr_linear", "Single controlled-sector INR (dB; zeros clipped at -120 dB)", "cached_inr_cdf", gamma_values, -120)
    _plot_cdfs(out, tn_rows, "snr_linear", "TN SNR (dB)", "cached_tn_snr_cdf", gamma_values, -120)
    _plot_cdfs(out, tn_rows, "power_fraction", "Transmit power / maximum power", "cached_power_cdf", gamma_values, -120, db=False)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3), constrained_layout=True)
    for name in splits:
        selected = [r for r in coverage if r["split"] == name]
        scores = np.sort([max(r["coefficient_norm"] / C, r["residual_norm"] / rho) for r in selected])
        axes[0].step(scores, np.arange(1, len(scores) + 1) / len(scores), where="post", label=name)
    axes[0].axvline(1, color="k", ls="--", lw=.8)
    axes[0].set(xlabel="Normalized full-channel decomposition score", ylabel="Empirical CDF", ylim=(0, 1.02))
    axes[0].legend(fontsize=8)
    axes[1].plot(gamma_values, [r["tn_static_feasible_fraction"] for r in summary_rows[1:]], "o-")
    axes[1].set(xlabel=r"INR threshold $\Gamma$ (dB)", ylabel="TN static feasible fraction", ylim=(-.02, 1.02))
    axes[1].set_title(f"Fixed TN SNR floor: {tn_snr_floor_db:g} dB", fontsize=9)
    for ax in axes:
        ax.grid(alpha=.25)
    for extension in ("pdf", "png"):
        fig.savefig(out / f"cached_coverage_and_tn_feasibility.{extension}", dpi=200)
    plt.close(fig)
    latex = [r"\begin{tabular}{lrrrr}", r"\toprule", r"$\Gamma$ (dB) & TN feasible & TN median SNR & Median $p$ & INR $95\%$ \\", r"\midrule"]
    for row in summary_rows:
        label = "No nulling" if row["gamma_db"] is None else f"{row['gamma_db']:g}"
        latex.append(f"{label} & {row['tn_static_feasible_fraction']:.3f} & {row['median_tn_snr_db']:.2f} & {row['median_power_fraction']:.3f} & {row['ntn_inr_p95_db']:.2f} " + r"\\")
    latex += [r"\bottomrule", r"\end{tabular}"]
    (out / "gamma_summary.tex").write_text("\n".join(latex) + "\n")
    metadata = {
        "status": "completed", "experiment": "E4/E5 static cached spatial validation; supplementary to dynamic E7",
        "source": info, "ul_case": ul_case, "ul_offset_percent": choice["offset_percent"],
        "split_macro_ids": {k: list(map(int, v)) for k, v in split_ids.items()},
        "seed": seed, "gamma_db": gamma_values, "fixed_tn_snr_floor_db": tn_snr_floor_db,
        "C": C, "rho": rho, "training_C": C0, "training_rho": rho0,
        "calibration_inflation": inflation, "calibration_quantile": calibration_quantile,
        "calibration_macro_scores": {str(k): v for k, v in macro_scores.items()},
        "evaluated_sectors": len(test), "held_out_sectors_available": len(splits["test"]),
        "all_ntn_including_zero_and_missed_in_evaluation": True,
        "source_config_sha256": hashlib.sha256((root / "run_config.json").read_bytes()).hexdigest(),
        "scope": [
            "Empirical offline calibration only: finite sample maxima/quantiles are not a conditional tail-risk certificate.",
            "Whole macro drops are disjoint; sectors and UEs within a drop are correlated, and all drops reuse the source geometry.",
            "Coverage reports an explicit orthogonal decomposition sufficient for set membership; it may undercount exact Minkowski-sum membership.",
            "chi=1 and source-cache NTN noise power are model assumptions; effective cached channels already include the simulated receiver response.",
            "The online beam reads own TN CSI, UL-derived anonymous span, and fixed offline C/rho; held-out NTN truth is used only for evaluation.",
            "One controlled sector at a time: plotted INR is that sector's contribution, not aggregate network INR.",
            "Static snapshots have no activity labels, deadlines, mobility age or sensing latency. Spatial exceedance is not active-time outage probability.",
            "TN SNR floor is held fixed and failed settings are reported; dynamic TN deadline constraints require the joint-policy experiment.",
            "All maximum-service robust solutions appear in CDFs, including static-TN-infeasible settings; there is no hidden filtering or auto-relaxation.",
            "Exact zero interference is retained in raw CSV and clipped only for plotting at -120 dB.",
        ],
        "summary": summary_rows,
    }
    _json_write(out / "summary.json", metadata)
    return metadata
