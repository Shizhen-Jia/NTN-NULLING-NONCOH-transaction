"""Scene-weighted statistics, paired cluster bootstrap, and service-cost accounting."""
from pathlib import Path
from dataclasses import asdict
import csv
import math
import numpy as np

from .config import TimingConfig, positive_int
from .runner import load_completed_run
from .storage import atomic_json, atomic_npz, fingerprint, code_signature


def episode_metrics(record, timing, threshold_linear):
    Ls = record["listening_windows"]
    fractions = [timing.durations(int(L)) for L in Ls]
    mrt = np.array([d["mrt"] for d in fractions])[None, :, None]
    final = np.array([d["final"] for d in fractions])[None, :, None]
    data = np.array([d["data"] for d in fractions])[None, :, None]
    exceed = record["inr_final_linear"] > threshold_linear
    mrt_exceed = record["inr_mrt_linear"] > threshold_linear
    rate_final = np.log2(1 + record["sinr_final_linear"])
    rate_mrt = np.log2(1 + record["sinr_mrt_linear"])
    shape = exceed.shape[:2]
    return dict(
        q_miss=np.any(record["J"] == 0, axis=-1).astype(float),
        q_out=np.mean(exceed, axis=-1), q_any=np.any(exceed, axis=-1).astype(float),
        q_mrt=np.full(shape, np.mean(mrt_exceed)),
        q_data=np.mean((mrt * mrt_exceed + final * exceed) / data, axis=-1),
        r_eff=np.mean((mrt * rate_mrt + final * rate_final) / timing.horizon, axis=-1),
        r_mrt=np.full(shape, np.mean(rate_mrt)),
        r_mrt_scheduled=np.broadcast_to((data[..., 0] / timing.horizon) * np.mean(rate_mrt), shape).copy(),
        empty_detection_fraction=np.mean(record["accepted_peaks"] == 0, axis=-1))


def coverage_exact(p, windows, protected_counts):
    n = np.asarray(protected_counts)
    if p == 0:
        return 1.0
    if p == 1:
        return 0.0
    zero = (1 - p) ** windows
    return float(np.mean(-np.expm1(n * np.log1p(-zero))))


def _wilson(successes, n):
    z = 1.959963984540054
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    radius = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def coverage_experiment(p_values, windows, protected_counts, *, samples, seed):
    """Cheap activity-only trials, independent of the expensive detector trials."""
    positive_int(samples, "coverage_samples")
    counts = np.asarray(protected_counts, dtype=int)
    if counts.size == 0 or np.any(counts <= 0):
        raise ValueError("Protected sets must be nonempty")
    rows = []
    for pi, p in enumerate(p_values):
        for L in windows:
            rng = np.random.default_rng(np.random.SeedSequence([seed, pi, int(L)]))
            n = rng.choice(counts, size=samples)
            hits = rng.binomial(int(L), p, size=(samples, int(counts.max())))
            missed = np.any((hits == 0) & (np.arange(counts.max())[None, :] < n[:, None]), axis=1)
            value = float(np.mean(missed))
            lo, hi = _wilson(int(np.sum(missed)), samples)
            if p in (0, 1):
                lo = hi = value  # Deterministic under the stipulated model.
            rows.append(dict(p=float(p), L=int(L), q_miss_mc=value, lower=lo, upper=hi,
                             q_miss_exact=coverage_exact(p, int(L), counts),
                             union_bound=float(np.mean(np.minimum(1, counts * (1 - p) ** int(L)))),
                             activity_trials=int(samples)))
    return rows


def _write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_run(run_dir, timing=TimingConfig(), *, threshold_db=None, bootstrap_draws=2000,
                coverage_samples=100_000, seed=20261014, coverage_epsilon=0.05):
    """Recompute metrics/plots from checkpoints; no RT or MUSIC calls."""
    positive_int(bootstrap_draws, "bootstrap_draws")
    if not 0 < coverage_epsilon < 1:
        raise ValueError("coverage_epsilon must lie in (0, 1)")
    manifest, config, records = load_completed_run(run_dir)
    threshold_db = config.inr_threshold_db if threshold_db is None else float(threshold_db)
    if not math.isfinite(threshold_db):
        raise ValueError("threshold_db must be finite")
    threshold = 10 ** (threshold_db / 10)
    macro_ids = [int(c["macro_id"]) for c in manifest["channels"]]
    grouped = {m: [] for m in macro_ids}
    protected = {}
    for record in records:
        macro = int(record["macro_id"])
        if macro in protected and not np.array_equal(protected[macro], record["ue_ids"]):
            raise ValueError("Protected user sets changed across activity trials")
        protected[macro] = record["ue_ids"]
        grouped[macro].append(episode_metrics(record, timing, threshold))
    names = list(next(iter(grouped.values()))[0])
    by_scene = {name: np.stack([np.mean([r[name] for r in grouped[m]], axis=0)
                               for m in macro_ids]) for name in names}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(macro_ids), size=(bootstrap_draws, len(macro_ids)))
    summaries = {}
    for name, values in by_scene.items():
        lower, upper = np.quantile(values[draws].mean(axis=1), [0.025, 0.975], axis=0)
        summaries[name] = (values.mean(axis=0), lower, upper)
    rows = []
    for pi, p in enumerate(config.p_values):
        for li, L in enumerate(config.listening_windows):
            d = timing.durations(int(L))
            row = dict(p=float(p), L=int(L), T=int(L * config.snapshots_per_window),
                       independent_scenes=len(macro_ids), episodes_per_scene=config.episodes_per_scene,
                       protected_users_min=min(map(len, protected.values())),
                       protected_users_max=max(map(len, protected.values())),
                       threshold_db=threshold_db, rf_fraction=d["rf"] / timing.horizon,
                       ready_fraction=d["ready"] / timing.horizon,
                       mrt_fraction=d["mrt"] / timing.horizon, final_fraction=d["final"] / timing.horizon)
            for name, (estimate, lo, hi) in summaries.items():
                row[name] = float(estimate[pi, li])
                row[name + "_lower"] = float(lo[pi, li]) if len(macro_ids) > 1 else None
                row[name + "_upper"] = float(hi[pi, li]) if len(macro_ids) > 1 else None
            row["zero_observed_final_exceedance"] = bool(row["q_out"] == 0)
            rows.append(row)
    coverage = coverage_experiment(config.p_values, config.listening_windows,
                                   [len(protected[m]) for m in macro_ids], samples=coverage_samples, seed=seed + 1)
    analysis_config = dict(timing=asdict(timing), threshold_db=threshold_db, bootstrap_draws=bootstrap_draws,
                           coverage_samples=coverage_samples, seed=seed, coverage_epsilon=coverage_epsilon,
                           code=code_signature(["listening_experiments/analysis.py", "listening_experiments/config.py"]))
    directory = Path(run_dir) / "analysis" / fingerprint(analysis_config)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    _write_csv(directory / "ac_summary.csv", rows)
    _write_csv(directory / "coverage_model.csv", coverage)
    selected = [r for r in rows if (r["p"] == 0.1 and r["L"] in (1, 16, 57, 64)) or (r["p"] == 1 and r["L"] == 64)]
    if selected:
        _write_csv(directory / "ac_working_points.csv", selected)
    atomic_npz(directory / "scene_metrics.npz", macro_ids=np.array(macro_ids), **by_scene)
    warnings = []
    if len(macro_ids) < 2:
        warnings.append("Only one independent scene: scene-level confidence intervals are unavailable.")
    if any(r["zero_observed_final_exceedance"] for r in rows):
        warnings.append("Zero observed exceedances do not establish zero risk; degenerate bootstrap intervals are not risk certificates.")
    output = dict(directory=str(directory.resolve()), run_dir=str(Path(run_dir).resolve()),
                  summary=rows, coverage=coverage, analysis_config=analysis_config,
                  experiment=manifest["experiment"], music=manifest["music"],
                  protected_counts=[len(protected[m]) for m in macro_ids],
                  independent_scenes=len(macro_ids), warnings=warnings,
                  intervals="pointwise paired scene-cluster bootstrap; coverage-only MC uses Wilson intervals")
    atomic_json(directory / "analysis.json", output)
    print(f"Analysis outputs: {directory.resolve()}", flush=True)
    for warning in warnings:
        print(warning, flush=True)
    return output
