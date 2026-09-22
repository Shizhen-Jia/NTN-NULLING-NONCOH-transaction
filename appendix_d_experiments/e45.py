"""E4 calibration and E5 spatial optimization experiments.

The default data are a DECLARED SYNTHETIC MODEL, not ray-tracing measurements.
Known finite envelopes are valid by construction. Empirical split calibration
has whole-scene marginal coverage only and is not a conditional-risk input to
the joint policy LP. Figures, CSVs, NPZs and LaTeX summaries are written together.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import beta
from .robust import ProtectionSet, support, worst_case_channel, solve_robust_beam


def _rng_complex(rng, shape):
    return (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2)


def _unit(x):
    n = np.linalg.norm(x)
    return x / n if n else np.zeros_like(x)


def _csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2) + "\n")


def _plot_import():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.grid": True,
                         "grid.alpha": 0.25, "savefig.bbox": "tight",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    return plt


def _save(fig, path):
    fig.savefig(str(path) + ".pdf")
    fig.savefig(str(path) + ".png", dpi=180)


def _interval(k, n, confidence=0.95):
    if n == 0:
        return float("nan"), float("nan")
    a = (1 - confidence) / 2
    return (0.0 if k == 0 else float(beta.ppf(a, k, n-k+1)),
            1.0 if k == n else float(beta.ppf(1-a, k+1, n-k)))


def _latex(path, headers, rows):
    def clean(x):
        return str(x).replace("_", r"\_").replace("%", r"\%")
    lines = [r"\begin{tabular}{" + "l" * len(headers) + "}",
             r"\toprule", " & ".join(map(clean, headers)) + r" \\",
             r"\midrule"]
    lines += [" & ".join(map(clean, row)) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    Path(path).write_text("\n".join(lines))


def _e4_calibration_plan(quick, alpha):
    """Keep the coverage target independent of the computational profile."""
    try:
        alpha = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError("E4 alpha must be a finite number in (0, 1).") from exc
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("E4 alpha must be a finite number in (0, 1).")
    ntrain, ncal, ntest = (32, 64, 128) if quick else (128, 256, 1024)
    order = int(np.ceil((ncal + 1) * (1 - alpha)))
    if order > ncal:
        raise ValueError(
            f"E4 alpha={alpha:g} needs more than {ncal} calibration scenes "
            "for a finite calibrated radius; choose a larger sample profile "
            "or explicitly increase alpha."
        )
    return ntrain, ncal, ntest, alpha, order


def run_e4(output_dir, seed=4104, quick=True, alpha=0.10):
    """Whole-scene split calibration of complete coherent synthetic channels.

    Four observable/model cases are retained: strong and weak accepted records,
    missed UL, and UL-silent/DL-active background. Independent entire scenes
    form train/calibration/test splits. Test labels are evaluator-only.
    alpha is the scene-marginal miscoverage target; quick/full changes only
    sample counts, never this target. Set alpha=0.05 explicitly for 95% coverage.
    """
    ntrain, ncal, ntest, alpha, order = _e4_calibration_plan(quick, alpha)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    nscene = ntrain + ncal + ntest
    m, rank, coefficient_bound = 8, 3, 1.2
    ages = np.asarray([0., 1., 2., 4.])
    durations = np.asarray([1., 2., 4.])
    groups = ["accepted_strong", "accepted_weak", "missed_UL",
              "UL_silent_DL_active"]
    # All f are already sqrt(chi*Pmax/N)-normalized complete DL channels.
    # A has orthonormal columns, so ||A c|| <= C.
    known_rho = np.zeros((4, len(durations), len(ages)))
    for g in range(4):
        quality = 1.0 if g == 0 else 1.6
        residual = quality * (0.16 / np.sqrt(durations[:, None])
                              + 0.075 * ages[None, :])
        known_rho[g] = residual if g < 2 else coefficient_bound + residual
    truth = np.empty((nscene, 4, len(durations), len(ages), m), complex)
    bases = np.empty((nscene, m, rank), complex)
    required_radius = np.zeros(truth.shape[:-1])
    ul_active = np.ones((nscene, 4), dtype=bool)
    ul_active[:, 3] = False
    # DL activity is independent of UL and remains present for silent UEs.
    dl_active = rng.random((nscene, 4)) < np.asarray([.7, .65, .8, .8])
    dl_active[:, 3] = True
    for scene in range(nscene):
        q, _ = np.linalg.qr(_rng_complex(rng, (m, m)))
        a, orth = q[:, :rank], q[:, rank:]
        bases[scene] = a
        severity = .35 + .65 * rng.beta(2.0, 1.5)
        for g in range(4):
            for ti in range(len(durations)):
                for ai in range(len(ages)):
                    c = coefficient_bound * _unit(_rng_complex(rng, rank))
                    c *= .55 + .45 * rng.random()
                    e_direction = orth @ _unit(_rng_complex(rng, m-rank))
                    residual_bound = (known_rho[g, ti, ai] if g < 2 else
                                      known_rho[g, ti, ai] - coefficient_bound)
                    e = residual_bound * severity * (.6 + .4*rng.random()) * e_direction
                    f = a @ c + e
                    truth[scene, g, ti, ai] = f
                    # For g<2, e is orthogonal and ||c||<=C, making this the
                    # EXACT distance to {A c: ||c||<=C}; backgrounds use A=[].
                    required_radius[scene, g, ti, ai] = (
                        np.linalg.norm(e) if g < 2 else np.linalg.norm(f))
    train = slice(0, ntrain)
    cal = slice(ntrain, ntrain+ncal)
    test = slice(ntrain+ncal, nscene)
    # The training shape is frozen before calibration; every calibration scene
    # contributes one maximum score across ALL quality/duration/age strata.
    trained_shape = np.maximum(np.quantile(required_radius[train], .8, axis=0), 1e-8)
    scores = np.max(required_radius[cal] / trained_shape, axis=(1, 2, 3))
    scale = float(np.sort(scores)[order-1])
    calibrated_rho = scale * trained_shape
    test_radii = required_radius[test]
    masks = {
        "known_finite_bound": test_radii <= known_rho + 1e-12,
        "split_calibrated": test_radii <= calibrated_rho + 1e-12,
    }
    no_background = masks["split_calibrated"].copy()
    no_background[:, 2:, :, :] = False
    masks["omit_background"] = no_background
    coverage_rows, radius_rows, raw_rows = [], [], []
    scene_rows = []
    for name, covered in masks.items():
        failed_scene = ~np.all(covered, axis=(1, 2, 3))
        k = int(failed_scene.sum())
        lo, hi = _interval(k, ntest)
        scene_rows.append(dict(method=name, independent_test_scenes=ntest,
                               failed_scenes=k, scene_failure_rate=k/ntest,
                               ci95_low=lo, ci95_high=hi))
        for g, group in enumerate(groups):
            for ti, duration in enumerate(durations):
                for ai, age in enumerate(ages):
                    failed = ~covered[:, g, ti, ai]
                    active = dl_active[test, g]
                    ka, na = int(np.sum(failed & active)), int(active.sum())
                    low, high = _interval(ka, na)
                    coverage_rows.append(dict(
                        method=name, group=group, sensing_duration=duration,
                        age=age, independent_scenes=ntest,
                        failures=int(failed.sum()),
                        failure_rate=float(failed.mean()),
                        dl_active_scenes=na, dl_active_failures=ka,
                        active_failure_rate=ka/na if na else float("nan"),
                        active_ci95_low=low, active_ci95_high=high))
    for g, group in enumerate(groups):
        for ti, duration in enumerate(durations):
            for ai, age in enumerate(ages):
                radius_rows.append(dict(
                    group=group, sensing_duration=duration, age=age,
                    rank=rank if g < 2 else 0,
                    C=coefficient_bound if g < 2 else 0.,
                    rho_known_bound=known_rho[g, ti, ai],
                    rho_calibrated=calibrated_rho[g, ti, ai],
                    n_train_scenes=ntrain, n_calibration_scenes=ncal,
                    alpha_scene_marginal=alpha,
                    conditional_risk_certified=False))
                for j in range(ntest):
                    raw_rows.append(dict(
                        test_scene=j+ntrain+ncal, group=group,
                        sensing_duration=duration, age=age,
                        ul_active=bool(ul_active[test][j, g]),
                        dl_active=bool(dl_active[test][j, g]),
                        required_radius=test_radii[j, g, ti, ai],
                        known_radius=known_rho[g, ti, ai],
                        calibrated_radius=calibrated_rho[g, ti, ai]))
    # Declared toy observation kernel: duration changes detection/acceptance.
    # This is NOT a GLRT or ray-traced detection calibration.
    observation_rows = []
    nobs = ntest
    for duration in durations:
        pd = 1.0 - np.exp(-.75*duration)
        p_quality = .72 + .25*(1.0-np.exp(-duration/2))
        detected = rng.random(nobs) < pd
        accepted = detected & (rng.random(nobs) < p_quality)
        for event, values, theoretical in [
                ("detected", detected, pd),
                ("accepted", accepted, pd*p_quality)]:
            k = int(values.sum())
            lo, hi = _interval(k, nobs)
            observation_rows.append(dict(
                sensing_duration=duration, event=event, independent_scenes=nobs,
                successes=k, empirical_probability=k/nobs,
                model_probability=theoretical, ci95_low=lo, ci95_high=hi))
    _csv(out / "coverage_by_state.csv", coverage_rows)
    _csv(out / "scene_coverage.csv", scene_rows)
    _csv(out / "calibration_bounds.csv", radius_rows)
    _csv(out / "test_coverage_samples.csv", raw_rows)
    _csv(out / "capture_acceptance.csv", observation_rows)
    np.savez_compressed(
        out / "calibration_dataset.npz", true_complete_channels=truth,
        observed_group_bases=bases, required_radius=required_radius,
        dl_active=dl_active, ul_active=ul_active, ages=ages,
        sensing_durations=durations, groups=np.asarray(groups),
        scene_split=np.asarray(["train"]*ntrain+["calibration"]*ncal+["test"]*ntest),
        coefficient_bound=coefficient_bound, known_rho=known_rho,
        calibrated_rho=calibrated_rho, calibration_scores=scores,
        calibration_order=order, calibration_alpha=alpha)
    _latex(out / "scene_coverage.tex",
           ["Envelope", "Scenes", "Failure", "95% CI"],
           [[r["method"], r["independent_test_scenes"],
             f'{r["scene_failure_rate"]:.3f}',
             f'[{r["ci95_low"]:.3f}, {r["ci95_high"]:.3f}]']
            for r in scene_rows])
    plt = _plot_import()
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.3))
    for event, label in [("detected", "UL detected"), ("accepted", "Record accepted")]:
        rr = [r for r in observation_rows if r["event"] == event]
        y = np.array([r["empirical_probability"] for r in rr])
        yerr = np.array([[y[i]-r["ci95_low"] for i, r in enumerate(rr)],
                        [r["ci95_high"]-y[i] for i, r in enumerate(rr)]])
        axes[0, 0].errorbar(durations, y, yerr=yerr, marker="o", capsize=3, label=label)
    axes[0, 0].set(xlabel="Sensing duration (model units)", ylabel="Probability",
                   ylim=(0, 1.05), title="Declared observation kernel")
    axes[0, 0].legend()
    for g in (0, 1, 3):
        label = groups[g].replace("_", " ")
        axes[0, 1].plot(ages, known_rho[g, 1], "--", label=label+" finite bound")
        axes[0, 1].plot(ages, calibrated_rho[g, 1], "o-", label=label+" calibrated")
    axes[0, 1].set(xlabel="Record age (model units)", ylabel="Complete-channel radius",
                   title="Duration = 2; background has rank zero")
    axes[0, 1].legend(fontsize=7)
    for g, group in enumerate(groups):
        rr = [r for r in coverage_rows if r["method"] == "split_calibrated"
              and r["group"] == group and r["sensing_duration"] == 2]
        axes[1, 0].plot(ages, [r["active_failure_rate"] for r in rr], "o-",
                        label=group.replace("_", " "))
    axes[1, 0].set(xlabel="Record age (model units)",
                   ylabel="DL-active test failure fraction",
                   title="DL-active failures by stratum")
    axes[1, 0].legend(fontsize=7)
    y = np.array([r["scene_failure_rate"] for r in scene_rows])
    err = np.array([[y[i]-r["ci95_low"] for i, r in enumerate(scene_rows)],
                    [r["ci95_high"]-y[i] for i, r in enumerate(scene_rows)]])
    axes[1, 1].bar(np.arange(3), y, yerr=err, capsize=4)
    axes[1, 1].set(xticks=np.arange(3), xticklabels=["Finite bound", "Calibrated", "No background"],
                   ylabel="Scene-wise failure fraction", ylim=(0, 1.08),
                   title="Scene failures with 95% binomial intervals")
    axes[1, 1].axhline(alpha, ls=":", color="black", label="Calibration alpha")
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("E4 — synthetic complete-channel calibration", fontsize=13)
    fig.tight_layout()
    _save(fig, out / "e4_calibration")
    plt.close(fig)
    model = dict(
        experiment="E4", data_source="declared_synthetic_complete_channel_model",
        seed=int(seed), quick=bool(quick), profile="quick" if quick else "full",
        m=m, rank=rank,
        coefficient_bound=coefficient_bound, ntrain=ntrain, ncal=ncal, ntest=ntest,
        calibration_alpha=alpha, calibration_order=order, calibration_scale=scale,
        calibration_target_coverage=1-alpha,
        calibration_unit="independent whole scene; maximum over all modeled strata",
        empirical_guarantee="exchangeable-scene marginal simultaneous coverage only",
        conditional_risk_certified=False,
        physical_receiver_inr_certified=False,
        normalization="f=sqrt(chi*Pmax/N) g; chi, antenna gain and noise are absorbed into a declared synthetic prior",
        limitations=[
            "Not a physical UL-to-DL calibration, GLRT fit, or measured antenna/noise bound.",
            "The finite synthetic envelope is valid by construction; empirical radii are not conditional LP risk bounds.",
            "Coefficients cover each complete coherent channel jointly; individual peaks are not separate receivers.",
            "UL-silent and missed receivers remain in evaluation through a full-channel background ball.",
        ], scene_coverage=scene_rows)
    _json(out / "manifest.json", model)
    (out / "captions.md").write_text(
        "# E4 synthetic calibration\n\n"
        "The full coherent normalized DL channel is generated inside a declared finite "
        "coefficient/residual envelope. Entire independent scenes form training, calibration "
        "and test splits. The empirical envelope uses one maximum score per calibration "
        "scene across all displayed conditions. Its coverage target is marginal over scenes; "
        "the state- and DL-activity-conditioned failure fractions are diagnostics, not "
        "conditional risk certificates for an adaptive controller. Error bars count independent "
        "scenes. Zero observed failures does not mean zero unknown physical risk. Background "
        "sets retain receivers with missed or silent uplinks. The capture/acceptance panel "
        "uses a declared toy observation kernel, not a GLRT calibration.\n")
    return dict(output_dir=str(out), manifest=str(out / "manifest.json"),
                scene_coverage=scene_rows, calibration_scale=scale,
                calibration_alpha=alpha, calibration_order=order,
                known_rho=known_rho, calibrated_rho=calibrated_rho,
                ages=ages, durations=durations)


def _sample_set(rng, s, count):
    m, k = s.A.shape
    c = _rng_complex(rng, (count, k))
    if k:
        c /= np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-300)
        c *= s.C * rng.random((count, 1))**(1/max(1, 2*k))
    e = _rng_complex(rng, (count, m))
    e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-300)
    e *= s.rho * rng.random((count, 1))**(1/(2*m))
    return c @ s.A.T + e


def run_e5(output_dir, seed=5105, quick=True):
    """Check support attainability, SOCP feasibility, phase and power recovery."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    tolerance = 2e-6
    identity_rows = []
    cases = ["multipath", "single_path", "rank_deficient", "full_rank",
             "zero_projection", "zero_residual", "zero_C", "zero_v", "empty_A"]
    nrep = 8 if quick else 48
    for case in cases:
        for trial in range(nrep):
            m = 8
            rank = 1 if case == "single_path" else (m if case == "full_rank" else 3)
            a = _rng_complex(rng, (m, rank)) / np.sqrt(m)
            if case == "rank_deficient":
                a[:, -1] = a[:, 0]
            if case == "empty_A":
                a = np.empty((m, 0), complex)
            c, rho = (0.0 if case == "zero_C" else 1.1), (.0 if case == "zero_residual" else .17)
            v = _unit(_rng_complex(rng, m))
            if case == "zero_projection":
                q, _ = np.linalg.qr(a, mode="complete")
                v = q[:, rank]
            if case == "zero_v":
                v *= 0
            s = ProtectionSet(a, c, rho, case)
            bound = support(v, s)
            adversary = worst_case_channel(v, s)
            achieved = abs(np.vdot(adversary, v))
            sampled = _sample_set(rng, s, 128 if quick else 1024)
            random_max = float(np.max(np.abs(sampled.conj() @ v)))
            identity_rows.append(dict(
                case=case, trial=trial, bound_amplitude=bound,
                constructed_amplitude=achieved,
                absolute_identity_error=abs(bound-achieved),
                random_max_amplitude=random_max,
                random_bound_violation=max(0., random_max-bound)))
    max_error = max(r["absolute_identity_error"] for r in identity_rows)
    if max_error > tolerance:
        raise AssertionError(f"Proposition 1 support mismatch: {max_error}")
    # Exact-zero budgets expose a useful edge case: with rho=0, nonzero
    # nullspace transmission remains possible; any positive isotropic rho
    # forces mute. These have analytic TN amplitudes.
    h_edge = np.asarray([1., 1.], complex) / np.sqrt(2.)
    a_edge = np.asarray([[1.], [0.]], complex)
    zero_budget_rows = []
    for label, rho_edge, expected in [
            ("zero_budget_nullspace", 0., 1./np.sqrt(2.)),
            ("zero_budget_isotropic", .1, 0.)]:
        edge = solve_robust_beam(
            h_edge, [ProtectionSet(a_edge, 1., rho_edge, label)], 0.)
        error = abs(edge.amplitude-expected)
        if error > tolerance or edge.max_violation > tolerance:
            raise AssertionError(f"Zero-budget edge case failed: {label}")
        zero_budget_rows.append(dict(
            case=label, expected_amplitude=expected,
            socp_amplitude=edge.amplitude, absolute_error=error,
            power_fraction=edge.power_fraction,
            max_constraint_residual=edge.max_violation))
    _csv(out / "zero_budget_checks.csv", zero_budget_rows)
    q, _ = np.linalg.qr(_rng_complex(rng, (8, 8)))
    a = q[:, :3]
    h = _unit(np.sqrt(.8) * a[:, 0] + np.sqrt(.2) * q[:, 3])
    gammas_db = np.asarray([-20., -15., -10., -6., -3., 0., 3.])
    rhos = np.asarray([0., .1, .3, .6])
    rows, phase_rows = [], []
    snr_reference = 10.0
    nominal = _unit(h-a @ (a.conj().T @ h))
    covariance = np.outer(h, h.conj()) - a @ a.conj().T
    _, eigenvectors = np.linalg.eigh(covariance)
    penalty = eigenvectors[:, -1]
    for rho in rhos:
        s = ProtectionSet(a, 1.2, float(rho), "joint_complete_channel")
        samples = _sample_set(rng, s, 256 if quick else 4096)
        actual_f = samples[0]
        oracle_set = ProtectionSet(actual_f[:, None], 1., 0., "oracle_actual_channel")
        for gamma_db in gammas_db:
            gamma = 10.**(gamma_db/10.)
            robust = solve_robust_beam(h, [s], gamma)
            oracle = solve_robust_beam(h, [oracle_set], gamma)
            methods = [("robust_SOCP", robust.v, robust),
                       ("unprotected", h, None),
                       ("nominal_null", nominal, None),
                       ("penalty_lambda_1", penalty, None),
                       ("true_DL_oracle", oracle.v, oracle)]
            for method, v, solution in methods:
                worst = support(v, s)**2
                actual_max = float(np.max(np.abs(samples.conj() @ v)**2))
                actual_single = float(abs(np.vdot(actual_f, v))**2)
                amplitude = float(abs(np.vdot(h, v)))
                rows.append(dict(
                    rho=float(rho), gamma_db=float(gamma_db), gamma_linear=gamma,
                    method=method, power_fraction=float(np.vdot(v, v).real),
                    tn_amplitude=amplitude,
                    tn_spectral_efficiency=float(np.log2(1+snr_reference*amplitude**2)),
                    worst_set_inr=worst, sample_max_inr=actual_max,
                    oracle_channel_inr=actual_single,
                    normalized_protection_violation=max(0., worst/gamma-1.),
                    solver_status=solution.status if solution else "not_applicable",
                    solver_residual=solution.max_violation if solution else float("nan"),
                    solver_raw_residual=solution.raw_max_violation if solution else float("nan"),
                    solver_raw_relative_residual=solution.raw_max_relative_violation if solution else float("nan"),
                    feasibility_scale=solution.feasibility_scale if solution else float("nan"),
                    solver_seconds=solution.solve_time if solution else 0.))
                if method == "robust_SOCP" and (worst > gamma*(1+tolerance)
                                               or actual_max > gamma*(1+tolerance)):
                    raise AssertionError("An in-set E5 channel violates the robust INR limit.")
            rotated = robust.v * np.exp(1j*.731)
            phase_rows.append(dict(
                rho=float(rho), gamma_db=float(gamma_db),
                phase_equality_residual=abs(np.vdot(h, robust.v).imag),
                objective_amplitude_gap=abs(robust.amplitude-abs(np.vdot(h, robust.v))),
                phase_rotation_power_error=abs(np.linalg.norm(rotated)**2-robust.power_fraction),
                phase_rotation_support_error=abs(support(rotated, s)-support(robust.v, s)),
                phase_rotation_tn_amplitude_error=abs(abs(np.vdot(h, rotated))-robust.amplitude)))
    # Objective scaling must not change the beam when physical TN amplitudes
    # are tiny. Returned amplitudes must still retain those physical units.
    scale_set = ProtectionSet(a, 1.2, .3, "physical_channel_scaling")
    scale_reference = solve_robust_beam(h, [scale_set], .1)
    scaling_rows = []
    for h_scale in (1e-12, 1e-6, 1., 1e6):
        beam = solve_robust_beam(h_scale*h, [scale_set], .1)
        error = abs(beam.amplitude/h_scale-scale_reference.amplitude)
        if error > tolerance:
            raise AssertionError("SOCP objective depends on physical TN channel units.")
        scaling_rows.append(dict(
            tn_channel_scale=h_scale, returned_physical_amplitude=beam.amplitude,
            normalized_amplitude=beam.amplitude/h_scale,
            reference_amplitude=scale_reference.amplitude,
            normalized_amplitude_error=error,
            power_fraction=beam.power_fraction,
            raw_max_violation=beam.raw_max_violation,
            raw_max_relative_violation=beam.raw_max_relative_violation,
            feasibility_scale=beam.feasibility_scale,
            solver_status=beam.status))
    _csv(out / "physical_channel_scaling.csv", scaling_rows)
    # Geometry sweeps distinguish uncertainty rank from TN/protection alignment.
    # Rank sweep holds h fixed while nesting columns; angle sweep fixes rank=3.
    geometry_rows = []
    h_rank = _unit(np.sqrt(.5)*q[:, 0]
                   + np.sqrt(.5/7)*np.sum(q[:, 1:], axis=1))
    for gamma_db in (-10., -3., 0.):
        for rank_value in (1, 2, 3, 4, 6, 8):
            group = ProtectionSet(q[:, :rank_value], 1.2, .1, "rank_sweep")
            beam = solve_robust_beam(h_rank, [group], 10.**(gamma_db/10))
            geometry_rows.append(dict(
                sweep="rank", gamma_db=gamma_db, rank=rank_value,
                angle_degrees=float("nan"), power_fraction=beam.power_fraction,
                tn_amplitude=beam.amplitude,
                tn_spectral_efficiency=float(np.log2(1+snr_reference*beam.amplitude**2)),
                worst_set_inr=support(beam.v, group)**2,
                solver_residual=beam.max_violation))
        for angle in (0., 15., 30., 45., 60., 75., 90.):
            radians = np.deg2rad(angle)
            h_angle = np.cos(radians)*q[:, 0] + np.sin(radians)*q[:, 3]
            group = ProtectionSet(q[:, :3], 1.2, .1, "angle_sweep")
            beam = solve_robust_beam(h_angle, [group], 10.**(gamma_db/10))
            geometry_rows.append(dict(
                sweep="angle", gamma_db=gamma_db, rank=3,
                angle_degrees=angle, power_fraction=beam.power_fraction,
                tn_amplitude=beam.amplitude,
                tn_spectral_efficiency=float(np.log2(1+snr_reference*beam.amplitude**2)),
                worst_set_inr=support(beam.v, group)**2,
                solver_residual=beam.max_violation))
    _csv(out / "geometry_sweeps.csv", geometry_rows)
    _csv(out / "support_identity.csv", identity_rows)
    _csv(out / "beam_sweeps.csv", rows)
    _csv(out / "phase_invariance.csv", phase_rows)
    summary_rows = []
    for case in cases:
        rr = [r for r in identity_rows if r["case"] == case]
        summary_rows.append(dict(
            case=case, realizations=len(rr),
            maximum_identity_error=max(r["absolute_identity_error"] for r in rr),
            maximum_random_violation=max(r["random_bound_violation"] for r in rr)))
    _csv(out / "identity_summary.csv", summary_rows)
    _latex(out / "identity_summary.tex", ["Case", "Trials", "Max. identity error"],
           [[r["case"], r["realizations"], f'{r["maximum_identity_error"]:.2e}']
            for r in summary_rows])
    selected = [r for r in rows if r["rho"] == .3 and r["gamma_db"] in (-10., -3., 0.)]
    _latex(out / "beam_comparison.tex",
           ["Gamma (dB)", "Method", "Power", "TN bit/s/Hz", "Worst INR"],
           [[int(r["gamma_db"]), r["method"], f'{r["power_fraction"]:.3f}',
             f'{r["tn_spectral_efficiency"]:.3f}', f'{r["worst_set_inr"]:.3f}']
            for r in selected])
    plt = _plot_import()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for rho in rhos:
        rr = [r for r in rows if r["method"] == "robust_SOCP" and r["rho"] == rho]
        x = [r["gamma_db"] for r in rr]
        axes[0].plot(x, [r["tn_spectral_efficiency"] for r in rr], "o-", label=f"rho={rho:g}")
        axes[1].plot(x, [r["power_fraction"] for r in rr], "o-", label=f"rho={rho:g}")
        axes[2].plot(x, [r["worst_set_inr"]/r["gamma_linear"] for r in rr], "o-", label=f"rho={rho:g}")
    axes[0].set(ylabel="TN spectral efficiency (bit/s/Hz)", xlabel="INR limit Gamma (dB)")
    axes[1].set(ylabel="Transmit-power fraction", xlabel="INR limit Gamma (dB)", ylim=(0, 1.06))
    axes[2].set(ylabel="Worst-case INR / Gamma", xlabel="INR limit Gamma (dB)", ylim=(0, 1.06))
    axes[2].axhline(1, color="black", ls=":")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle("E5 — exact robust SOCP, declared synthetic channel sets")
    fig.tight_layout()
    _save(fig, out / "e5_gamma_rho_sweep")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for method in ["robust_SOCP", "unprotected", "nominal_null", "penalty_lambda_1", "true_DL_oracle"]:
        rr = [r for r in rows if r["method"] == method and r["rho"] == .3]
        x = [r["gamma_db"] for r in rr]
        axes[0].plot(x, [r["tn_spectral_efficiency"] for r in rr], "o-", label=method)
        axes[1].plot(x, [10*np.log10(max(r["worst_set_inr"],1e-16)) for r in rr], "o-", label=method)
    axes[1].plot(gammas_db, gammas_db, "k:", label="INR limit")
    axes[0].set(xlabel="INR limit Gamma (dB)", ylabel="TN spectral efficiency (bit/s/Hz)")
    axes[1].set(xlabel="INR limit Gamma (dB)", ylabel="Worst INR over full set (dB)")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.suptitle("E5 — beamforming references at rho = 0.3")
    fig.tight_layout()
    _save(fig, out / "e5_method_comparison")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for gamma_db in (-10., -3., 0.):
        ranks = [r for r in geometry_rows if r["sweep"] == "rank"
                 and r["gamma_db"] == gamma_db]
        angles = [r for r in geometry_rows if r["sweep"] == "angle"
                  and r["gamma_db"] == gamma_db]
        axes[0].plot([r["rank"] for r in ranks],
                     [r["tn_spectral_efficiency"] for r in ranks], "o-",
                     label=f"Gamma={gamma_db:g} dB")
        axes[1].plot([r["angle_degrees"] for r in angles],
                     [r["tn_spectral_efficiency"] for r in angles], "o-",
                     label=f"Gamma={gamma_db:g} dB")
    axes[0].set(xlabel="Nested uncertainty-subspace rank (fixed TN channel)",
                ylabel="TN spectral efficiency (bit/s/Hz)")
    axes[1].set(xlabel="TN angle from protection subspace (degrees)",
                ylabel="TN spectral efficiency (bit/s/Hz)")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle("E5 — geometry sensitivity, rho = 0.1")
    fig.tight_layout()
    _save(fig, out / "e5_geometry_sensitivity")
    plt.close(fig)
    diagnostics = dict(
        support_identity_max_error=float(max_error),
        physical_channel_scale_max_amplitude_error=max(r["normalized_amplitude_error"] for r in scaling_rows),
        robust_max_raw_constraint_residual=max(r["solver_raw_residual"] for r in rows if r["method"] == "robust_SOCP"),
        robust_max_raw_relative_residual=max(r["solver_raw_relative_residual"] for r in rows if r["method"] == "robust_SOCP"),
        robust_max_feasibility_correction=max(1-r["feasibility_scale"] for r in rows if r["method"] == "robust_SOCP"),
        zero_budget_max_amplitude_error=max(r["absolute_error"] for r in zero_budget_rows),
        geometry_max_constraint_residual=max(r["solver_residual"] for r in geometry_rows),
        robust_max_constraint_residual=max(r["solver_residual"] for r in rows if r["method"] == "robust_SOCP"),
        robust_max_normalized_protection_violation=max(r["normalized_protection_violation"] for r in rows if r["method"] == "robust_SOCP"),
        maximum_phase_invariance_error=max(
            max(r[k] for k in r if k not in ("rho", "gamma_db")) for r in phase_rows))
    if max(diagnostics.values()) > tolerance:
        raise AssertionError(f"E5 tolerance {tolerance} exceeded: {diagnostics}")
    manifest = dict(
        experiment="E5", seed=int(seed), quick=bool(quick),
        data_source="declared_synthetic_complete_channel_model",
        normalized_reference_tn_snr_linear=snr_reference,
        tolerance=tolerance, diagnostics=diagnostics,
        oracle_scope="protects its one known true DL channel, not every channel in the uncertainty set",
        solver_statuses=sorted({r["solver_status"] for r in rows if r["solver_status"] != "not_applicable"}),
        physical_receiver_inr_certified=False)
    _json(out / "manifest.json", manifest)
    np.savez_compressed(out / "beam_geometry.npz", A=a, h=h, gammas_db=gammas_db, rhos=rhos)
    (out / "captions.md").write_text(
        "# E5 robust spatial optimization\n\n"
        "Constructed adversarial complete channels attain the analytic support function, "
        "including zero-vector, zero-projection, single-path, full-rank and rank-deficient "
        "cases. Random in-set channels provide a secondary implementation check. SOCP "
        "curves preserve the transmit-power inequality and phase normalization. Tight "
        "budgets can require power backoff even after nominal directional nulling. "
        "The reference TN SNR is 10 linear. The oracle protects one true DL channel, "
        "whereas the robust solution protects the complete declared uncertainty set; "
        "unprotected, nominal-null and lambda=1 beams are diagnostic references without "
        "that guarantee. Every channel and envelope here is synthetic. These numerical "
        "checks do not replace Proposition 1's proof or physical E4 calibration.\n")
    return dict(output_dir=str(out), manifest=str(out / "manifest.json"), diagnostics=diagnostics)
