"""Complex robust beamforming in the noise-normalized Appendix D model.

A protection set contains COMPLETE coherent effective channels:
    G = {A c + e : ||c||_2 <= C, ||e||_2 <= rho}.
The caller supplies valid bounds; this module does not infer NTN antenna gains
or a receiver-noise lower bound from passive uplink observations.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable
import numpy as np


@dataclass(frozen=True)
class ProtectionSet:
    A: np.ndarray
    C: float
    rho: float
    name: str = ""

    def __post_init__(self):
        a = np.asarray(self.A, dtype=complex)
        if a.ndim != 2:
            raise ValueError("A must be an M-by-K complex matrix (K may be zero).")
        if not np.all(np.isfinite(a)):
            raise ValueError("A must contain finite values.")
        if not np.isfinite(self.C) or not np.isfinite(self.rho):
            raise ValueError("C and rho must be finite.")
        if self.C < 0 or self.rho < 0:
            raise ValueError("C and rho must be nonnegative.")
        object.__setattr__(self, "A", a)
        object.__setattr__(self, "C", float(self.C))
        object.__setattr__(self, "rho", float(self.rho))


@dataclass
class BeamSolution:
    v: np.ndarray
    amplitude: float
    power_fraction: float
    status: str
    max_violation: float
    solver: str = ""
    solve_time: float = 0.0
    raw_max_violation: float = 0.0
    feasibility_scale: float = 1.0
    raw_max_relative_violation: float = 0.0


def _vector(v, n=None):
    v = np.asarray(v, dtype=complex).reshape(-1)
    if n is not None and v.size != n:
        raise ValueError(f"Beam/channel dimension {v.size} does not equal {n}.")
    if not np.all(np.isfinite(v)):
        raise ValueError("Beam/channel must contain finite values.")
    return v


def support(v: np.ndarray, protection_set: ProtectionSet) -> float:
    """Return sup_{f in G} |f^H v|, an amplitude, not INR."""
    s = protection_set
    v = _vector(v, s.A.shape[0])
    return float(s.C * np.linalg.norm(s.A.conj().T @ v)
                 + s.rho * np.linalg.norm(v))


def worst_case_channel(v: np.ndarray, protection_set: ProtectionSet) -> np.ndarray:
    """Construct an in-set full channel attaining support(v, protection_set)."""
    s = protection_set
    v = _vector(v, s.A.shape[0])
    u = s.A.conj().T @ v
    c = np.zeros(s.A.shape[1], dtype=complex)
    if np.linalg.norm(u) > 0:
        c = s.C * u / np.linalg.norm(u)
    e = np.zeros_like(v)
    if np.linalg.norm(v) > 0:
        e = s.rho * v / np.linalg.norm(v)
    return s.A @ c + e


def solve_robust_beam(
    h: np.ndarray,
    sets: Iterable[ProtectionSet],
    gamma_linear,
    solver: str | None = None,
) -> BeamSolution:
    """Maximize Re(h^H v) with phase, norm, and robust amplitude constraints.

    h is the known effective TN channel after its fixed receive combiner.
    v = sqrt(p) w uses power FRACTION p, with ||v|| <= 1. Each NTN channel
    in sets must already include sqrt(chi*Pmax/N). Gamma is linear INR;
    a scalar applies to every set, or one value per set may be supplied.

    Tiny numerical violations are removed by common radial scaling, which
    preserves the phase equality and all homogeneous protection constraints.
    A solver exception is propagated; it is never silently replaced by mute.
    The objective and phase equality use h/||h|| to avoid loss of numerical
    significance for physical small-amplitude channels; returned amplitude
    remains in the original units. Diagnostics expose the raw primal residual
    before phase/scaling correction and the applied feasibility scale. A raw
    violation or necessary relative correction above 1e-5 raises an error.
    """
    h = _vector(h)
    if h.size == 0:
        raise ValueError("The TN channel must have at least one antenna.")
    h_scale = float(np.max(np.abs(h)))
    h_norm = h_scale * np.linalg.norm(h/h_scale) if h_scale else 0.0
    if not np.isfinite(h_norm):
        raise ValueError("The TN channel norm overflows; rescale its physical units.")
    h_unit = h/h_norm if h_norm else np.zeros_like(h)
    groups = list(sets)
    if any(s.A.shape[0] != h.size for s in groups):
        raise ValueError("Every protection set must have the TN beam dimension.")
    gamma = np.asarray(gamma_linear, dtype=float)
    if gamma.ndim == 0:
        gamma = np.full(len(groups), float(gamma))
    elif gamma.ndim != 1 or gamma.size != len(groups):
        raise ValueError("Gamma must be scalar or have one value per set.")
    if np.any(~np.isfinite(gamma)) or np.any(gamma < 0):
        raise ValueError("INR budgets must be finite and nonnegative.")
    # A zero INR budget is an exact nullspace constraint when rho=0.
    # Solve in that nullspace: naive radial scaling by sqrt(0)/roundoff would
    # incorrectly erase useful TN transmission along an exact protected null.
    zero_indices = np.flatnonzero(gamma == 0)
    if len(zero_indices):
        if any(groups[i].rho > 0 for i in zero_indices):
            return BeamSolution(np.zeros_like(h), 0.0, 0.0, "optimal", 0.0,
                                "analytic_zero_budget")
        matrices = [groups[i].A.conj().T for i in zero_indices
                    if groups[i].C > 0 and groups[i].A.shape[1] > 0]
        basis = np.eye(h.size, dtype=complex)
        if matrices:
            linear = np.vstack(matrices)
            _, singular, vh = np.linalg.svd(linear, full_matrices=True)
            cutoff = (max(linear.shape) * np.finfo(float).eps *
                      (singular[0] if singular.size else 0.))
            rank = int(np.sum(singular > cutoff))
            basis = vh.conj().T[:, rank:]
        if basis.shape[1] == 0:
            return BeamSolution(np.zeros_like(h), 0.0, 0.0, "optimal", 0.0,
                                "analytic_zero_budget")
        positive = np.flatnonzero(gamma > 0)
        reduced_sets = [ProtectionSet(basis.conj().T @ groups[i].A,
                                      groups[i].C, groups[i].rho, groups[i].name)
                        for i in positive]
        reduced = solve_robust_beam(basis.conj().T @ h, reduced_sets,
                                    gamma[positive], solver=solver)
        out = basis @ reduced.v
        violations = [0.0, np.linalg.norm(out)-1.,
                      abs(np.vdot(h_unit, out).imag)]
        violations.extend(support(out, s)-np.sqrt(g)
                          for s, g in zip(groups, gamma))
        if max(violations) > 1e-5:
            raise RuntimeError(
                "Zero-budget nullspace reconstruction is materially infeasible: "
                f"residual={max(violations):.3g}.")
        return BeamSolution(
            out, float(np.vdot(h, out).real), float(np.vdot(out, out).real),
            reduced.status, float(max(violations)), reduced.solver,
            reduced.solve_time,
            max(reduced.raw_max_violation, float(max(violations))),
            reduced.feasibility_scale, reduced.raw_max_relative_violation)
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise ImportError(
            "Appendix D beamforming requires cvxpy and clarabel. In the "
            "notebook kernel environment run: python -m pip install cvxpy clarabel"
        ) from exc
    installed = cp.installed_solvers()
    if solver is None:
        solver = next((s for s in ("CLARABEL", "ECOS", "SCS") if s in installed), None)
    if solver is None or solver not in installed:
        raise RuntimeError(f"No requested conic solver available; installed={installed}.")
    if h_norm == 0:
        return BeamSolution(np.zeros_like(h), 0.0, 0.0, "optimal", 0.0, solver)
    v = cp.Variable(h.size, complex=True)
    inner = h_unit.conj() @ v
    constraints = [cp.norm(v, 2) <= 1, cp.imag(inner) == 0]
    for s, g in zip(groups, gamma):
        terms = []
        if s.C > 0 and s.A.shape[1] > 0:
            terms.append(s.C * cp.norm(s.A.conj().T @ v, 2))
        if s.rho > 0:
            terms.append(s.rho * cp.norm(v, 2))
        if terms:
            constraints.append(sum(terms) <= np.sqrt(g))
    problem = cp.Problem(cp.Maximize(cp.real(inner)), constraints)
    options = {}
    if solver == "CLARABEL":
        options = dict(tol_gap_abs=1e-8, tol_gap_rel=1e-8, tol_feas=1e-8,
                       max_iter=300)
    elif solver == "SCS":
        options = dict(eps=1e-7, max_iters=50000)
    start = perf_counter()
    problem.solve(solver=solver, **options)
    elapsed = perf_counter() - start
    if problem.status not in ("optimal", "optimal_inaccurate") or v.value is None:
        raise RuntimeError(f"Robust SOCP failed: status={problem.status}.")
    out = _vector(v.value, h.size).copy()
    # Report/reject a malformed primal solve BEFORE any feasibility repair.
    # Relative checks prevent a small absolute error at a very tight INR
    # budget from silently causing a large beam/power correction.
    raw_violations = [0.0, np.linalg.norm(out)-1.0,
                      abs(np.vdot(h_unit, out).imag)]
    raw_relative = list(raw_violations)
    for s, g in zip(groups, gamma):
        excess = support(out, s)-np.sqrt(g)
        raw_violations.append(excess)
        raw_relative.append(excess/np.sqrt(g))  # g>0 after zero-budget branch
    raw_max_violation = float(max(raw_violations))
    raw_max_relative_violation = float(max(raw_relative))
    acceptance_tolerance = 1e-5
    if max(raw_max_violation, raw_max_relative_violation) > acceptance_tolerance:
        raise RuntimeError(
            "Robust SOCP returned a materially infeasible primal solution: "
            f"status={problem.status}, raw_absolute={raw_max_violation:.3g}, "
            f"raw_relative={raw_max_relative_violation:.3g}; "
            f"allowed={acceptance_tolerance:g}.")
    phase = np.vdot(h_unit, out)
    if abs(phase) > 0:
        out *= np.exp(-1j * np.angle(phase))
    scale = min(1.0, 1.0 / max(np.linalg.norm(out), 1e-300))
    for s, g in zip(groups, gamma):
        psi = support(out, s)
        if psi > 0:
            scale = min(scale, np.sqrt(g) / psi)
    if 1.0-scale > acceptance_tolerance:
        raise RuntimeError(
            f"Robust SOCP requires excessive feasibility correction: scale={scale:.9g}.")
    feasibility_scale = float(scale*(1.0-1e-10) if scale < 1.0 else 1.0)
    out *= feasibility_scale
    violations = [0.0, np.linalg.norm(out) - 1.0,
                  abs(np.vdot(h_unit, out).imag)]
    violations.extend(support(out, s) - np.sqrt(g)
                      for s, g in zip(groups, gamma))
    return BeamSolution(
        v=out,
        amplitude=float(np.vdot(h, out).real),
        power_fraction=float(np.vdot(out, out).real),
        status=str(problem.status),
        max_violation=float(max(violations)),
        solver=solver,
        solve_time=elapsed,
        raw_max_violation=raw_max_violation,
        feasibility_scale=feasibility_scale,
        raw_max_relative_violation=raw_max_relative_violation,
    )
