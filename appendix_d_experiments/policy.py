"""Exact finite-history occupation-measure LP used by Appendix D experiments.

The caller supplies the complete, acyclic observable-history graph, transition
probabilities, and additive expected action costs. Hidden physical variables may
be integrated into transitions/costs, but may not be used to select an action.
This module does not learn a transition law or certify one against real data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
import math
import time
from typing import Mapping

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


@dataclass
class Action:
    name: str
    reward: float
    transitions: list[tuple[str, float]]
    costs: dict[str, float] = field(default_factory=dict)


@dataclass
class Graph:
    actions: dict[str, list[Action]]
    initial: dict[str, float]
    terminals: set[str]


@dataclass
class PolicyEvaluation:
    objective: float
    costs: dict[str, float]
    occupancy: dict[tuple[str, str], float]
    terminal_mass: dict[str, float]
    state_mass: dict[str, float]


@dataclass
class PolicyResult:
    status: str
    feasible: bool
    objective: float | None
    occupancy: dict[tuple[str, str], float]
    policy: dict[str, dict[str, float]]
    residuals: dict[str, float]
    costs: dict[str, float] = field(default_factory=dict)
    message: str = ""
    solve_time_s: float = 0.0
    n_variables: int = 0


@dataclass
class MixtureResult:
    status: str
    feasible: bool
    objective: float | None
    weights: np.ndarray
    policies: list[dict[str, dict[str, float]]]
    evaluations: list[PolicyEvaluation]
    costs: dict[str, float]
    residuals: dict[str, float]


def validate_graph(graph: Graph) -> list[str]:
    """Validate a finite probability DAG and return its topological order.

    Full-history sufficiency is a modeling obligation of the caller: the DAG
    check cannot detect an invalid merge of two physically distinct histories.
    """
    states = set(graph.actions) | set(graph.terminals)
    if not states or not graph.initial:
        raise ValueError("A graph needs states and an initial distribution.")
    if set(graph.initial) - states:
        raise ValueError("Initial distribution contains an unknown state.")
    if any(not np.isfinite(p) or p < 0 for p in graph.initial.values()):
        raise ValueError("Initial probabilities must be finite and nonnegative.")
    if not math.isclose(sum(graph.initial.values()), 1.0, abs_tol=1e-10):
        raise ValueError("Initial probabilities must sum to one.")
    indegree = dict.fromkeys(states, 0)
    adjacency: dict[str, set[str]] = {s: set() for s in states}
    for state in states:
        actions = graph.actions.get(state, [])
        if state in graph.terminals:
            if actions:
                raise ValueError(f"Terminal state {state!r} cannot have actions.")
            continue
        if not actions:
            raise ValueError(f"Nonterminal state {state!r} has no actions.")
        if len({a.name for a in actions}) != len(actions):
            raise ValueError(f"Duplicate action names at {state!r}.")
        for action in actions:
            if not np.isfinite(action.reward) or any(
                not np.isfinite(value) for value in action.costs.values()
            ):
                raise ValueError("Rewards and costs must be finite.")
            if not action.transitions:
                raise ValueError("Every action must lead to an explicit next/terminal state.")
            if any(p < 0 or not np.isfinite(p) for _, p in action.transitions):
                raise ValueError("Transition probabilities must be finite and nonnegative.")
            if not math.isclose(sum(p for _, p in action.transitions), 1.0, abs_tol=1e-10):
                raise ValueError(f"Transition probabilities do not sum to one at {state!r}.")
            for next_state, probability in action.transitions:
                if next_state not in states:
                    raise ValueError(f"Unknown next state {next_state!r}.")
                if probability > 0 and next_state not in adjacency[state]:
                    adjacency[state].add(next_state)
                    indegree[next_state] += 1
    ready = sorted(state for state, degree in indegree.items() if degree == 0)
    order: list[str] = []
    while ready:
        state = ready.pop()
        order.append(state)
        for next_state in sorted(adjacency[state]):
            indegree[next_state] -= 1
            if indegree[next_state] == 0:
                ready.append(next_state)
    if len(order) != len(states):
        raise ValueError("Finite-history graph must be acyclic.")
    return order


def evaluate_policy(
    graph: Graph,
    policy: Mapping[str, Mapping[str, float]],
    *,
    tolerance: float = 1e-9,
) -> PolicyEvaluation:
    """Integrate a behavioral randomized policy exactly on the supplied DAG.

    At an unreachable state a missing policy is harmless. A missing action
    distribution at any positive-probability state is an error, preventing
    accidental replacement of randomization by an arbitrary action.
    """
    order = validate_graph(graph)
    masses = dict.fromkeys(order, 0.0)
    for state, probability in graph.initial.items():
        masses[state] += probability
    costs = {key: 0.0 for acts in graph.actions.values() for a in acts for key in a.costs}
    occupancy: dict[tuple[str, str], float] = {}
    reward = 0.0
    for state in order:
        if state in graph.terminals:
            continue
        actions = graph.actions[state]
        distribution = policy.get(state)
        if distribution is None:
            if masses[state] > tolerance:
                raise ValueError(f"Missing policy at reachable state {state!r}.")
            distribution = {actions[0].name: 1.0}
        if set(distribution) - {a.name for a in actions}:
            raise ValueError(f"Policy uses an unavailable action at {state!r}.")
        if any(not np.isfinite(p) or p < 0 for p in distribution.values()):
            raise ValueError("Policy probabilities must be finite and nonnegative.")
        if not math.isclose(sum(distribution.values()), 1.0, abs_tol=tolerance):
            raise ValueError(f"Policy probabilities do not sum to one at {state!r}.")
        for action in actions:
            mass = masses[state] * distribution.get(action.name, 0.0)
            occupancy[state, action.name] = mass
            reward += mass * action.reward
            for key, value in action.costs.items():
                costs[key] += mass * value
            for next_state, probability in action.transitions:
                masses[next_state] += mass * probability
    terminal_mass = {s: masses[s] for s in graph.terminals}
    if not math.isclose(sum(terminal_mass.values()), 1.0, abs_tol=10 * tolerance):
        raise RuntimeError("Policy evaluation lost probability mass.")
    return PolicyEvaluation(reward, costs, occupancy, terminal_mass, masses)


def solve_occupancy(
    graph: Graph,
    bounds: Mapping[str, float],
    *,
    tolerance: float = 1e-7,
) -> PolicyResult:
    """Maximize expected reward subject to additive expected-cost bounds.

    A chance event must be encoded as an expected indicator cost; constraining
    expected delivered bits does not implement a deadline chance constraint.
    An activity-conditioned budget uses costs c - delta*d and bound zero,
    including d on sensing/mute actions even when their controlled-BS c is zero.
    """
    started = time.perf_counter()
    order = validate_graph(graph)
    if any(not np.isfinite(b) for b in bounds.values()):
        raise ValueError("Cost bounds must be finite.")
    states = [s for s in order if s not in graph.terminals]
    indices = {s: i for i, s in enumerate(states)}
    entries = [(s, a) for s in states for a in graph.actions[s]]
    keys = list(bounds)
    # All-terminal graphs have no decision variables and need no solver call.
    if not entries:
        feasible = all(0 <= bound + tolerance for bound in bounds.values())
        return PolicyResult(
            "optimal" if feasible else "infeasible", feasible, 0.0 if feasible else None,
            {}, {}, {"flow_max_abs": 0.0, "budget_max_violation": max([0.0] + [-b for b in bounds.values()])},
            {}, "Graph has no decision states.", time.perf_counter() - started, 0,
        )
    row, col, values = [], [], []
    for j, (state, action) in enumerate(entries):
        row.append(indices[state]); col.append(j); values.append(1.0)
        for next_state, probability in action.transitions:
            if next_state in indices and probability != 0:
                row.append(indices[next_state]); col.append(j); values.append(-probability)
    flow = coo_matrix((values, (row, col)), shape=(len(states), len(entries))).tocsr()
    initial = np.array([graph.initial.get(s, 0.0) for s in states])
    reward = np.array([a.reward for _, a in entries], dtype=float)
    cost = np.array([[a.costs.get(key, 0.0) for _, a in entries] for key in keys])
    bound_values = np.array([bounds[key] for key in keys], dtype=float)
    result = linprog(
        -reward, A_ub=cost if keys else None, b_ub=bound_values if keys else None,
        A_eq=flow, b_eq=initial, bounds=(0, None), method="highs",
        options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9},
    )
    elapsed = time.perf_counter() - started
    if not result.success:
        status = {1: "iteration_limit", 2: "infeasible", 3: "unbounded", 4: "solver_error"}.get(result.status, "solver_error")
        return PolicyResult(status, False, None, {}, {}, {}, {}, str(result.message), elapsed, len(entries))
    vector = np.maximum(np.asarray(result.x), 0.0)
    occupancy = {(s, a.name): float(vector[j]) for j, (s, a) in enumerate(entries)}
    policy: dict[str, dict[str, float]] = {}
    for state in states:
        mass = sum(occupancy[state, a.name] for a in graph.actions[state])
        if mass > 0:
            policy[state] = {a.name: occupancy[state, a.name] / mass for a in graph.actions[state]}
        else:
            # The choice at a zero-occupancy state cannot change this policy's law.
            policy[state] = {a.name: float(j == 0) for j, a in enumerate(graph.actions[state])}
    evaluated = evaluate_policy(graph, policy)
    residuals = {
        "flow_max_abs": float(np.max(np.abs(flow @ vector - initial), initial=0)),
        "budget_max_violation": float(np.max(cost @ vector - bound_values, initial=0)) if keys else 0.0,
        "nonnegative_max_violation": float(np.max(-result.x, initial=0)),
        "terminal_mass_abs_error": abs(sum(evaluated.terminal_mass.values()) - 1.0),
        "policy_objective_abs_error": abs(evaluated.objective - float(reward @ vector)),
        "policy_occupancy_max_abs_error": max([0.0] + [abs(evaluated.occupancy[k] - v) for k, v in occupancy.items()]),
    }
    # Also check bounds after reconstructing the executable randomized policy.
    residuals["policy_budget_max_violation"] = max([0.0] + [evaluated.costs.get(k, 0.0) - b for k, b in bounds.items()])
    feasible = max(residuals.values()) <= tolerance
    return PolicyResult(
        "optimal" if feasible else "numerical_failure", feasible,
        evaluated.objective if feasible else None, occupancy, policy, residuals,
        evaluated.costs, str(result.message), elapsed, len(entries),
    )


def enumerate_deterministic_policies(
    graph: Graph,
    *,
    max_policies: int = 100_000,
) -> list[dict[str, dict[str, float]]]:
    """Enumerate complete contingent policies, including unreachable branches.

    This deliberately does not enumerate only root actions or open-loop action
    sequences. It is suitable only for the tiny E6 validation graph.
    """
    order = validate_graph(graph)
    states = [s for s in order if s not in graph.terminals]
    count = math.prod(len(graph.actions[s]) for s in states)
    if count > max_policies:
        raise ValueError(f"Full enumeration needs {count:,} policies; cap is {max_policies:,}.")
    return [
        {s: {a.name: 1.0} for s, a in zip(states, choices)}
        for choices in product(*(graph.actions[s] for s in states))
    ]


def solve_deterministic_mixture(
    graph: Graph,
    bounds: Mapping[str, float],
    *,
    max_policies: int = 100_000,
) -> MixtureResult:
    """Independent LP over mixtures of ALL deterministic contingent policies."""
    policies = enumerate_deterministic_policies(graph, max_policies=max_policies)
    evaluations = [evaluate_policy(graph, policy) for policy in policies]
    keys = list(bounds)
    rewards = np.array([e.objective for e in evaluations])
    costs = np.array([[e.costs.get(k, 0.0) for e in evaluations] for k in keys])
    bound_values = np.array([bounds[k] for k in keys])
    result = linprog(
        -rewards, A_ub=costs if keys else None, b_ub=bound_values if keys else None,
        A_eq=np.ones((1, len(policies))), b_eq=np.ones(1), bounds=(0, None), method="highs",
        options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9},
    )
    if not result.success:
        status = "infeasible" if result.status == 2 else "solver_error"
        return MixtureResult(status, False, None, np.zeros(len(policies)), policies, evaluations, {}, {})
    weights = np.maximum(result.x, 0.0)
    residuals = {
        "normalization_abs_error": abs(float(weights.sum()) - 1),
        "budget_max_violation": float(np.max(costs @ weights - bound_values, initial=0)) if keys else 0.0,
        "nonnegative_max_violation": float(np.max(-result.x, initial=0)),
    }
    all_keys = {k for e in evaluations for k in e.costs}
    mixed_costs = {k: float(sum(w * e.costs.get(k, 0.0) for w, e in zip(weights, evaluations))) for k in all_keys}
    return MixtureResult("optimal", True, float(rewards @ weights), weights, policies, evaluations, mixed_costs, residuals)
