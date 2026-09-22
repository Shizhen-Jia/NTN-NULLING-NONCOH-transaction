"""E6: exact finite-model validation of the Appendix D policy LP.

This is a deliberately tiny declared model, not a ray-traced performance claim.
Its exhaustive reference mixes ALL deterministic contingent policies, including
their decisions after accept/reject/no-detection observations. Comparing with
only the best deterministic policy would incorrectly disallow randomization.
"""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .policy import (
    Action, Graph, evaluate_policy, solve_occupancy,
    solve_deterministic_mixture, enumerate_deterministic_policies,
)


def tiny_graph(delta_ntn: float) -> Graph:
    """Two decision times; full histories are distinct after both actions.

    The initial choice is short or long sensing. The second choice is Serve or
    Mute after observing accept/reject/no_detection. The observation probabilities,
    deterministic TN payloads, and conditional protection-failure bounds are
    specified finite-model inputs. A rejected observation uses a declared older
    set rather than being treated as a successful refresh.

    Every episode lasts three normalized time units and NTN DL is active during
    all three, including sensing and mute. The root action represents an RF gap,
    so controlled-BS NTN interference and TN service are both zero there. These
    simplifications isolate the LP equivalence and chance-constraint semantics.
    """
    horizon = 3.0
    required_bits = 1.2
    specifications = {
        "short": {"duration": 1.0, "probability": (0.55, 0.25, 0.20),
                  "bits": (1.90, 1.25, 0.40), "eta": (0.05, 0.15, 0.30)},
        "long": {"duration": 1.5, "probability": (0.80, 0.10, 0.10),
                 "bits": (1.50, 0.80, 0.20), "eta": (0.02, 0.08, 0.20)},
    }
    observations = ("accept", "reject", "no_detection")
    actions: dict[str, list[Action]] = {"initial": []}
    terminals: set[str] = set()
    for sensing, spec in specifications.items():
        duration = spec["duration"]
        destinations = [(f"sense={sensing}/obs={obs}", p)
                        for obs, p in zip(observations, spec["probability"])]
        actions["initial"].append(Action(
            sensing, 0.0, destinations,
            {"ntn_risk": 0.0, "ntn_active": duration,
             "ntn_excess": -delta_ntn * duration, "rf_gap": duration},
        ))
        service_duration = horizon - duration
        for index, observation in enumerate(observations):
            state = f"sense={sensing}/obs={observation}"
            actions[state] = []
            for service in ("serve", "mute"):
                bits = spec["bits"][index] if service == "serve" else 0.0
                risk = spec["eta"][index] * service_duration if service == "serve" else 0.0
                # The event belongs to the terminal deadline, not the mean payload.
                fail = float(bits < required_bits)
                terminal = f"{state}/action={service}/end"
                terminals.add(terminal)
                actions[state].append(Action(
                    service, bits, [(terminal, 1.0)],
                    {"tn_failure": fail, "ntn_risk": risk,
                     "ntn_active": service_duration,
                     "ntn_excess": risk - delta_ntn * service_duration,
                     "mute_time": service_duration if service == "mute" else 0.0},
                ))
    return Graph(actions, {"initial": 1.0}, terminals)



def validate_beam_elimination() -> dict[str, Any]:
    """Check Prop. 2's dominance logic on declared finite beam alternatives.

    This is not a second continuous SOCP solver: the lower-service beam is an
    explicitly specified finite alternative. Both nonmute choices have exactly
    the same certified risk cost, so pruning is legal under the model's stated
    common-bound assumption. No energy or beam-switching constraint is present.
    """
    reduced = tiny_graph(0.025)
    full = copy.deepcopy(reduced)
    for state, actions in full.actions.items():
        if state == "initial":
            continue
        maximizing = next(action for action in actions if action.name == "serve")
        lower = copy.deepcopy(maximizing)
        lower.name = "serve_lower"
        lower.reward *= 0.7
        lower.costs["tn_failure"] = float(lower.reward < 1.2)
        terminal = state + "/action=serve_lower/end"
        lower.transitions = [(terminal, 1.0)]
        full.terminals.add(terminal)
        actions.append(lower)
    bounds = _bounds(0.30)
    full_optimum = solve_occupancy(full, bounds)
    reduced_optimum = solve_occupancy(reduced, bounds)
    assert full_optimum.feasible and reduced_optimum.feasible
    assert abs(full_optimum.objective - reduced_optimum.objective) < 1e-8
    # Keep each original contingent decision in a virtual policy, but execute
    # the maximum-service beam whenever that policy requests a nonmute beam.
    # Observation histories and certified risk costs are unchanged in this tree.
    complete_policies = enumerate_deterministic_policies(full)
    for policy in complete_policies:
        original = evaluate_policy(full, policy)
        upgraded_policy = {
            state: {("serve" if next(iter(distribution)) == "serve_lower"
                     else next(iter(distribution))): 1.0}
            for state, distribution in policy.items()
        }
        upgraded = evaluate_policy(reduced, upgraded_policy)
        assert upgraded.objective + 1e-10 >= original.objective
        assert upgraded.costs["tn_failure"] <= original.costs["tn_failure"] + 1e-10
        assert abs(upgraded.costs["ntn_excess"] - original.costs["ntn_excess"]) < 1e-10
    return {
        "passed": True, "full_objective": full_optimum.objective,
        "pruned_objective": reduced_optimum.objective,
        "objective_gap": abs(full_optimum.objective - reduced_optimum.objective),
        "complete_full_policies_checked": len(complete_policies),
        "lower_beam_service_fraction": 0.7,
        "common_nonmute_certified_cost": True,
        "scope": "Finite dominance sanity check; actual continuous SOCP is checked by E5.",
    }


def _bounds(delta_tn: float) -> dict[str, float]:
    return {"tn_failure": delta_tn, "ntn_excess": 0.0}


def validate() -> dict[str, Any]:
    """Run meaningful numerical and semantic checks, raising on failure."""
    comparisons = []
    for label, delta_ntn, delta_tn in (
        ("main", 0.025, 0.30),
        ("randomization_required", 0.0075, 0.25),
        ("infeasible_tn_deadline", 0.20, 0.10),
    ):
        graph = tiny_graph(delta_ntn)
        bounds = _bounds(delta_tn)
        occupancy = solve_occupancy(graph, bounds)
        mixture = solve_deterministic_mixture(graph, bounds)
        assert len(mixture.policies) == 128, "Enumeration must include every contingent policy."
        assert occupancy.feasible == mixture.feasible, "Independent LP formulations disagree."
        deterministic_feasible = [
            evaluation for evaluation in mixture.evaluations
            if all(evaluation.costs.get(key, 0.0) <= bound + 1e-8 for key, bound in bounds.items())
        ]
        difference = None
        if occupancy.feasible:
            difference = abs(occupancy.objective - mixture.objective)
            assert difference < 1e-8, "Occupation and complete-policy-mixture optimum differ."
            assert max(occupancy.residuals.values()) < 1e-8
            assert max(mixture.residuals.values()) < 1e-8
            evaluation = evaluate_policy(graph, occupancy.policy)
            assert abs(evaluation.costs["ntn_active"] - 3.0) < 1e-10
        if label == "randomization_required":
            assert occupancy.feasible and not deterministic_feasible
            accepted = occupancy.policy["sense=long/obs=accept"]["serve"]
            assert abs(accepted - 0.9375) < 1e-8
            assert abs(occupancy.objective - 1.125) < 1e-8
            assert abs(occupancy.costs["tn_failure"] - 0.25) < 1e-8
        if label == "infeasible_tn_deadline":
            assert not occupancy.feasible and occupancy.status == "infeasible"
        comparisons.append({
            "case": label, "delta_ntn": delta_ntn, "delta_tn": delta_tn,
            "occupation_status": occupancy.status, "mixture_status": mixture.status,
            "occupation_objective": occupancy.objective, "mixture_objective": mixture.objective,
            "objective_difference": difference, "enumerated_complete_policies": len(mixture.policies),
            "feasible_deterministic_policies": len(deterministic_feasible),
            "max_occupation_residual": max(occupancy.residuals.values(), default=None),
        })
    # Muting preserves the activity denominator and yields zero controlled-BS risk.
    graph = tiny_graph(0.025)
    mute_policy = {state: {("short" if state == "initial" else "mute"): 1.0}
                   for state in graph.actions}
    muted = evaluate_policy(graph, mute_policy)
    assert muted.costs["ntn_risk"] == 0.0
    assert abs(muted.costs["ntn_active"] - 3.0) < 1e-10
    assert abs(muted.costs["tn_failure"] - 1.0) < 1e-10
    # A positive mean payload does not replace the terminal deadline event.
    high_payload = {state: {("short" if state == "initial" else "serve"): 1.0}
                    for state in graph.actions}
    serviced = evaluate_policy(graph, high_payload)
    assert serviced.objective > 1.2
    assert abs(serviced.costs["tn_failure"] - 0.20) < 1e-10
    # Invalid cycles and unavailable actions must not silently produce a policy.
    cyclic = Graph({"s": [Action("loop", 1.0, [("s", 1.0)])]}, {"s": 1.0}, set())
    try:
        solve_occupancy(cyclic, {})
    except ValueError:
        pass
    else:
        raise AssertionError("A history graph cycle was not rejected.")
    try:
        evaluate_policy(graph, {"initial": {"clairvoyant": 1.0}})
    except ValueError:
        pass
    else:
        raise AssertionError("An unavailable action was not rejected.")
    return {"passed": True, "comparisons": comparisons,
            "beam_elimination": validate_beam_elimination(),
            "all_mute_active_time": muted.costs["ntn_active"],
            "all_mute_risk": muted.costs["ntn_risk"],
            "mean_payload_above_target_but_deadline_failure": serviced.costs["tn_failure"]}


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def run(output_dir: str | Path, seed: int = 2026) -> dict[str, Any]:
    """Validate E6 and save vector figures, preview PNGs, CSVs and a LaTeX table."""
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    validation = validate()
    comparisons = validation["comparisons"]
    _write_csv(out / "E6_validation.csv", comparisons)
    _write_csv(out / "E6_beam_elimination.csv", [validation["beam_elimination"]])
    main_graph = tiny_graph(0.025)
    main = solve_occupancy(main_graph, _bounds(0.30))
    main_mixture = solve_deterministic_mixture(main_graph, _bounds(0.30))
    _write_csv(out / "E6_optimal_policy.csv", [
        {"history": state, "action": action.name,
         "occupation": main.occupancy[state, action.name],
         "conditional_probability": main.policy[state][action.name],
         "reward": action.reward, **{key: action.costs.get(key, 0.0) for key in
          ("tn_failure", "ntn_risk", "ntn_active", "ntn_excess")}}
        for state, actions in main_graph.actions.items() for action in actions
    ])
    deterministic_rows = [
        {"policy_index": index, "mean_payload": evaluation.objective,
         "tn_failure": evaluation.costs["tn_failure"],
         "ntn_risk_active_ratio": evaluation.costs["ntn_risk"] / evaluation.costs["ntn_active"],
         "optimal_mixture_weight": float(main_mixture.weights[index])}
        for index, evaluation in enumerate(main_mixture.evaluations)
    ]
    _write_csv(out / "E6_all_deterministic_policies.csv", deterministic_rows)
    sweep_rows = []
    for delta_ntn in (0.004, 0.007, 0.0075, 0.008, 0.01, 0.015, 0.025, 0.04, 0.08):
        graph = tiny_graph(delta_ntn)
        bound = _bounds(0.25)
        opt = solve_occupancy(graph, bound)
        exhaustive = solve_deterministic_mixture(graph, bound)
        feasible_values = [evaluation.objective for evaluation in exhaustive.evaluations
                           if all(evaluation.costs.get(k, 0.0) <= v + 1e-8 for k, v in bound.items())]
        assert opt.feasible == exhaustive.feasible
        if opt.feasible:
            assert abs(opt.objective - exhaustive.objective) < 1e-8
        sweep_rows.append({
            "delta_ntn": delta_ntn, "delta_tn_fixed": 0.25, "occupation_status": opt.status,
            "occupation_objective": opt.objective, "full_mixture_objective": exhaustive.objective,
            "best_deterministic_objective": max(feasible_values, default=None),
            "randomization_needed_for_feasibility": bool(opt.feasible and not feasible_values),
            "tn_failure": opt.costs.get("tn_failure") if opt.feasible else None,
            "ntn_risk_active_ratio": opt.costs["ntn_risk"] / opt.costs["ntn_active"] if opt.feasible else None,
        })
    _write_csv(out / "E6_randomization_sweep.csv", sweep_rows)
    with plt.rc_context({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.grid": True, "grid.alpha": 0.25}):
        fig, ax = plt.subplots(figsize=(4.4, 3.1), constrained_layout=True)
        sc = ax.scatter([r["ntn_risk_active_ratio"] for r in deterministic_rows],
                        [r["tn_failure"] for r in deterministic_rows],
                        c=[r["mean_payload"] for r in deterministic_rows], s=35,
                        cmap="viridis", alpha=0.7, label="Complete deterministic policies")
        ax.scatter([main.costs["ntn_risk"] / main.costs["ntn_active"]],
                   [main.costs["tn_failure"]], marker="*", s=170, c="tab:red",
                   edgecolor="black", linewidth=0.5, label="Occupation LP optimum", zorder=5)
        ax.axvline(0.025, color="0.35", linestyle="--", linewidth=1)
        ax.axhline(0.30, color="0.35", linestyle="--", linewidth=1)
        ax.set(xlabel="Certified risk / NTN active time", ylabel="TN deadline failure probability",
               title="E6: declared finite model")
        ax.legend(loc="upper right", fontsize=6.5)
        fig.colorbar(sc, ax=ax, label="Expected TN payload (normalized bits)")
        for extension in ("pdf", "png"):
            fig.savefig(out / f"E6_policy_feasible_region.{extension}", dpi=220)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
        limits = np.array([r["delta_ntn"] for r in sweep_rows])
        convert = lambda key: [np.nan if r[key] is None else r[key] for r in sweep_rows]
        ax.plot(limits, convert("occupation_objective"), "o-", markersize=4, label="Occupation LP")
        ax.plot(limits, convert("full_mixture_objective"), "x", markersize=7,
                label="Mixture of all 128 policies")
        ax.plot(limits, convert("best_deterministic_objective"), "s--", markersize=4,
                label="Best feasible deterministic policy")
        for row in sweep_rows:
            if row["randomization_needed_for_feasibility"]:
                ax.annotate("Randomization required", (row["delta_ntn"], row["occupation_objective"]),
                            xytext=(30, 18), textcoords="offset points", fontsize=7,
                            arrowprops={"arrowstyle": "->", "lw": 0.7})
        ax.set(xscale="log", xlabel=r"NTN risk budget $\delta_{\mathrm{NTN}}$",
               ylabel="Expected TN payload (normalized bits)",
               title=r"E6: fixed TN failure limit $\delta_{\mathrm{TN}}=0.25$")
        ax.legend(fontsize=6.5, loc="best")
        for extension in ("pdf", "png"):
            fig.savefig(out / f"E6_lp_vs_exhaustive.{extension}", dpi=220)
        plt.close(fig)
    latex_rows = []
    labels = {"main": "Main case", "randomization_required": "Randomization required",
              "infeasible_tn_deadline": "Infeasible TN deadline"}
    for row in comparisons:
        fmt = lambda x: "--" if x is None else f"{x:.6f}"
        latex_rows.append(f"{labels[row['case']]} & {fmt(row['occupation_objective'])} & "
                          f"{fmt(row['mixture_objective'])} & {row['feasible_deterministic_policies']} \\\\")
    table = ("% Declared finite model. Both LPs enumerate the same causal policy class.\n"
             "\\begin{tabular}{lrrr}\n\\toprule\n"
             "Case & Occupation LP & Full-policy mixture & Feasible deterministic \\\\\n\\midrule\n"
             + "\n".join(latex_rows) + "\n\\bottomrule\n\\end{tabular}\n")
    (out / "E6_validation_table.tex").write_text(table, encoding="utf-8")
    summary = {
        "experiment": "E6", "data_source": "declared_synthetic_finite_history_model",
        "seed": seed, "random_sampling_used": False,
        "description": "Exact occupation LP versus mixture over every complete deterministic contingent policy.",
        "history_decision_times": 2, "complete_deterministic_policy_count": 128,
        "checks": validation,
        "main_objective": main.objective, "main_residuals": main.residuals,
        "main_tn_failure": main.costs["tn_failure"],
        "main_certified_ntn_ratio": main.costs["ntn_risk"] / main.costs["ntn_active"],
        "limitations": [
            "Exactness is for this supplied finite history graph and declared transition/cost model.",
            "Risk costs are declared conditional set-failure bounds, not measured physical outage.",
            "E6 checks the policy LP; continuous SOCP beam support is separately validated in E5.",
            "Infeasible points are omitted from objective curves and explicitly retained in CSV.",
        ],
        "artifacts": [path.name for path in sorted(out.glob("E6_*"))],
    }
    (out / "E6_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="result/appendix_d/E6")
    parser.add_argument("--seed", type=int, default=2026)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.output_dir, arguments.seed), indent=2, ensure_ascii=False))
