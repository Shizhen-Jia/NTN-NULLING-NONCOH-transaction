"""Meaningful Appendix D regression checks on its declared finite model.

Run with an environment containing NumPy/SciPy/CVXPY/Clarabel:
    python -m unittest discover -s tests -p test_appendix_d.py -v

These tests establish implementation consistency, not validity of the synthetic
channel/observation model for a physical deployment or a Sionna trajectory.
"""

from dataclasses import replace
import unittest

import numpy as np

from appendix_d_experiments.dynamic import JointModel, ModelConfig
from appendix_d_experiments.e6 import validate as validate_e6
from appendix_d_experiments.policy import evaluate_policy


class AppendixDPolicyTests(unittest.TestCase):
    def test_exact_complete_policy_reference_and_randomization(self):
        result = validate_e6()
        self.assertTrue(result["passed"])
        random_case = next(row for row in result["comparisons"]
                           if row["case"] == "randomization_required")
        self.assertEqual(random_case["feasible_deterministic_policies"], 0)
        self.assertEqual(random_case["occupation_status"], "optimal")
        self.assertTrue(result["beam_elimination"]["passed"])


class AppendixDDynamicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This demand makes the early DL-window chance constraint active and
        # leaves a genuine adaptive-sensing advantage over the fixed baselines.
        cls.model = JointModel(ModelConfig(theta_dl=0.60))
        cls.graph = cls.model.build()
        cls.solution, _ = cls.model.optimize("J")
        if not cls.solution.feasible:
            raise AssertionError("The declared reference test model should be feasible.")

    def test_pending_observation_hidden_until_release(self):
        model = self.model
        initial = model.state_info["s"]
        accepted_zero, old_zero, _, _ = model.segment(initial, 1, True, "accept0")
        accepted_one, old_one, _, _ = model.segment(initial, 1, True, "accept1")
        rejected, old_reject, _, _ = model.segment(initial, 1, True, "reject")
        # gin=1, tau=1, gout=0, processing=1: usable at t=3. At t=2
        # processing is nonblocking and all outcomes must use the same old beam.
        for zero, one, rejection in zip(old_zero[:3], old_one[:3], old_reject[:3]):
            np.testing.assert_allclose(zero["v"], one["v"], atol=1e-12)
            np.testing.assert_allclose(one["v"], rejection["v"], atol=1e-12)
            self.assertEqual(one["record"], initial["record"])
            self.assertEqual(one["age"], one["t"] - initial["stamp"])
        self.assertEqual(old_one[2]["processing"], 1)
        self.assertEqual(old_one[2]["blocked"], 0)
        self.assertEqual(old_one[2]["tx"], 1)
        self.assertEqual(old_one[3]["record"], 1)
        self.assertEqual(old_one[3]["released"], 1)
        self.assertEqual(old_one[3]["age"], 2)
        self.assertEqual(accepted_one["stamp"], 1)
        self.assertEqual(rejected["stamp"], -1)
        self.assertGreater(np.linalg.norm(old_zero[3]["v"] - old_one[3]["v"]), 1e-3)
        # A long estimate is released exactly at the next decision boundary.
        next_long, long_steps, _, _ = model.segment(initial, 2, True, "accept1")
        self.assertTrue(all(step["record"] == 0 for step in long_steps))
        self.assertTrue(all(step["released"] == 0 for step in long_steps))
        self.assertEqual(next_long["record"], 1)
        self.assertEqual(next_long["stamp"], 1)
        np.testing.assert_allclose(next_long["belief"], np.eye(2)[1] @ np.linalg.matrix_power(model.P, 3))

    def test_failed_refresh_never_resets_age(self):
        initial = self.model.state_info["s"]
        info, _, _, _ = self.model.segment(initial, 2, True, "accept1")
        for outcome in ("reject", "no_detection"):
            next_info, steps, _, _ = self.model.segment(info, 1, True, outcome)
            self.assertEqual(next_info["stamp"], info["stamp"])
            self.assertEqual(next_info["record"], info["record"])
            self.assertEqual([step["age"] for step in steps], [3, 4, 5, 6])
            self.assertTrue(all(step["released"] == 0 for step in steps))
            np.testing.assert_allclose(next_info["belief"], info["belief"] @ np.linalg.matrix_power(self.model.P, 4))

    def test_dl_mute_preserves_ul_and_ntn_activity_denominator(self):
        info = self.model.state_info["s"]
        nxt, steps, reward, costs = self.model.segment(info, 0, False, "none")
        self.assertEqual(nxt["dl"], 0.0)
        self.assertEqual(nxt["ul"], self.model.config.ul_rate)
        self.assertEqual(reward, self.model.config.ul_rate)
        self.assertEqual(costs["active_0"], self.model.config.block)
        self.assertEqual(costs["active_1"], self.model.config.block)
        self.assertEqual(costs["certified_0"], 0.0)
        self.assertEqual(costs["risk_0"], -self.model.config.delta_ntn * self.model.config.block)
        self.assertEqual(sum(step["tx"] for step in steps), 0)
        self.assertEqual(sum(step["ul_bits_per_hz"] for step in steps), self.model.config.ul_rate)

    def test_blocking_processing_masks_the_declared_service_tick(self):
        cfg = replace(self.model.config, epochs=2, blocking_processing=True)
        blocked_model = JointModel(cfg)
        initial = self.model.state_info["s"]
        _, blocked, _, costs = blocked_model.segment(initial, 1, True, "accept1")
        self.assertEqual(blocked[2]["processing"], 1)
        self.assertEqual(blocked[2]["blocked"], 1)
        self.assertEqual(blocked[2]["dl_bits_per_hz"], 0.0)
        self.assertEqual(costs["gap_ticks"], 3.0)

    def test_complete_histories_and_executable_policy_flow(self):
        incoming = {state: 0 for state in self.model.state_info}
        for actions in self.graph.actions.values():
            for action in actions:
                self.assertAlmostEqual(sum(p for _, p in action.transitions), 1.0)
                for nxt, probability in action.transitions:
                    if probability:
                        incoming[nxt] += 1
        self.assertEqual(incoming.pop("s"), 0)
        self.assertTrue(all(count == 1 for count in incoming.values()), "Histories were merged.")
        evaluated = evaluate_policy(self.graph, self.solution.policy)
        self.assertAlmostEqual(evaluated.objective, self.solution.objective, places=8)
        self.assertAlmostEqual(sum(evaluated.terminal_mass.values()), 1.0, places=9)
        self.assertLess(max(self.solution.residuals.values()), 1e-7)
        for key, bound in self.model.bounds.items():
            self.assertLessEqual(evaluated.costs[key], bound + 1e-8)

    def test_joint_policy_contains_all_fixed_and_partial_classes(self):
        for mode in ("F", "T", "L"):
            baseline, _ = self.model.optimize(mode)
            self.assertTrue(baseline.feasible)
            self.assertGreaterEqual(self.solution.objective + 1e-7, baseline.objective)
        # For a fixed sensing start schedule, L must actually start there;
        # it may choose duration, but it may not silently skip that sensing.
        restricted = self.model.restrict("L", schedule=(1,))
        for state, actions in restricted.actions.items():
            epoch = self.model.state_info[state]["epoch"]
            durations = {int(action.name[1]) for action in actions}
            self.assertEqual(durations, {1, 2} if epoch == 1 else {0})

    def test_unavailable_listening_is_removed_from_every_policy_class(self):
        model = JointModel(replace(self.model.config, listening_epochs=(0,)))
        graph = model.build()
        for state, actions in graph.actions.items():
            if model.state_info[state]["epoch"] > 0:
                self.assertEqual({int(action.name[1]) for action in actions}, {0})
        joint, _ = model.optimize("J")
        for mode in ("F", "T", "L"):
            baseline, menu = model.optimize(mode)
            self.assertTrue(set(menu["schedule"]).issubset({0}))
            if baseline.feasible:
                self.assertTrue(joint.feasible)
                self.assertGreaterEqual(joint.objective + 1e-7, baseline.objective)

    def test_new_background_receiver_only_counts_after_arrival(self):
        model = JointModel(replace(self.model.config, background_arrival_tick=7))
        policy, _ = model.optimize()
        self.assertTrue(policy.feasible)
        self.assertAlmostEqual(policy.costs["active_1"], model.H - 7)
        self.assertAlmostEqual(policy.costs["risk_1"], -model.config.delta_ntn * (model.H - 7))
        initial = model.state_info["s"]
        _, _, _, first_costs = model.segment(initial, 0, True, "none")
        self.assertEqual(first_costs["active_1"], 0.0)
        self.assertEqual(first_costs["risk_1"], 0.0)
        # Deliberately amplified physical background checks event counting;
        # it is an out-of-bound stress case, not a certificate test.
        rows, samples, _ = model.simulate(policy, episodes=20, seed=92, background_gain=10.0)
        threshold = 10 ** (model.config.gamma_db / 10) * (1 + 1e-7)
        for row in rows:
            active_samples = [sample for sample in samples
                              if sample["episode"] == row["episode"] and sample["receiver"] == 1]
            self.assertEqual(sum(sample["dl_active"] for sample in active_samples), model.H - 7)
            self.assertEqual(row["active_1"], model.H - 7)
            self.assertEqual(row["exceed_1"], sum(sample["dl_active"] and sample["inr_linear"] > threshold
                                                for sample in active_samples))
        self.assertTrue(all(sample["dl_active"] == int(sample["t"] >= 7)
                            for sample in samples if sample["receiver"] == 1))

    def test_gamma_sweep_keeps_tn_requirements_fixed(self):
        base = ModelConfig(epochs=2, theta_dl=0.40)
        tight = JointModel(replace(base, gamma_db=-15.0))
        loose = JointModel(replace(base, gamma_db=-5.0))
        self.assertEqual(tight.reference, loose.reference)
        self.assertEqual(tight.bounds, loose.bounds)
        self.assertEqual(tight.config.theta_dl, loose.config.theta_dl)
        self.assertEqual(tight.config.theta_ul, loose.config.theta_ul)
        self.assertEqual(tight.config.delta_tn, loose.config.delta_tn)
        for age in (1, 3, 7):
            self.assertLessEqual(tight.beam(0, age).amplitude, loose.beam(0, age).amplitude + 1e-7)
        tight_policy, _ = tight.optimize()
        loose_policy, _ = loose.optimize()
        if tight_policy.feasible:
            self.assertTrue(loose_policy.feasible)
            self.assertLessEqual(tight_policy.objective, loose_policy.objective + 1e-7)

    def test_unattainable_deadline_is_reported_infeasible(self):
        model = JointModel(ModelConfig(epochs=2, gamma_db=-30.0, theta_dl=0.95))
        policy, _ = model.optimize()
        self.assertFalse(policy.feasible)
        self.assertEqual(policy.status, "infeasible")
        self.assertIsNone(policy.objective)
        self.assertEqual(model.config.theta_dl, 0.95, "TN target was silently relaxed.")
        self.assertEqual(model.simulate(policy, episodes=2), ([], [], []))

    def test_exogenous_physical_gain_does_not_leak_into_control(self):
        rows_a, samples_a, trace_a = self.model.simulate(self.solution, episodes=25, seed=283, physical_gain=1.0)
        rows_b, samples_b, trace_b = self.model.simulate(self.solution, episodes=25, seed=283, physical_gain=3.0)
        for a, b in zip(rows_a, rows_b):
            self.assertEqual(a["total_bits_per_hz"], b["total_bits_per_hz"])
        self.assertEqual([(t["action"], t["outcome"], t["record"]) for t in trace_a],
                         [(t["action"], t["outcome"], t["record"]) for t in trace_b])
        for a, b in zip(samples_a, samples_b):
            self.assertEqual(a["bs_transmitting"], b["bs_transmitting"])
            if a["receiver"] == 0:
                self.assertAlmostEqual(b["inr_linear"], 9 * a["inr_linear"], places=9)

    def test_independent_episode_simulation_matches_the_declared_law(self):
        count = 4000
        rows, _, _ = self.model.simulate(self.solution, episodes=count, seed=19027)
        payload = np.array([row["total_bits_per_hz"] for row in rows])
        service_se = payload.std(ddof=1) / np.sqrt(count)
        self.assertLessEqual(abs(payload.mean() - self.solution.objective), 5 * service_se + 1e-8)
        for name in self.model.bounds:
            if not name.startswith("fail_"):
                continue
            events = np.array([row[name] for row in rows], dtype=float)
            expected = self.solution.costs[name]
            # Independence is across trajectories, never between the correlated
            # slots within one trajectory. A small additive term handles p=0.
            standard_error = np.sqrt(max(expected * (1 - expected), 1 / count) / count)
            self.assertLessEqual(abs(events.mean() - expected), 5 * standard_error + 1 / count)
        for receiver in (0, 1):
            episode_outage = np.array([row[f"exceed_{receiver}"] / row[f"active_{receiver}"] for row in rows])
            # All trajectories have the same active denominator in this model.
            # The actual INR event is bounded by set failure and need not equal it.
            ratio = sum(row[f"exceed_{receiver}"] for row in rows) / sum(row[f"active_{receiver}"] for row in rows)
            bound = self.solution.costs[f"certified_{receiver}"] / self.solution.costs[f"active_{receiver}"]
            standard_error = episode_outage.std(ddof=1) / np.sqrt(count)
            self.assertLessEqual(ratio, bound + 5 * standard_error + 1 / count)


if __name__ == "__main__":
    unittest.main()
