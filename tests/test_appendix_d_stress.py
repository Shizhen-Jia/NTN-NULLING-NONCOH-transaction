"""Tests for executed repeated observations and hidden-arrival causal controls."""
import unittest
import numpy as np

from appendix_d_experiments.dynamic import JointModel, ModelConfig
from appendix_d_experiments.stress import (
    ArrivalController, ArrivalDiagnosticConfig, fixed_repeated_policy,
    observation_diagnostics, simulate_random_arrival,
)


class RepeatedObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = JointModel(ModelConfig(gamma_db=-10))
        cls.policy = fixed_repeated_policy(cls.model)

    def test_two_executed_observations_expose_correlation_without_changing_marginals(self):
        independent, _, _ = self.model.simulate_policy(self.policy, episodes=1200, seed=639)
        correlated, _, _ = self.model.simulate_policy(self.policy, episodes=1200, seed=639,
                                                      burst_success=True)
        a = observation_diagnostics(independent, bootstrap=100)
        b = observation_diagnostics(correlated, bootstrap=100)
        for summary in (a, b):
            self.assertEqual(summary['min_sensing_count'], 2)
            self.assertEqual(summary['max_sensing_count'], 2)
            self.assertEqual(summary['repeated_listening_fraction'], 1.0)
            self.assertEqual(summary['observation_pair_count'], 1200)
            self.assertTrue(summary['temporal_correlation_stress_effective'])
        self.assertLess(abs(a['accepted_indicator_correlation']), .10)
        self.assertAlmostEqual(b['accepted_indicator_correlation'], 1.0)
        self.assertAlmostEqual(b['accepted_correlation_ci_low'], 1.0)
        self.assertEqual([r['observation_accepted_0'] for r in independent],
                         [r['observation_accepted_0'] for r in correlated])
        for rows in (independent, correlated):
            for epoch in (0, 1):
                self.assertLess(abs(np.mean([r[f'observation_accepted_{epoch}'] for r in rows])-.48), .05)
        # This diagnostic is intentionally not advertised as meeting TN service:
        # two scans consume two of the original three UL opportunities.
        self.assertTrue(all(r['fail_UL_12'] == 1 for r in independent))

    def test_nominal_j_single_listening_is_marked_ineffective_for_correlation(self):
        result, _ = self.model.optimize('J')
        rows, _, _ = self.model.simulate(result, episodes=80, seed=5, burst_success=True)
        summary = observation_diagnostics(rows)
        self.assertLessEqual(summary['max_sensing_count'], 1)
        self.assertFalse(summary['temporal_correlation_stress_effective'])
        self.assertIsNone(summary['accepted_indicator_correlation'])


class RandomArrivalTests(unittest.TestCase):
    def simulate(self, mode, kind='ul_active', arrivals=(0, 2, 6, 10)):
        return simulate_random_arrival(episodes=len(arrivals), receiver_kind=kind, mode=mode,
            arrival_ticks=arrivals, detection_uniforms=np.zeros((len(arrivals), 2)),
            arrival_config=ArrivalDiagnosticConfig(trace_episodes=len(arrivals)))

    def test_controller_interface_uses_only_released_detection(self):
        controller = ArrivalController('delayed_detection')
        for t in range(5):
            controller.receive_released_detection(t, False)
            self.assertFalse(controller.protect_background)
        controller.receive_released_detection(5, True)
        self.assertTrue(controller.protect_background)
        self.assertEqual(controller.protection_effective_tick, 5)
        controller.receive_released_detection(9, True)
        self.assertEqual(controller.protection_effective_tick, 5)

    def test_delayed_protection_waits_for_release_and_counts_only_active_time(self):
        rows, traces = self.simulate('delayed_detection')
        self.assertEqual(rows[0]['first_detection_reference_tick'], 3)
        self.assertEqual(rows[0]['first_detection_tick'], 4)
        self.assertEqual(rows[0]['first_release_tick'], 5)
        self.assertEqual(rows[0]['protection_effective_tick'], 5)
        self.assertGreater(rows[0]['before_protection_exceed_ticks'], 0)
        for row in rows:
            self.assertEqual(row['all_active_ticks'], 12-row['first_appearance_tick'])
            self.assertEqual(row['all_active_ticks'], row['before_detection_active_ticks']+row['after_detection_active_ticks'])
            self.assertEqual(row['all_active_ticks'], row['before_protection_active_ticks']+row['after_protection_active_ticks'])
            self.assertEqual(row['after_protection_exceed_ticks'], 0)
            if row['first_detection_tick'] is not None:
                self.assertGreaterEqual(row['first_detection_reference_tick'], row['first_appearance_tick'])
                self.assertEqual(row['protection_effective_tick'], row['first_release_tick'])
        self.assertIsNone(rows[-1]['first_detection_tick'])  # Arrival after the final scan.
        for tick in traces:
            row = rows[tick['episode']]
            release = row['first_release_tick']
            self.assertEqual(tick['protected'], int(release is not None and tick['t'] >= release))
            self.assertEqual(tick['dl_active'], int(tick['t'] >= row['first_appearance_tick']))

    def test_unknown_arrivals_with_same_observations_produce_same_controls(self):
        _, traces_a = self.simulate('delayed_detection', arrivals=(1,))
        _, traces_b = self.simulate('delayed_detection', arrivals=(2,))
        control_keys = ('t', 'detection_released', 'protected', 'rf_gap', 'bs_transmitting', 'power')
        self.assertEqual([tuple(r[k] for k in control_keys) for r in traces_a],
                         [tuple(r[k] for k in control_keys) for r in traces_b])
        # A nonresponding comparator remains identical even when the detection
        # sequence changes because the second receiver arrives after all scans.
        _, traces_a = self.simulate('no_prior_no_response', arrivals=(0,))
        _, traces_b = self.simulate('no_prior_no_response', arrivals=(10,))
        self.assertEqual([r['power'] for r in traces_a], [r['power'] for r in traces_b])

    def test_ul_silent_receiver_requires_prior_protection(self):
        prior, trace_prior = self.simulate('prior_background', 'ul_silent')
        delayed, trace_delayed = self.simulate('delayed_detection', 'ul_silent')
        for row in prior+delayed:
            self.assertIsNone(row['first_detection_tick'])
            self.assertIsNone(row['first_release_tick'])
        self.assertTrue(all(r['all_exceed_ticks'] == 0 for r in prior))
        self.assertTrue(all(r['protection_effective_tick'] == 0 for r in prior))
        self.assertTrue(all(r['protection_effective_tick'] is None for r in delayed))
        self.assertGreater(sum(r['all_exceed_ticks'] for r in delayed), 0)
        self.assertEqual([r['rf_gap'] for r in trace_prior], [r['rf_gap'] for r in trace_delayed])
        self.assertEqual([r['bs_transmitting'] for r in trace_prior], [r['bs_transmitting'] for r in trace_delayed])


if __name__ == '__main__':
    unittest.main()
