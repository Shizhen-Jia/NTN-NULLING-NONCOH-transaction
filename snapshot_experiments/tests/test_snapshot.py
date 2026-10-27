"""Scientific invariants that guard against incorrect snapshot comparisons."""
import unittest
import numpy as np

from snapshot_experiments.config import SnapshotConfig
from snapshot_experiments.cache import load_inputs
from snapshot_experiments.metrics import match_directions, to_db, coverage_metrics
from snapshot_experiments.validation import check_observations
from listening_experiments.storage import read_npz
from listening_experiments.evaluation import final_beams, evaluate_beams
from multipath_support import summarize_path_estimates


class MatchingTests(unittest.TestCase):
    def test_one_peak_cannot_match_two_paths_and_ue_ids_are_deduplicated(self):
        truth = dict(valid=np.ones((1, 2, 2), bool),
                     phi_t=np.zeros((1, 2, 2)), theta_t=np.full((1, 2, 2), np.pi/2))
        directions = [dict(phi=np.array([0., 90.]), theta=np.array([90., 90.]), g=np.ones(2)),
                      dict(phi=np.array([0.]), theta=np.array([90.]), g=np.ones(1))]
        out = match_directions(truth, directions)
        np.testing.assert_array_equal(out["counts"], [[1, 1, 1], [1, 0, 1]])
        self.assertEqual(int(out["detected_mask"].sum()), 1)

    def test_zero_post_nulling_power_is_retained(self):
        values = to_db([0., 1., 10.], -120)
        np.testing.assert_array_equal(values, [-120., 0., 10.])

    def test_power_weights_and_fixed_high_inr_population(self):
        truth = dict(valid=np.ones((2, 2, 1), bool),
                     path_power=np.array([[[9.], [1.]], [[.1], [.01]]]))
        evaluation = dict(matches=np.array([[1, 0, 0, 0, 0], [0, 1, 0, 1, 0]]),
                          detected_mask=np.array([True, True]))
        out = coverage_metrics(truth, evaluation, [0, 1], [1., 10.], 0.)
        np.testing.assert_allclose(out["path_power_by_sector"].sum(axis=0), [1.1, 10.11])
        np.testing.assert_array_equal(out["high_inr_ue_ids"], [1])
        np.testing.assert_array_equal(out["high_inr_ue_counts"], [1, 1])

    def test_missing_high_inr_ue_is_not_credited_for_detecting_another_user(self):
        truth = dict(valid=np.ones((2, 1, 1), bool), path_power=np.array([[[9.]], [[1.]]]))
        evaluation = dict(matches=np.array([[0, 0, 0, 0, 0]]),
                          detected_mask=np.array([True, False]))
        out = coverage_metrics(truth, evaluation, [0, 1], [.01, 10.], -3.)
        np.testing.assert_array_equal(out["high_inr_ue_counts"], [0, 1])
        empty = coverage_metrics(truth, evaluation, [0, 1], [.01, .01], -3.)
        np.testing.assert_array_equal(empty["high_inr_ue_counts"], [0, 0])


class CacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = SnapshotConfig(num_scenes=1, repeats=1)
        cases, cls.physical, cls.observation, cls.manifest = load_inputs(cls.config)
        cls.case = cases[0]

    def test_baseline_matches_the_original_cdf(self):
        self.assertTrue(self.manifest["baseline_agrees_with_cdf"])

    def test_observations_have_actual_shared_prefixes_and_independent_noise(self):
        self.assertTrue(all(check_observations(self.case, self.observation).values()))

    def test_empty_detection_reduces_to_no_nulling(self):
        m = self.case.channel.h_ntn_dl.shape[-1]
        directions = [dict(u=np.empty((0, m)), g=np.empty(0)) for _ in self.case.pairs]
        beams = final_beams(self.case.pairs, directions, self.physical)
        inr, sinr = evaluate_beams(self.case.channel, self.case.pairs, beams, self.case.ue_ids, self.physical)
        np.testing.assert_allclose(inr, self.case.baseline_inr, rtol=1e-9, atol=1e-14)
        np.testing.assert_allclose(sinr, self.case.baseline_sinr, rtol=1e-9, atol=1e-14)

    def test_matching_agrees_with_the_original_path_evaluator(self):
        source = self.config.path(self.config.source_run)
        z = read_npz(source / "channels" / self.manifest["ul_subdirectory"] / "music_0000.npz")
        directions = []
        for tx in range(self.case.channel.h_ntn_ul.shape[2]):
            keep = z["peak_t_idx"] == tx
            directions.append(dict(phi=z["peak_phi_hat_deg"][keep], theta=z["peak_theta_hat_deg"][keep],
                                   g=z["peak_g_hat"][keep]))
        actual = match_directions(self.case.truth, directions)
        expected = summarize_path_estimates(self.case.truth, z)
        self.assertEqual(int(actual["counts"][:, 0].sum()), expected["matched_paths"])
        self.assertEqual(np.flatnonzero(actual["detected_mask"]).tolist(), expected["detected_rx_indices"])


if __name__ == "__main__":
    unittest.main()

