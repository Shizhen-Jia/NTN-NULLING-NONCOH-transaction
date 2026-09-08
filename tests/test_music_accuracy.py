"""Regression tests for blind MUSIC accuracy and numerical scale handling."""
import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

import ntn_music_detection as nmd
import nulling_cdf_utils as ncu


def array_positions(n=4):
    y, z = np.meshgrid(np.arange(n) * 0.5, np.arange(n) * 0.5)
    p = np.column_stack((np.zeros(n*n), y.ravel(), z.ravel()))
    return p - p.mean(axis=0)


class MusicAccuracyTests(unittest.TestCase):
    def setUp(self):
        self.p = array_positions()
        self.orientation = (0., np.deg2rad(6.), 0.)
        self.noise = 4e-13

    def steering(self, phi, theta, orientation=None):
        return nmd.array_position_steering_global(
            phi, theta, self.p, orientation_rad=orientation or self.orientation,
        ).ravel()

    def pipeline(self, h, **overrides):
        kwargs = dict(
            tx_rows=4, tx_cols=4, nsect=1, detect_num_sources=1,
            detect_covariance_mode="analytic", detect_noise_var=self.noise,
            array_positions_local=self.p, tx_orientations_rad=np.array([self.orientation]),
            peak_min_sep_phi_deg=0, peak_min_sep_theta_deg=0,
            peak_max_correlation=.98, peak_refine=True,
            phi_grid_deg=np.arange(0., 360., 2.), theta_grid_deg=np.arange(75., 116., 2.),
        )
        kwargs.update(overrides)
        return nmd.run_music_standard_pipeline(h, **kwargs)

    def test_mdl_scale_invariance(self):
        eig = np.r_[5e-12, 2e-12, np.full(62, 4e-13)]
        counts = [nmd._estimate_num_sources_mdl(eig*s, 800) for s in (1e-6, 1., 1e6, 1e12)]
        self.assertEqual(counts, [2]*4)

    def test_mdl_empty_and_isotropic(self):
        self.assertEqual(nmd._estimate_num_sources_mdl(np.zeros(16), 800), 0)
        self.assertEqual(nmd._estimate_num_sources_mdl(np.empty(0), 800), 0)
        self.assertEqual(nmd._estimate_num_sources_mdl(np.full(16, 4e-13), 800), 0)

    def test_noise_estimation_has_no_absolute_floor(self):
        out = {"eigenvalues_desc": np.r_[3e-12, np.full(15, self.noise)], "num_sources_est": np.array(1)}
        self.assertAlmostEqual(nmd.estimate_noise_power_from_music_out(out) / self.noise, 1.)

    def test_weak_gain_scale_invariance(self):
        u = np.eye(16, dtype=complex)[:2]
        gains = np.array([2e-12, 3e-13])
        signal = np.einsum("k,ka,kb->ab", gains, u, u.conj())
        for scale in (1e-6, 1., 1e6):
            result = nmd.estimate_noncoh_gains_from_covariance(
                scale*(signal+self.noise*np.eye(16)), u, noise_power=scale*self.noise)
            np.testing.assert_allclose(result/scale, gains, rtol=1e-10, atol=0)

    def test_nnls_correlated_columns_and_empty(self):
        u = np.array([self.steering(10, 95), self.steering(13, 95)])
        g = np.array([2e-12, 3e-13])
        r = np.einsum("k,ka,kb->ab", g, u, u.conj())
        recovered = nmd.estimate_noncoh_gains_from_covariance(r, u)
        np.testing.assert_allclose(recovered, g, rtol=1e-8, atol=0)
        self.assertEqual(nmd.estimate_noncoh_gains_from_covariance(r, np.empty((0,16))).size, 0)

    def test_exact_single_source_both_channel_conventions(self):
        u = self.steering(35.3, 95.4)
        h = (1e-5*u).reshape(1,1,1,16)
        for mode in ("raw", "conj"):
            out = self.pipeline(h, channel_mode=mode, covariance_refine=True)
            self.assertEqual(out["accepted_peak_counts"].tolist(), [1])
            self.assertGreater(abs(np.vdot(u, out["peak_u_hat_raw"][0])), .999999)
            np.testing.assert_allclose(out["peak_g_hat"], [1e-10], rtol=1e-6, atol=0)
            self.assertLessEqual(out["covariance_fit_after"][0], out["covariance_fit_before"][0]+1e-14)

    def test_all_sector_yaws_and_downtilt(self):
        for yaw in (0., 120., 240.):
            orientation = (np.deg2rad(yaw), np.deg2rad(6), 0.)
            u = self.steering((yaw+20.3)%360, 94.6, orientation)
            out = self.pipeline((1e-5*u).reshape(1,1,1,16),
                                tx_orientations_rad=np.array([orientation]), covariance_refine=True)
            self.assertGreater(abs(np.vdot(u, out["peak_u_hat_raw"][0])), .99999)

    def test_rear_signal_has_front_phase_representative(self):
        u = self.steering(140., 96.)
        out = self.pipeline((1e-5*u).reshape(1,1,1,16), covariance_refine=True,
                            theta_grid_deg=np.arange(65.,125.,1.))
        self.assertGreater(abs(np.vdot(u, out["peak_u_hat_raw"][0])), .99999)

    def test_empty_covariance_pipeline(self):
        out = self.pipeline(np.zeros((1,1,1,16), complex), detect_num_sources=None,
                            detect_source_estimation="rank", covariance_refine=True)
        self.assertEqual(out["peak_u_hat_raw"].shape, (0,16))
        self.assertEqual(out["accepted_peak_counts"].tolist(), [0])

    def test_invalid_refinement_settings_fail_before_detection(self):
        h = np.zeros((1,1,1,16), complex)
        for kwargs in ({"sector_forward_only": False}, {"array_positions_local": None},
                       {"covariance_refine_maxiter": 0}, {"pair_keys": [(0,0)]}):
            with self.assertRaises(ValueError):
                self.pipeline(h, covariance_refine=True, **kwargs)

    def test_flat_spectrum_does_not_fill_k(self):
        peaks = nmd.music_top_peaks(
            np.eye(16), num_rows=4, num_cols=4, array_positions_local=self.p,
            phi_grid_deg=np.arange(0.,360.,10.), theta_grid_deg=np.arange(75.,116.,5.),
            top_n=5, local_maxima_only=True, max_noise_projection=.2)
        self.assertEqual(peaks, [])

    def test_azimuth_seam(self):
        u = self.steering(359.7, 95.)
        out = self.pipeline((1e-5*u).reshape(1,1,1,16), covariance_refine=True)
        self.assertEqual(len(out["peak_g_hat"]), 1)
        self.assertGreater(abs(np.vdot(u, out["peak_u_hat_raw"][0])), .99999)

    def test_zenith_scan_never_refines_outside_physical_bounds(self):
        u = self.steering(20., 179.8)
        hi = (1e-5*u).reshape(1,1,16)
        out = nmd.detect_music_from_hi(hi, num_sources=1, noise_var=self.noise,
                                       covariance_mode="analytic", compute_user_scores=False)
        peaks = nmd.music_top_peaks(
            nmd.noise_subspace_from_music_out(out), num_rows=4, num_cols=4,
            array_positions_local=self.p, orientation_rad=self.orientation,
            phi_grid_deg=np.arange(0.,360.,10.), theta_grid_deg=np.arange(177.,181.,.5),
            top_n=1, refine_peaks=True, local_maxima_only=True,
        )
        self.assertTrue(all(0 <= th <= 180 for _, _, th, _ in peaks))

    def test_joint_refinement_improves_offset_initialization(self):
        u = np.array([self.steering(20.2, 94.4), self.steering(45.3, 98.1)])
        g = np.array([2e-10, 7e-11])
        r = np.einsum("k,ka,kb->ab", g, u, u.conj()) + self.noise*np.eye(16)
        _, vectors = np.linalg.eigh(r)
        en = vectors[:, :-2]
        peaks = [(1., ph, th, self.steering(ph, th).reshape(-1,1))
                 for ph, th in [(22.,95.), (43.5,97.)]]
        refined, _, diag = nmd.refine_music_covariance(
            r, peaks, array_positions_local=self.p, orientation_rad=self.orientation,
            en=en, noise_power=self.noise, maxiter=80)
        self.assertLess(diag["after"], diag["before"] * .01)
        self.assertGreater(abs(np.vdot(u[0], refined[0][3])), .9999)
        self.assertGreater(abs(np.vdot(u[1], refined[1][3])), .9999)

    def test_analytic_ignores_snapshot_count(self):
        h = (1e-5*self.steering(20,95)).reshape(1,1,1,16)
        a = self.pipeline(h, detect_num_snapshots=80)
        b = self.pipeline(h, detect_num_snapshots=8000)
        np.testing.assert_allclose(a["peak_g_hat"], b["peak_g_hat"])
        np.testing.assert_allclose(a["peak_u_hat_raw"], b["peak_u_hat_raw"])

    def test_sample_mode_is_seeded(self):
        h = (1e-5*self.steering(20,95)).reshape(1,1,1,16)
        kwargs = dict(detect_covariance_mode="sample", detect_num_snapshots=200, detect_rng_seed=42)
        a, b = self.pipeline(h, **kwargs), self.pipeline(h, **kwargs)
        np.testing.assert_allclose(a["peak_g_hat"], b["peak_g_hat"])
        self.assertGreater(abs(np.vdot(h.ravel()/1e-5, a["peak_u_hat_raw"][0])), .99)

    def test_global_matching_summary_and_true_g_agree(self):
        truth = np.array([self.steering(15,95), self.steering(30,95)])
        g = np.array([2e-12, 3e-13])
        h = (np.sqrt(g)[:,None]*truth).reshape(2,1,1,16)
        estimated = truth[::-1]
        matched_g, rho, _ = ncu._match_true_g_to_music_u(estimated, truth, g)
        lookup = {0: {"u": estimated, "g": matched_g}}
        stats = ncu.summarize_music_noncoh_quality(h, lookup)
        self.assertAlmostEqual(stats["u_rho_mean"], rho.mean())
        self.assertLess(stats["g_rel_err_mean"], 1e-12)

    def test_missing_sources_reduce_coverage(self):
        truth = np.array([self.steering(15,95), self.steering(75,95)])
        h = (1e-5*truth).reshape(2,1,1,16)
        stats = ncu.summarize_music_covariance_quality(h, {0:{"u": truth[:1], "g": np.array([1e-10])}})
        self.assertAlmostEqual(stats["power_coverage_rho95"], .5)
        empty = ncu.summarize_music_covariance_quality(h, {})
        self.assertEqual(empty["power_coverage_rho95"], 0.)
        self.assertAlmostEqual(empty["covariance_nrmse"], 1.)

    def test_nulling_without_true_gain_ablation_keeps_gain_accuracy(self):
        u = self.steering(15, 95)
        g = 2e-10
        h_ntn = (np.sqrt(g) * u).reshape(1, 1, 1, 16)
        h_tn = self.steering(40, 95).reshape(16, 1)
        lookup = {0: {"u": u[None, :], "g": np.array([1.25 * g])}}
        pairs = {0: [{"tn_idx": 0, "h_tn": h_tn, "w_t": h_tn,
                      "w_r": np.ones((1, 1)), "snr_raw_db": 0.}]}
        with patch.object(ncu, "_match_true_g_to_music_u",
                          side_effect=AssertionError("Truth matching in nulling")), \
             patch.object(ncu, "nulling_bf_music_noncoh",
                          wraps=ncu.nulling_bf_music_noncoh) as beamformer:
            out = ncu.run_small_round(
                h_tn.T.reshape(1, 1, 1, 16), h_ntn,
                pairs_by_tx=pairs, music_lookup=lookup, round_idx=0,
                lambda_ranges_music_est=[1e10], lambda_ranges_music_real=[1e10],
                tx_power=1., snr_noise_power=1., inr_noise_power=1.)
            self.assertEqual(beamformer.call_count, 2)
        self.assertFalse(any("music_u_true_g" in key for key in out))
        for mode in ("est", "music_real"):
            for metric in ("inr", "snr", "sinr"):
                values = out[f"{mode}_{metric}_db"][1e10]
                self.assertEqual(values.size, 1)
                self.assertTrue(np.isfinite(values).all())
        stats = ncu.summarize_music_noncoh_quality(h_ntn, lookup)
        self.assertAlmostEqual(stats["g_rel_err_mean"], .25)
        self.assertAlmostEqual(stats["g_rel_err_median"], .25)

    def test_experiment_cache_and_rng_isolation(self):
        class Scene:
            tx_pos = np.array([[0.,0.,35.]])
            ntn_look_pos = np.array([0.,0.,1000.])
            tn_pos = rx_ntn_pos = np.array([[100.,0.,2.]])
            tx_orientation_rad = np.array([[0.,0.,0.]])
            def compute_positions(self, **kwargs):
                self.random_draw = np.random.random()
            def compute_paths(self, **kwargs):
                self.a_tn = np.zeros((1,1,1,16,1,1), complex)
                self.a_ntn = np.zeros((1,1,1,16,1,1), complex)
                self.paths_ntn = None
        scene = Scene()
        music = dict(tx_rows=4, tx_cols=4, nsect=1, detect_source_estimation="rank",
                     detect_covariance_mode="analytic", covariance_refine=True,
                     phi_grid_deg=[0.], theta_grid_deg=[90.])
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(ncu, "_scene_tx_array_positions_local", return_value=self.p), \
             patch.object(ncu, "build_ntn_truth_from_paths", return_value={"pair_map": {}}):
            state = np.random.get_state()
            out = ncu.run_nulling_cdf_experiment(
                scene, num_macro_sims=1, compute_positions_kwargs={"azimuth":10.,"elevation":40.},
                compute_paths_kwargs={}, h_tn_th=1., tx_antennas=16, tx_power=1.,
                snr_noise_power=1., inr_noise_power=1., music_kwargs=music,
                show_progress=False, print_music_u_corr=False,
                channel_cache_dir=tmp, position_rng_seed=123)
            np.testing.assert_array_equal(state[1], np.random.get_state()[1])
            self.assertEqual(out["raw_inr_db"].size, 0)
            cached = np.load(Path(tmp)/"channels_0000.npz", allow_pickle=False)
            np.testing.assert_array_equal(cached["satellite_angles_deg"], [10.,40.])
            self.assertTrue((Path(tmp)/"music_0000.npz").exists())
            saved = ncu.save_experiment_metrics(out, result_dir=tmp)
            with np.load(saved) as metrics:
                self.assertIn("macro_stats_noncoh_covariance_nrmse", metrics.files)
                self.assertIn("macro_stats_noncoh_g_rel_err_mean", metrics.files)
                self.assertFalse(any("music_u_true_g" in key for key in metrics.files))
            self.assertFalse(any("music_u_true_g" in key for key in out))

    def test_notebook_syntax_and_configuration(self):
        nb = json.loads(Path("Nulling_CDF.ipynb").read_text())
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]), filename=f"cell_{i}")
        source = "\n".join("".join(c.get("source",[])) for c in nb["cells"])
        self.assertIn("music_covariance_refine = True", source)
        self.assertIn('channel_cache_dir=result_dir / "channels"', source)
        self.assertNotIn("music_u_true_g", source)
        self.assertNotIn("MUSIC u + true g", source)


if __name__ == "__main__":
    unittest.main()
