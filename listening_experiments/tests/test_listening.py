"""Fast physical/statistical checks; run with unittest, without importing Sionna."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np

from listening_experiments import (RTConfig, ExperimentConfig, MusicConfig, TimingConfig,
                                   prepare_channels, run_experiment, analyze_run, plot_figures)
from listening_experiments.cache import ChannelSnapshot
from listening_experiments.observations import window_observations, pooled_observations
from listening_experiments.detection import detect_directions
from listening_experiments.evaluation import prepare_links, final_beams, evaluate_beams
from listening_experiments.analysis import coverage_experiment, episode_metrics
from listening_experiments.runner import load_completed_run
from ntn_music_detection import array_position_steering_global


def fixture(macro=0, users=2):
    positions = np.array([[0, y, z] for y in (-.25, .25) for z in (-.25, .25)])
    tn = np.zeros((2, 1, 2, 4), complex)
    tn[0, 0, 0] = 1e-4 * np.array([1, 1, 1, 1])
    tn[1, 0, 1] = 1e-4 * np.array([1, 1j, -1, -1j])
    ntn = np.stack([array_position_steering_global(20 + 10 * i, 80, positions).ravel()
                    for i in range(users)]) * 1e-5
    ntn = np.tile(ntn[:, None, None, :], (1, 1, 2, 1))
    return ChannelSnapshot(macro, tn, ntn, ntn.copy(), positions, np.zeros((2, 3)),
                           7e9, 7e9, source_id=f"controlled-fixture-{macro}-{users}").validate()


class ObservationChecks(unittest.TestCase):
    def test_grouped_and_concatenated_covariances_agree_and_prefixes_are_stable(self):
        c = fixture()
        config = ExperimentConfig(p_values=(0., .3, 1.), listening_windows=(1, 4), snapshots_per_window=32)
        pooled = list(pooled_observations(c, config, episode=0))
        observations = [window_observations(c, config, 0, w) for w in range(4)]
        combined = np.concatenate([x[0] for x in observations], axis=-1)
        direct = combined @ combined.swapaxes(-1, -2).conj() / 128
        np.testing.assert_allclose(pooled[-1][1], direct, rtol=1e-12, atol=1e-24)
        np.testing.assert_array_equal(pooled[-1][2], sum(x[1].astype(int) for x in observations))
        np.testing.assert_array_equal(pooled[-1][2][0], 0)
        np.testing.assert_array_equal(pooled[-1][2][-1], 4)
        smaller = replace(config, p_values=(.3,), listening_windows=(1,))
        single = next(pooled_observations(c, smaller, 0))
        np.testing.assert_array_equal(single[1][0], pooled[0][1][1])
        self.assertFalse(np.array_equal(observations[0][0], window_observations(c, config, 1, 0)[0]))

    def test_waveforms_are_shared_across_bs_but_noise_is_independent(self):
        c = fixture()
        config = ExperimentConfig(p_values=(0., 1.), snapshots_per_window=20_000)
        data, _ = window_observations(c, config, 0, 0)
        # Identical channels at both BSs: subtract paired p=0 noise to expose the same signal.
        np.testing.assert_allclose(data[1, 0] - data[0, 0], data[1, 1] - data[0, 1], atol=1e-20)
        self.assertFalse(np.array_equal(data[0, 0], data[0, 1]))
        self.assertAlmostEqual(float(np.mean(abs(data[0]) ** 2)) / config.sensing_noise_normalized, 1, delta=.025)
        np.testing.assert_allclose(window_observations(c, replace(config, ul_power_dbm=33), 0, 0)[0][0],
                                   data[0] / np.sqrt(10), atol=1e-20)


class DetectionChecks(unittest.TestCase):
    def test_known_direction_power_transfer_and_noise_only_fallback(self):
        positions = np.array([[0, y / 2, z / 2] for y in range(4) for z in range(4)])
        vector = array_position_steering_global(20, 80, positions * .9).ravel()
        gain, noise = 1e-8, 1e-12
        covariance = gain * np.outer(vector.conj(), vector) + noise * np.eye(16)
        kwargs = dict(positions_dl=positions, orientations=np.zeros((1, 3)), f_ul=6.3e9, f_dl=7e9,
                      snapshots=4096, noise_normalized=noise,
                      config=MusicConfig(spatial_smoothing=True, subarray_rows=3, subarray_cols=3,
                                         grid_step_deg=2, peak_refine=False))
        for method in ("mdl", "rank"):
            with self.subTest(method=method):
                kwargs["config"] = replace(kwargs["config"], source_estimation=method)
                directions, diag = detect_directions(covariance[None], **kwargs)
                self.assertEqual(diag[0]["num_sources"], 1)
                self.assertEqual(diag[0]["accepted_peaks"], 1)
                np.testing.assert_allclose(directions[0]["g"], [.81 * gain], rtol=1e-6)
                target = array_position_steering_global(20, 80, positions).ravel()
                self.assertAlmostEqual(abs(np.vdot(directions[0]["u"][0], target)), 1, places=10)
                empty, diag = detect_directions((noise * np.eye(16))[None], **kwargs)
                self.assertEqual(diag[0]["accepted_peaks"], 0)
                self.assertEqual(empty[0]["u"].shape, (0, 16))

    def test_source_rule_and_rank_thresholds_change_the_selected_count(self):
        c = fixture()
        covariance = (1e-8 * np.diag([1.1, 1.01, 1., 1.]))[None]
        base = MusicConfig(spatial_smoothing=False, grid_step_deg=10, peak_refine=False,
                           rank_noise_margin=.001)  # Explicit legacy threshold for this contrast.
        self.assertEqual(base.source_estimation, "rank")
        cases = [(replace(base, source_estimation="mdl"), 0), (base, 2),
                 (replace(base, rank_noise_margin=.2), 0),
                 (replace(base, rank_relative_threshold=.2), 1)]
        for config, expected in cases:
            with self.subTest(config=config):
                _, diag = detect_directions(covariance, positions_dl=c.array_positions_dl,
                                            orientations=c.orientations[:1], f_ul=c.f_ul, f_dl=c.f_dl,
                                            snapshots=64, noise_normalized=1e-8, config=config)
                self.assertEqual(diag[0]["num_sources"], expected)
        for bad in (replace(base, source_estimation="invalid"),
                    replace(base, rank_relative_threshold=-1), replace(base, rank_noise_margin=float("nan"))):
            with self.assertRaises(ValueError):
                bad.validate()

    def test_calibrated_margin_scales_with_snapshots_and_rejects_sample_noise(self):
        from listening_experiments.calibrate_rank import noise_excess_samples
        positions = np.array([[0, y / 2, z / 2] for y in range(8) for z in range(8)])
        config = MusicConfig(rank_noise_margin=.45, rank_noise_reference_snapshots=128)
        self.assertEqual(config.rank_margin_for_snapshots(128), .45)
        self.assertEqual(config.rank_margin_for_snapshots(2048), .1125)
        self.assertEqual(config.rank_margin_for_snapshots(8192), .05625)
        with self.assertRaises(ValueError):
            config.rank_margin_for_snapshots(64)
        # Loading older configurations without a reference retains their fixed margin.
        legacy = MusicConfig(rank_noise_margin=.001)
        self.assertEqual(legacy.rank_margin_for_snapshots(8192), .001)
        # Actual random finite-snapshot noise, rather than an exact sigma^2 I matrix.
        excess = noise_excess_samples(positions, config, snapshots=128, trials=64, seed=778154)
        self.assertGreater(np.mean(excess > .001), .95)
        self.assertEqual(np.sum(excess > config.rank_margin_for_snapshots(128)), 0)

    def test_scaled_margin_reaches_source_counter(self):
        c = fixture()
        covariance = (1e-8 * np.diag([1.1, 1.01, 1., 1.]))[None]
        config = MusicConfig(spatial_smoothing=False, grid_step_deg=10, peak_refine=False,
                             rank_noise_margin=.45, rank_noise_reference_snapshots=128)
        for T, expected in ((128, 0), (8192, 1)):
            _, diag = detect_directions(covariance, positions_dl=c.array_positions_dl,
                                        orientations=c.orientations[:1], f_ul=c.f_ul, f_dl=c.f_dl,
                                        snapshots=T, noise_normalized=1e-8, config=config)
            self.assertEqual(diag[0]["num_sources"], expected)
            self.assertEqual(diag[0]["rank_noise_margin_effective"], config.rank_margin_for_snapshots(T))

    def test_empty_estimates_and_zero_lambda_match_mrt(self):
        c, config = fixture(), ExperimentConfig()
        pairs, protected = prepare_links(c, config)
        reference = evaluate_beams(c, pairs, [p["w_t"] for p in pairs], protected, config)
        estimates = [dict(u=np.empty((0, 4), complex), g=np.empty(0)) for _ in pairs]
        fallback = evaluate_beams(c, pairs, final_beams(pairs, estimates, config), protected, config)
        for a, b in zip(reference, fallback):
            np.testing.assert_allclose(a, b, rtol=1e-12)
        estimates = [dict(u=c.h_ntn_dl[:, 0, b], g=np.ones(2)) for b in range(2)]
        zero = replace(config, lambda_weight=0)
        for a, b in zip(reference, evaluate_beams(c, pairs, final_beams(pairs, estimates, zero), protected, zero)):
            np.testing.assert_allclose(a, b, rtol=1e-12)


class CacheChecks(unittest.TestCase):
    def test_frozen_cache_reuses_original_provenance_and_checks_physical_config(self):
        from listening_experiments import load_cached_channels
        from listening_experiments.storage import atomic_json, atomic_npz, canonical_json, fingerprint
        config = RTConfig(ntn_users=2, tx_rows=2, tx_cols=2, f_ul=7e9)
        identity = dict(schema=1, config=config, code={"old_rt_version": "original"})
        c = fixture()
        with TemporaryDirectory() as temp:
            root = Path(temp)
            atomic_json(root / "rt_config.json", identity)
            path = root / "macro_0000.npz"
            atomic_npz(path, macro_id=np.array(0), h_tn_dl=c.h_tn_dl, h_ntn_dl=c.h_ntn_dl,
                       h_ntn_ul=c.h_ntn_ul, array_positions_dl=c.array_positions_dl, orientations=c.orientations,
                       f_dl=np.array(c.f_dl), f_ul=np.array(c.f_ul),
                       metadata_json=np.array(canonical_json(dict(schema=1, cache_key=fingerprint(identity)))))
            timestamp = path.stat().st_mtime_ns
            with patch("listening_experiments.cache.rt_identity", side_effect=AssertionError("must use frozen manifest")):
                loaded = load_cached_channels(root, num_macros=1, expected_config=config)
            np.testing.assert_array_equal(loaded[0].h_ntn_ul, c.h_ntn_ul)
            self.assertEqual(path.stat().st_mtime_ns, timestamp)
            with self.assertRaises(ValueError):
                load_cached_channels(root, num_macros=1, expected_config=replace(config, f_ul=6.3e9))
            with self.assertRaises(FileNotFoundError):
                load_cached_channels(root, num_macros=2, expected_config=config)

    def test_rt_cache_expansion_and_reuse_only(self):
        with TemporaryDirectory() as temp, patch("listening_experiments.cache.rt_identity", return_value={"fixture": 1}), \
                patch("listening_experiments.rt_backend.RTBackend") as backend:
            def trace(macro):
                c = fixture(macro)
                return dict(macro_id=np.array(macro), h_tn_dl=c.h_tn_dl, h_ntn_dl=c.h_ntn_dl,
                            h_ntn_ul=c.h_ntn_ul, array_positions_dl=c.array_positions_dl,
                            orientations=c.orientations, f_dl=np.array(c.f_dl), f_ul=np.array(c.f_ul)), {}
            backend.return_value.trace.side_effect = trace
            first = prepare_channels(RTConfig(), 2, temp)
            self.assertEqual(backend.return_value.trace.call_count, 2)
            backend.reset_mock()
            cached = prepare_channels(RTConfig(), 2, temp, policy="reuse_only")
            backend.assert_not_called()
            self.assertEqual([c.source_id for c in first], [c.source_id for c in cached])
            with self.assertRaises(FileNotFoundError):
                prepare_channels(RTConfig(), 3, temp, policy="reuse_only")
            backend.assert_not_called()
            prepare_channels(RTConfig(), 3, temp)
            backend.return_value.trace.assert_called_once_with(2)


class AccountingChecks(unittest.TestCase):
    def test_statistics_weight_scenes_equally_despite_different_user_counts(self):
        from dataclasses import asdict
        config = ExperimentConfig(p_values=(.1,), listening_windows=(1,), episodes_per_scene=1)
        manifest = dict(channels=[dict(macro_id=0), dict(macro_id=1)], experiment=asdict(config), music={})
        records = []
        for macro, n, inr in ((0, 1, 10.), (1, 3, .01)):
            records.append(dict(macro_id=np.array(macro), ue_ids=np.arange(n), listening_windows=np.array([1]),
                                inr_mrt_linear=np.full(n, inr), inr_final_linear=np.full((1, 1, n), inr),
                                sinr_mrt_linear=np.array([1.]), sinr_final_linear=np.ones((1, 1, 1)),
                                J=np.ones((1, 1, n)), accepted_peaks=np.zeros((1, 1, 1))))
        with TemporaryDirectory() as temp, patch("listening_experiments.analysis.load_completed_run",
                                               return_value=(manifest, config, records)):
            a = analyze_run(temp, bootstrap_draws=100, coverage_samples=100)
        self.assertEqual(a["summary"][0]["q_out"], .5)  # Flattening users would incorrectly produce .25.

    def test_mrt_service_identity_and_time_validation(self):
        timing = TimingConfig()
        d = timing.durations(64)
        self.assertAlmostEqual(d["rf"] + d["mrt"] + d["final"], timing.horizon)
        record = dict(listening_windows=np.array([1, 64]),
                      inr_mrt_linear=np.array([.1, 1, 2]),
                      inr_final_linear=np.tile([.1, 1, 2], (1, 2, 1)),
                      sinr_mrt_linear=np.array([1., 3.]),
                      sinr_final_linear=np.tile([1., 3.], (1, 2, 1)),
                      J=np.ones((1, 2, 3)), accepted_peaks=np.zeros((1, 2, 2)))
        metrics = episode_metrics(record, timing, .5)
        np.testing.assert_allclose(metrics["q_data"], metrics["q_mrt"])
        np.testing.assert_allclose(metrics["r_eff"], metrics["r_mrt_scheduled"])
        np.testing.assert_allclose(metrics["r_mrt"], 1.5)  # Mean of link rates, not log of mean SINR.
        with self.assertRaises(ValueError):
            TimingConfig(horizon=1).durations(1)

    def test_coverage_model_matches_independent_trials(self):
        rows = coverage_experiment((0., .3, 1.), (1, 16), [1, 20], samples=20_000, seed=17)
        for row in rows:
            self.assertLessEqual(row["q_miss_exact"], row["union_bound"] + 1e-15)
            self.assertAlmostEqual(row["q_miss_mc"], row["q_miss_exact"], delta=.012)
        self.assertEqual(rows[-1]["q_miss_mc"], 0)
        self.assertEqual(rows[0]["q_miss_mc"], 1)


class EndToEndChecks(unittest.TestCase):
    def test_resume_and_reanalysis_reuse_saved_physical_results(self):
        import matplotlib.pyplot as plt
        config = ExperimentConfig(p_values=(0., 1.), listening_windows=(1, 2),
                                  snapshots_per_window=32, episodes_per_scene=2)
        music = MusicConfig(spatial_smoothing=False, grid_step_deg=10, peak_refine=False)
        channels = [fixture(0, 1), fixture(1, 3)]
        with TemporaryDirectory() as temp:
            run_dir = run_experiment(channels, config, music, temp)
            first = {p: p.stat().st_mtime_ns for p in (run_dir / "episodes").glob("*.npz")}
            # No estimator call is allowed when all episode checkpoints already exist.
            with patch("listening_experiments.runner.detect_directions", side_effect=AssertionError("must reuse")):
                self.assertEqual(run_experiment(channels, config, music, temp), run_dir)
            self.assertEqual(first, {p: p.stat().st_mtime_ns for p in first})
            manifest, _, records = load_completed_run(run_dir)
            self.assertEqual(manifest["music"]["source_estimation"], "rank")
            self.assertIn("RANK-MUSIC", manifest["estimator"])
            self.assertEqual([len(r["ue_ids"]) for r in records], [1, 1, 3, 3])
            # Use saved records to check equal-scene aggregation with unequal user counts.
            expected = np.mean([np.mean([episode_metrics(r, TimingConfig(), 10 ** (-.3))["q_out"]
                                         for r in records if int(r["macro_id"]) == m], axis=0) for m in (0, 1)], axis=0)
            a = analyze_run(run_dir, bootstrap_draws=100, coverage_samples=1000)
            np.testing.assert_allclose(np.array([r["q_out"] for r in a["summary"]]).reshape(2, 2), expected)
            b = analyze_run(run_dir, TimingConfig(horizon=512), bootstrap_draws=100, coverage_samples=1000)
            self.assertNotEqual(a["directory"], b["directory"])
            self.assertGreater(b["summary"][0]["r_eff"], a["summary"][0]["r_eff"])
            for fig in plot_figures(a):
                plt.close(fig)
            self.assertTrue((Path(a["directory"]) / "fig11_listening_service_cost.pdf").is_file())
            # A partial run is rejected by the analysis and repaired by the runner.
            missing = next(iter(first))
            missing.unlink()
            with self.assertRaises(FileNotFoundError):
                analyze_run(run_dir, coverage_samples=10)
            run_experiment(channels, config, music, temp)
            self.assertTrue(missing.is_file())
            for path, timestamp in first.items():
                if path != missing:
                    self.assertEqual(path.stat().st_mtime_ns, timestamp)

    def test_single_scene_does_not_claim_scene_confidence_intervals(self):
        config = ExperimentConfig(p_values=(0.,), listening_windows=(1,), snapshots_per_window=16, episodes_per_scene=1)
        music = MusicConfig(spatial_smoothing=False, grid_step_deg=10, peak_refine=False)
        with TemporaryDirectory() as temp:
            result = run_experiment([fixture()], config, music, temp)
            a = analyze_run(result, bootstrap_draws=10, coverage_samples=10)
            self.assertIsNone(a["summary"][0]["r_eff_lower"])
            self.assertTrue(any("one independent scene" in w for w in a["warnings"]))


if __name__ == "__main__":
    unittest.main()
