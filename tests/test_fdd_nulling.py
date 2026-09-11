"""Fixed-array FDD regression: independent UL sensing and shared DL evaluation."""
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


def positions():
    y, z = np.meshgrid(np.arange(4)*.5, np.arange(4)*.5)
    p = np.column_stack((np.zeros(16), y.ravel(), z.ravel()))
    return p - p.mean(axis=0)


def steering(p, phi=35., theta=95., orientation=(0., 0., 0.)):
    return nmd.array_position_steering_global(phi, theta, p, orientation_rad=orientation).ravel()


class ToyScene:
    """Single LoS source whose electrical array spacing follows actual frequency."""
    fc = 7e9
    tx_pos = np.array([[0., 0., 30.]])
    ntn_look_pos = np.array([0., 0., 1e6])
    tn_pos = np.array([[100., -20., 2.]])
    rx_ntn_pos = np.array([[100., 70., 2.]])
    tx_orientation_rad = np.zeros((1, 3))

    def __init__(self):
        self.position_calls = self.dl_calls = 0
        self.ul_calls = []

    def compute_positions(self, **kwargs):
        self.position_calls += 1

    def compute_paths(self, **kwargs):
        self.dl_calls += 1
        self.a_tn = (2e-5*steering(positions(), 5., 92.)).reshape(1, 1, 1, 16, 1, 1)
        self.a_ntn = (1e-5*steering(positions())).reshape(1, 1, 1, 16, 1, 1)
        self.paths_ntn = None

    def compute_ntn_ul_paths(self, frequency_hz, **kwargs):
        self.ul_calls.append(frequency_hz)
        ratio = frequency_hz/self.fc
        cir = (1e-5/ratio*steering(positions()*ratio)).reshape(1, 1, 1, 16, 1, 1)
        return None, cir, None


class FddTests(unittest.TestCase):
    def run_toy(self, percentages=None, mode="angle", directory=None):
        scene = ToyScene()
        music = dict(
            tx_rows=4, tx_cols=4, nsect=1, detect_num_sources=1,
            detect_covariance_mode="analytic", detect_noise_var=1e-13,
            phi_grid_deg=np.arange(0, 90, 2), theta_grid_deg=np.arange(80, 111, 2),
            peak_min_sep_phi_deg=0, peak_min_sep_theta_deg=0,
            peak_max_correlation=.98, peak_refine=True, covariance_refine=True,
        )
        with patch.object(ncu, "_scene_tx_array_positions_local", return_value=positions()), \
             patch.object(ncu, "build_ntn_truth_from_paths", return_value={"pair_map": {}}):
            out = ncu.run_nulling_cdf_experiment(
                scene, num_macro_sims=2,
                compute_positions_kwargs=dict(azimuth=0., elevation=40.),
                compute_paths_kwargs=dict(fc=7e9),
                lambda_ranges_music_est=[1e10, 1.2e10], lambda_ranges_music_real=[1e10],
                h_tn_th=0., tx_antennas=16, tx_power=1.,
                snr_noise_power=1e-13, inr_noise_power=1e-13,
                music_kwargs=music, ul_frequency_percentages=percentages, ul_to_dl_mode=mode,
                show_progress=False, print_music_u_corr=False, channel_cache_dir=directory,
            )
        return scene, out

    def test_ul_tracer_keeps_physical_array_and_restores_dl_on_failure(self):
        # Execute the production method without requiring Sionna for unit tests.
        tree = ast.parse(Path("SceneConfigSionnaSectorDrop.py").read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                      and node.name == "compute_ntn_ul_paths")
        original_positions = positions()
        scene = SimpleNamespace(frequency=np.array([7e9]),
                                receivers={"ntn-0": None},
                                tx_array=SimpleNamespace(normalized_positions=original_positions))
        config = SimpleNamespace(scene=scene, fc=7e9, ntn_rx=1, rx_ntn_pos=np.zeros((1, 3)),
                                 paths_ntn="DL paths", a_ntn="DL CIR", tau_ntn="DL delays")
        frequencies = []
        def trace(**kwargs):
            frequencies.append(scene.frequency)
            np.testing.assert_allclose(scene.tx_array.normalized_positions / scene.frequency,
                                       original_positions / 7e9, rtol=1e-14, atol=0)
            return SimpleNamespace(cir=lambda **kw: ("UL CIR", "UL delays"))
        scope = dict(np=np, PathSolver=lambda: trace)
        exec(compile(ast.Module(body=[method], type_ignores=[]), "UL method", "exec"), scope)
        compute = scope["compute_ntn_ul_paths"]
        for frequency in (6.3e9, 7.7e9):
            _, a, _ = compute(config, frequency, max_depth=0)
            self.assertEqual(a, "UL CIR")
            self.assertIs(scene.tx_array.normalized_positions, original_positions)
            self.assertEqual(scene.frequency, 7e9)
            self.assertEqual(config.a_ntn, "DL CIR")
        self.assertEqual(compute(config, 7e9), ("DL paths", "DL CIR", "DL delays"))
        self.assertEqual(frequencies, [6.3e9, 7.7e9])
        def fail(**kwargs):
            raise RuntimeError("Simulated solver failure")
        scope["PathSolver"] = lambda: fail
        with self.assertRaises(RuntimeError):
            compute(config, 7.7e9)
        self.assertIs(scene.tx_array.normalized_positions, original_positions)
        self.assertEqual(scene.frequency, 7e9)

    def test_invalid_percentages(self):
        for value in ([], [-100], [-101], [np.nan], [np.inf], [0, 0]):
            with self.assertRaises(ValueError):
                ncu.validate_ul_frequency_percentages(value)
        self.assertEqual(ncu.validate_ul_frequency_percentages([-5, 0, 5]), [-5., 0., 5.])

    def test_zero_regression_and_shared_dl(self):
        _, legacy = self.run_toy()
        scene, sweep = self.run_toy([-10, 0, 10])
        self.assertEqual((scene.position_calls, scene.dl_calls), (2, 2))
        np.testing.assert_allclose(scene.ul_calls, [6.3e9, 7.7e9]*2)
        self.assertEqual(sweep['num_estimated_curves'], 6)
        for case in sweep['by_percentage'].values():
            for metric in ('inr', 'snr', 'sinr'):
                np.testing.assert_array_equal(case[f'raw_{metric}_db'], legacy[f'raw_{metric}_db'])
                np.testing.assert_array_equal(case[f'music_real_{metric}_db'][1e10],
                                              legacy[f'music_real_{metric}_db'][1e10])
                self.assertEqual(case[f'est_{metric}_db'][1e10].size, 2)
        zero = sweep['by_percentage'][0]
        for metric in ('inr', 'snr', 'sinr'):
            for l in (1e10, 1.2e10):
                np.testing.assert_array_equal(zero[f'est_{metric}_db'][l], legacy[f'est_{metric}_db'][l])

    def test_angle_transfer_uses_ul_estimate_and_preserves_weights(self):
        p = positions()
        ori = np.array([[np.deg2rad(120), np.deg2rad(6), 0.]])
        true_dl = steering(p, 145., 96., ori[0])
        for ratio in (.9, 1.1):
            true_ul = steering(p*ratio, 145., 96., ori[0])
            for mode in ('raw', 'conj'):
                kwargs = dict(
                    tx_rows=4, tx_cols=4, nsect=1, detect_num_sources=1,
                    array_positions_local=p*ratio, tx_orientations_rad=ori,
                    detect_covariance_mode='analytic', detect_noise_var=1e-13,
                    channel_mode=mode, covariance_refine=True,
                    phi_grid_deg=np.arange(100, 181, 2), theta_grid_deg=np.arange(80, 111, 2),
                    phi_offset_deg=10., phi_mirror_about_sector=True,
                )
                out = nmd.run_music_standard_pipeline((1e-5*true_ul).reshape(1, 1, 1, 16), **kwargs)
                original = out['peak_u_hat_raw'].copy()
                transferred = ncu.transfer_music_peaks_to_dl(
                    out, dl_array_positions=p, music_kwargs=kwargs, num_tx=1)
                self.assertGreater(abs(np.vdot(transferred['peak_u_hat_raw'][0], true_dl)), .999999)
                self.assertLess(abs(np.vdot(original[0], true_dl)), .9999)
                np.testing.assert_array_equal(out['peak_u_hat_raw'], original)
                np.testing.assert_array_equal(transferred['peak_g_hat'], out['peak_g_hat'])

    def test_raw_and_angle_nulling_differ(self):
        _, angle = self.run_toy([10], 'angle')
        _, raw = self.run_toy([10], 'raw')
        a = angle['by_percentage'][10]['est_inr_db'][1e10]
        r = raw['by_percentage'][10]['est_inr_db'][1e10]
        self.assertGreater(np.max(np.abs(a-r)), .01)

    def test_cache_and_archive_keep_cases_and_close_lambdas(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, out = self.run_toy([-5, 0, 5], directory=Path(tmp)/'channels')
            path = ncu.save_experiment_metrics(out, result_dir=tmp)
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['ul_frequency_percentages'], [-5, 0, 5])
                for index, case in enumerate(out['by_percentage'].values()):
                    prefix = f'ul_{index:03d}/'
                    self.assertEqual(saved[prefix+'ul_frequency_hz'], case['ul_frequency_hz'])
                    for i, l in enumerate((1e10, 1.2e10)):
                        np.testing.assert_array_equal(saved[prefix+f'est_inr_db_index_{i:03d}'], case['est_inr_db'][l])
                    cache = Path(tmp)/'channels'/f'ul_{index:03d}'/'sensing_0000.npz'
                    with np.load(cache) as sensing:
                        np.testing.assert_allclose(sensing['array_positions_local_ul']/sensing['ul_frequency_hz'], positions()/7e9)

    def test_notebook_plots_cartesian_product(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        nb = json.loads(Path('Nulling_CDF_SectorDrop.ipynb').read_text())
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']), filename=f'cell_{i}')
        _, sweep = self.run_toy([-5, 0, 5])
        with tempfile.TemporaryDirectory() as tmp, patch.object(plt, 'show'):
            scope = dict(np=np, plt=plt, nulling_cdf_results=sweep,
                         results_by_percentage=sweep['by_percentage'], result_dir=Path(tmp), plot_oracle=False)
            exec(''.join(nb['cells'][5]['source']), scope)
            self.assertEqual(len(scope['inr_ax'].lines), 1+6)
            self.assertEqual(len(set(line.get_label() for line in scope['inr_ax'].lines)), 7)
            self.assertTrue((Path(tmp)/'nulling_inr_cdf.pdf').exists())
            plt.close('all')


if __name__ == '__main__':
    unittest.main()
