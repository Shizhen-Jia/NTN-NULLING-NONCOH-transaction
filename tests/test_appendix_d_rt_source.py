"""Cache provenance/isolation checks that do not require a GPU or Sionna."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from appendix_d_experiments.cache_adapter import inspect_cache
from appendix_d_experiments.rt_source import fresh_rt_config, generate_fresh_rt_cache
from multipath_support import collapse_cir_to_narrowband


class FreshRTTests(unittest.TestCase):
    def test_rejects_insufficient_drops_before_loading_rt(self):
        with patch('appendix_d_experiments.rt_source._load_backend') as backend:
            for n in [0, 1, 5, 6.5, True]:
                with self.assertRaises(ValueError):
                    generate_fresh_rt_cache('/tmp/unused-fresh-rt-test', num_macros=n)
            backend.assert_not_called()
        config = fresh_rt_config()
        self.assertEqual(config['expected_split_counts'], dict(train=2, calibration=2, test=2))
        self.assertEqual(config['music']['detect_covariance_mode'], 'sample')
        self.assertEqual(config['music']['detect_source_estimation'], 'mdl')
        self.assertAlmostEqual(config['music']['detect_noise_var'],
                               config['bs_noise_power'] / config['ul_tx_power'])

    def test_backend_failure_marks_archive_failed_and_preserves_prior_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = Path(tmp) / 'existing_cache'
            previous.mkdir()
            marker = previous / 'channels.npz'
            marker.write_bytes(b'preserved archive')
            with patch('appendix_d_experiments.rt_source._load_backend',
                       side_effect=RuntimeError('RT backend unavailable')):
                with self.assertRaisesRegex(RuntimeError, 'RT backend unavailable'):
                    generate_fresh_rt_cache(tmp)
            archives = list(Path(tmp).glob('fresh_rt_*'))
            self.assertEqual(len(archives), 1)
            root = archives[0]
            status = json.loads((root / 'run_status.json').read_text())
            self.assertEqual(status['status'], 'failed')
            self.assertEqual(status['completed_macros'], 0)
            self.assertIn('RT backend unavailable', status['error'])
            self.assertTrue((root / 'run_config.json').is_file())
            self.assertTrue((root / 'artifacts.json').is_file())
            self.assertFalse((root / 'fresh_rt_summary.json').exists())
            self.assertEqual(marker.read_bytes(), b'preserved archive')

    def test_whole_drop_provenance_coherent_truth_and_blind_ul_boundary(self):
        # The fake RT backend gives two cancelling DL paths. Any strongest-ray
        # or path-power shortcut is detectable, as is feeding DL truth to MUSIC.
        draws, detector_draws, rt_seeds = [], [], []

        class Scene:
            def __init__(self, scene):
                self.scene = scene
                self.tx_pos = np.zeros((4, 3))
                self.tx_orientation_rad = np.zeros((2, 3))
                self.ntn_look_pos = np.ones(3)

            def build_coverage_map(self, **kwargs):
                pass

            def compute_positions(self, **kwargs):
                self.tn_pos = np.random.uniform(size=(2, 3))
                self.rx_ntn_pos = np.random.uniform(size=(2, 3))
                draws.append(self.tn_pos.copy())

            def compute_paths(self, **kwargs):
                rt_seeds.append(kwargs['propagation_options']['seed'])
                self.a_tn = np.ones((2, 1, 2, 4, 2), complex)
                self.a_ntn = np.ones((2, 1, 2, 4, 2), complex)
                self.a_ntn[..., 1] = -1
                self.tau_tn = self.tau_ntn = np.zeros((2, 2, 2))

            def compute_ntn_ul_paths(self, frequency, **kwargs):
                return None, np.full((2, 1, 2, 4, 1), 3 + 4j), np.zeros((2, 2, 1))

        def detector(ul, *, music_kwargs, **kwargs):
            np.testing.assert_array_equal(ul, np.full((2, 1, 2, 4), 3 + 4j))
            self.assertEqual(music_kwargs['detect_covariance_mode'], 'sample')
            detector_draws.append(music_kwargs['detect_rng_seed'].random())
            return dict(peak_t_idx=np.array([0, 1]), peak_g_hat=np.ones(2),
                        peak_u_hat_raw=np.ones((2, 4), complex), anonymous_only={})

        def transfer(detected, *, dl_array_positions, music_kwargs, num_tx):
            # Only angles/manifold inputs are accepted by this signature.
            return dict(detected, peak_u_hat_raw=2 * detected['peak_u_hat_raw'])

        positions = np.array([[0, 0, 0], [0, .5, 0], [0, 0, .5], [0, .5, .5]])
        backend = (lambda _: object(), Scene,
                   SimpleNamespace(collapse_cir_to_narrowband=collapse_cir_to_narrowband,
                                   run_multipath_music_pipeline=detector),
                   SimpleNamespace(_scene_tx_array_positions_local=lambda _: positions,
                                   transfer_music_peaks_to_dl=transfer))
        before = np.random.get_state()
        with tempfile.TemporaryDirectory() as tmp:
            with patch('appendix_d_experiments.rt_source._load_backend', return_value=backend):
                result = generate_fresh_rt_cache(tmp, seed=451)
            root = Path(result['cache_dir'])
            self.assertEqual(inspect_cache(root)['num_macros'], 6)
            self.assertFalse(result['temporal_data_available'])
            self.assertEqual(json.loads((root / 'run_status.json').read_text())['status'], 'complete')
            seed_rows = json.loads((root / 'macro_seeds.json').read_text())
            for stream in ['position', 'satellite', 'propagation', 'music']:
                self.assertEqual(len({r[stream] for r in seed_rows}), 6)
            self.assertEqual(rt_seeds, [r['propagation'] for r in seed_rows])
            self.assertEqual(len(set(detector_draws)), 6)
            self.assertEqual(len({d.tobytes() for d in draws}), 6)
            with np.load(root / 'channels/channels_0000.npz') as data:
                np.testing.assert_array_equal(data['h_ntn'], 0)
            with np.load(root / 'channels/ul_000/music_0000.npz') as data:
                np.testing.assert_array_equal(data['peak_u_used_for_dl'], 2)
                np.testing.assert_array_equal(data['peak_u_hat_raw'], 1)
        after = np.random.get_state()
        for first, second in zip(before, after):
            np.testing.assert_equal(first, second)


if __name__ == '__main__':
    unittest.main()
