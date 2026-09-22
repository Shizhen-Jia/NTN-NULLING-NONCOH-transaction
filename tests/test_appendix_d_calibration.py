"""Regression checks for E4's explicit scene-level calibration target.

Only two small E4 datasets are generated. Plot rendering is replaced with a
mock; the real calibration, CSV, NPZ and manifest paths remain exercised.
"""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from appendix_d_experiments.e45 import _e4_calibration_plan, run_e4


class E4CalibrationPlanTests(unittest.TestCase):
    def test_profiles_change_sample_counts_without_changing_target(self):
        for requested_alpha in (0.10, 0.05):
            with self.subTest(alpha=requested_alpha):
                quick = _e4_calibration_plan(True, requested_alpha)
                full = _e4_calibration_plan(False, requested_alpha)
                self.assertTrue(all(large > small for small, large in zip(quick[:3], full[:3])))
                self.assertEqual(quick[3], requested_alpha)
                self.assertEqual(full[3], requested_alpha)
                for _, ncal, _, alpha, order in (quick, full):
                    # The chosen finite rank must reach the requested marginal
                    # coverage; the preceding rank must not already reach it.
                    self.assertGreaterEqual(order / (ncal + 1), 1 - alpha)
                    self.assertLess((order - 1) / (ncal + 1), 1 - alpha)
                    self.assertLessEqual(order, ncal)

    def test_invalid_or_unattainable_target_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "must_not_exist"
            for alpha in (None, "invalid", -0.1, 0, 1, 1.1, np.nan, np.inf):
                with self.subTest(alpha=alpha):
                    with self.assertRaisesRegex(ValueError, "alpha"):
                        run_e4(output, alpha=alpha)
                    self.assertFalse(output.exists())
            # A 0.5% target is too small for 64 calibration scenes but can use
            # a finite order statistic with the full profile's 256 scenes.
            with self.assertRaisesRegex(ValueError, "calibration scenes"):
                run_e4(output, quick=True, alpha=0.005)
            self.assertFalse(output.exists())
            self.assertEqual(_e4_calibration_plan(False, 0.005)[3], 0.005)


class E4CalibrationArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.folder.cleanup)
        cls.outputs = {}
        for label, kwargs in (("default", {}), ("explicit_95", {"alpha": 0.05})):
            output = Path(cls.folder.name) / label
            plotting = Mock()
            plotting.subplots.return_value = (
                Mock(), np.array([[Mock(), Mock()], [Mock(), Mock()]], dtype=object)
            )
            with patch("appendix_d_experiments.e45._plot_import", return_value=plotting), \
                    patch("appendix_d_experiments.e45._save"):
                result = run_e4(output, seed=2071, quick=True, **kwargs)
            manifest = json.loads((output / "manifest.json").read_text())
            with np.load(output / "calibration_dataset.npz", allow_pickle=False) as saved:
                data = {key: saved[key].copy() for key in saved.files}
            cls.outputs[label] = (output, result, manifest, data)

    def test_explicit_target_is_used_and_recorded_consistently(self):
        for label, expected_alpha in (("default", 0.10), ("explicit_95", 0.05)):
            with self.subTest(label=label):
                output, result, manifest, data = self.outputs[label]
                self.assertEqual(result["calibration_alpha"], expected_alpha)
                self.assertEqual(manifest["calibration_alpha"], expected_alpha)
                self.assertEqual(float(data["calibration_alpha"]), expected_alpha)
                self.assertEqual(manifest["calibration_target_coverage"], 1 - expected_alpha)
                self.assertEqual(manifest["profile"], "quick")
                with (output / "calibration_bounds.csv").open() as handle:
                    bounds = list(csv.DictReader(handle))
                self.assertTrue(all(float(row["alpha_scene_marginal"]) == expected_alpha
                                    for row in bounds))

                # Independently reconstruct the scene score and selected
                # order statistic from the archived train/calibration split.
                split = data["scene_split"]
                radii = data["required_radius"]
                shape = np.maximum(np.quantile(radii[split == "train"], 0.8, axis=0), 1e-8)
                scores = np.max(radii[split == "calibration"] / shape, axis=(1, 2, 3))
                rank = int(np.ceil((len(scores) + 1) * (1 - expected_alpha)))
                self.assertEqual(manifest["calibration_order"], rank)
                self.assertEqual(int(data["calibration_order"]), rank)
                np.testing.assert_allclose(data["calibration_scores"], scores)
                np.testing.assert_allclose(data["calibrated_rho"], np.sort(scores)[rank - 1] * shape)
                self.assertAlmostEqual(manifest["calibration_scale"], np.sort(scores)[rank - 1])

    def test_stricter_target_keeps_data_fixed_and_cannot_shrink_envelope(self):
        _, _, default_manifest, default = self.outputs["default"]
        _, _, strict_manifest, strict = self.outputs["explicit_95"]
        np.testing.assert_array_equal(default["true_complete_channels"], strict["true_complete_channels"])
        np.testing.assert_array_equal(default["scene_split"], strict["scene_split"])
        self.assertTrue(np.all(strict["calibrated_rho"] >= default["calibrated_rho"]))
        self.assertGreater(strict_manifest["calibration_order"], default_manifest["calibration_order"])
        def failed_scenes(manifest):
            return next(row["failed_scenes"] for row in manifest["scene_coverage"]
                        if row["method"] == "split_calibrated")
        self.assertLessEqual(failed_scenes(strict_manifest), failed_scenes(default_manifest))


if __name__ == "__main__":
    unittest.main()
