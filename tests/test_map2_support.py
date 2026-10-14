"""Check map power conventions and boundaries against the actual TN sampler."""
import unittest
import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np

from map2_support import aggregate_inr, plot_ntn_inr_cdf, sector_boundary_segments
from tn_sector_drop import SectorTNSampler


class Map2Checks(unittest.TestCase):
    def test_cdf_retains_zero_and_worsened_ues_in_both_groups(self):
        snapshot = dict(inr_before=np.array([100., 0., 1., 10.]),
                        inr_after=np.array([1., 0., 10., 1.]), detected_idx=np.array([0, 2]))
        with TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            figure = plot_ntn_inr_cdf(snapshot, directory)
            try:
                summary = json.loads((Path(directory)/"map2_ntn_inr_cdf_summary.json").read_text())
                self.assertEqual(summary["detected"]["count"], 2)
                self.assertEqual(summary["detected"]["improved_count"], 1)
                self.assertEqual(summary["detected"]["worsened_count"], 1)
                self.assertEqual(summary["detected"]["median_paired_suppression_db"], 5.)
                self.assertEqual(summary["all"]["count"], 4)
                self.assertEqual(summary["all"]["zero_before_count"], 1)
                rows = np.loadtxt(Path(directory)/"map2_ntn_inr_cdf.csv", delimiter=",", skiprows=1)
                np.testing.assert_array_equal(rows[:, 1], [1, 0, 1, 0])
                np.testing.assert_array_equal(rows[:, 2], snapshot["inr_before"])
                np.testing.assert_array_equal(rows[:, 3], snapshot["inr_after"])
                for axis, expected_count in zip(figure.axes, (2, 4)):
                    for line in axis.lines:
                        self.assertEqual(len(line.get_ydata()), expected_count+1)
                        self.assertEqual(line.get_ydata()[-1], 1.)
            finally:
                plt.close(figure)

    def test_complex_beam_and_incoherent_sector_sum(self):
        # Opposite sectors' signals must not cancel; each antenna beam must use v^H h.
        h = np.array([[[[1., 1j], [1., 1j]]]])
        beams = np.array([[1., 1j], [-1., -1j]]) / np.sqrt(2)
        np.testing.assert_allclose(aggregate_inr(h, beams, 3., 2.), [6.])

    def test_boundaries_match_rotated_sampler_and_stop_at_own_cell(self):
        axis = np.arange(-10., 11.)
        zero = np.zeros((len(axis), len(axis)))
        bs = np.array([[-5,-5,2], [5,-5,2], [-5,5,2], [5,5,2]], dtype=float)
        yaw = .31
        sampler = SectorTNSampler(axis, axis, zero.astype(bool), zero, zero, bs,
            outdoor_probability=1., yaw_offset_rad=yaw, rng=np.random.default_rng(3))
        segments = sector_boundary_segments(bs, sampler.rectangles, yaw)
        self.assertEqual(len(segments), 12)
        for b, segment in segments:
            xmin, xmax, ymin, ymax = sampler.rectangles[b]
            np.testing.assert_allclose(segment[0], bs[b,:2])
            endpoint = segment[1]
            self.assertTrue(xmin-1e-10 <= endpoint[0] <= xmax+1e-10)
            self.assertTrue(ymin-1e-10 <= endpoint[1] <= ymax+1e-10)
            self.assertTrue(any(np.isclose(endpoint[k], edge) for k,edge in
                ((0,xmin),(0,xmax),(1,ymin),(1,ymax))))
            direction = endpoint-segment[0]
            angle = np.arctan2(direction[1], direction[0])
            centers = yaw + np.arange(3)*2*np.pi/3
            distances = np.sort(np.abs(np.angle(np.exp(1j*(angle-centers)))))
            np.testing.assert_allclose(distances[:2], [np.pi/3,np.pi/3], atol=1e-12)
            # Grid cells immediately on either side must belong to different sectors.
            normal = np.array([-direction[1],direction[0]]) / np.linalg.norm(direction)
            midpoint = segment.mean(axis=0)
            targets = []
            for side in (-1,1):
                p = midpoint + side*1.1*normal
                ix, iy = np.argmin(abs(axis-p[0])), np.argmin(abs(axis-p[1]))
                targets.append(sampler.region_grid[iy,ix])
            self.assertNotEqual(targets[0],targets[1])
            self.assertTrue(all(t//3 == b for t in targets))


if __name__ == "__main__":
    unittest.main()
