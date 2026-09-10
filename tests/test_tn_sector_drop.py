"""Tests for fixed-sector TN placement and channel-power dominance."""
import ast
import json
import unittest
import tempfile
from unittest.mock import patch
from pathlib import Path
import numpy as np
from tn_sector_drop import SectorTNSampler, channel_dominance


class SectorDropTests(unittest.TestCase):
    def sampler(self, p=1., seed=12, yaw=0., attempts=8):
        x = np.linspace(-100, 100, 41)
        y = np.linspace(-80, 80, 33)
        row, col = np.indices((len(y), len(x)))
        building = (row + col) % 2 == 0
        ground = 10. + row * .1
        roof = ground + 15.
        bs = np.array([[-50,-40,45], [50,-40,45], [-50,40,45], [50,40,45]])
        return SectorTNSampler(x,y,building,ground,roof,bs,
                               outdoor_probability=p, yaw_offset_rad=yaw,
                               rng=np.random.default_rng(seed), max_attempts=attempts)

    def test_rectangles_sectors_and_heights(self):
        for p in (0., 1.):
            s=self.sampler(p)
            positions, labels=s.draw(np.arange(12), 3)
            self.assertEqual(len(positions),36)
            for pos,t in zip(positions, labels):
                ix=np.searchsorted(s.x,pos[0]); iy=np.searchsorted(s.y,pos[1])
                self.assertEqual(s.region_grid[iy,ix],t)
                self.assertEqual((iy+ix)%2==0, p==0)
                self.assertAlmostEqual(pos[2],10+iy*.1+(13.5 if p==0 else 1.8))
                xmin,xmax,ymin,ymax=s.rectangles[t//3]
                self.assertTrue(xmin <= pos[0] <= xmax and ymin <= pos[1] <= ymax)

    def test_random_type_stays_fixed_and_retries_do_not_repeat(self):
        s=self.sampler(.4,attempts=8)
        types=s.is_outdoor.copy()
        a,ta=s.draw(np.arange(12),4)
        b,tb=s.draw(np.arange(12),8)
        np.testing.assert_array_equal(s.is_outdoor,types)
        self.assertEqual(len(a),48)
        self.assertEqual(len(b),48)
        self.assertEqual(len(set(map(tuple,np.vstack((a,b))))),96)
        self.assertEqual(s.exhausted(np.arange(12)),list(range(12)))
        empty,_=s.draw(np.arange(12))
        self.assertEqual(empty.shape,(0,3))

    def test_seed_and_global_rng_isolation(self):
        state=np.random.get_state()
        a=self.sampler(.4); b=self.sampler(.4)
        np.testing.assert_array_equal(a.is_outdoor,b.is_outdoor)
        np.testing.assert_array_equal(a.draw(np.arange(12))[0],b.draw(np.arange(12))[0])
        np.testing.assert_array_equal(state[1],np.random.get_state()[1])

    def test_sector_yaw_rotates_partition(self):
        a=self.sampler(yaw=0.)
        b=self.sampler(yaw=2*np.pi/3)
        mask=a.region_grid>=0
        np.testing.assert_array_equal(a.region_grid[mask]//3,b.region_grid[mask]//3)
        np.testing.assert_array_equal((a.region_grid[mask]%3-1)%3,b.region_grid[mask]%3)

    def test_missing_selected_type_fails(self):
        with self.assertRaisesRegex(ValueError, "No indoor"):
            SectorTNSampler([-2,-1,0,1,2],[-2,-1,0,1,2],np.zeros((5,5),bool),
                            np.zeros((5,5)),np.ones((5,5))*5,
                            [[-1,-1,35],[1,-1,35],[-1,1,35],[1,1,35]],
                            outdoor_probability=0.,yaw_offset_rad=0.,
                            rng=np.random.default_rng(0))

    def test_invalid_parameters(self):
        for p in (-.1,1.1,np.nan):
            with self.assertRaises(ValueError):
                self.sampler(p)
        for n in (0,1.5,True):
            with self.assertRaises(ValueError):
                self.sampler(attempts=n)

    def test_margin_is_power_db_and_includes_sibling_sectors(self):
        h=np.zeros((1,1,12,1),complex)
        h[0,0,0,0]=2e-6
        h[0,0,1,0]=1e-6
        ok,margin,_,_=channel_dominance(h,[0],6.,0.)
        self.assertTrue(ok[0])
        self.assertAlmostEqual(margin[0],20*np.log10(2))
        self.assertFalse(channel_dominance(h,[0],7.,0.)[0][0])
        self.assertFalse(channel_dominance(h,[0],0.,3e-6)[0][0])
        h[0,0,11,0]=3e-6
        self.assertFalse(channel_dominance(h,[0],0.,0.)[0][0])

    def test_zero_ties_nonfinite_and_scaling(self):
        h=np.zeros((1,1,12,1),complex)
        self.assertFalse(channel_dominance(h,[0],0.,0.)[0][0])
        h[0,0,0,0]=1e-5
        self.assertTrue(channel_dominance(h,[0],100.,0.)[0][0])
        h[0,0,1,0]=1e-5
        self.assertFalse(channel_dominance(h,[0],0.,0.)[0][0])
        h[0,0,1,0]=2e-6
        a=channel_dominance(h,[0],3.,0.)
        b=channel_dominance(h*1e-3,[0],3.,0.)
        np.testing.assert_allclose(a[1],b[1])
        h[0,0,11,0]=np.nan
        self.assertFalse(channel_dominance(h,[0],0.,0.)[0][0])

    def test_copied_notebook_configuration_and_syntax(self):
        nb=json.loads(Path("Nulling_CDF_SectorDrop.ipynb").read_text())
        source=""
        for i,cell in enumerate(nb["cells"]):
            if cell["cell_type"]=="code":
                text="".join(cell["source"]); ast.parse(text)
                source+=text
                self.assertEqual(cell["outputs"],[])
        self.assertIn("SceneConfigSionnaSectorDrop",source)
        self.assertIn("from SceneConfigSionnaSectorDrop import SceneConfigSionna\n",source)
        self.assertIn("tn_rx = 12",source)
        self.assertIn("tn_outdoor_probability=tn_outdoor_probability",source)
        self.assertIn("tn_min_channel_norm=h_tn_th",source)
        self.assertNotIn("tn_building_ratio=0.6",source)
        self.assertIn("pair_keys=None",source)
        ast.parse(Path("SceneConfigSionnaSectorDrop.py").read_text())

    def scene_fixture(self, attempts=8, probability=.4):
        from SceneConfigSionnaSectorDrop import SceneConfigSionna
        scene = object.__new__(SceneConfigSionna)
        scene.tn_sampler = self.sampler(probability, attempts=attempts)
        scene.tn_pos, scene.tn_target_tx_index = scene.tn_sampler.draw(np.arange(12))
        scene.tn_outdoor_probability = probability
        scene.tn_association_margin_db = 3.
        scene.tn_min_channel_norm = 1e-6
        scene.tn_candidates_per_batch = 2
        scene.tn_drop_output_dir = None
        scene._tn_drop_macro_index = 0
        return scene

    def test_scene_retry_keeps_targets_types_and_final_arrays_aligned(self):
        scene = self.scene_fixture()
        initial_types = scene.tn_sampler.is_outdoor.copy()
        calls = []

        def trace(*args):
            calls.append(scene.tn_target_tx_index.copy())
            h = np.zeros((len(scene.tn_pos), 1, 12, 1), complex)
            for i, target in enumerate(scene.tn_target_tx_index):
                h[i, 0, target, 0] = 1e-5 if len(calls) > 1 or target % 2 == 0 else 0
            return object(), h, np.zeros(1)

        with patch.object(scene, "_trace_tn_candidates", side_effect=trace):
            scene._compute_sector_tn_paths(None, 0)
        np.testing.assert_array_equal(calls[1], np.repeat([1,3,5,7,9,11], 2))
        np.testing.assert_array_equal(calls[-1], np.arange(12))
        np.testing.assert_array_equal(scene.tn_is_outdoor, initial_types)
        self.assertEqual(scene.tn_pos.shape, (12, 3))
        self.assertEqual(scene.a_tn.shape, (12, 1, 12, 1))
        self.assertEqual(scene.tn_drop_diagnostics["status"], "success")

    def test_scene_failure_is_bounded_and_writes_diagnostics(self):
        scene = self.scene_fixture(attempts=1, probability=0.)
        h = np.zeros((12, 1, 12, 1), complex)
        with tempfile.TemporaryDirectory() as directory:
            scene.tn_drop_output_dir = directory
            with patch.object(scene, "_trace_tn_candidates", return_value=(None,h,None)) as trace:
                with self.assertRaisesRegex(RuntimeError, "No feasible sector TN"):
                    scene._compute_sector_tn_paths(None, 0)
                self.assertEqual(trace.call_count, 1)
            report = json.loads((Path(directory)/"drop_0000.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertFalse(any(report["is_outdoor"]))
            self.assertFalse(any(report["accepted"]))

    def test_final_retrace_must_pass_again(self):
        scene = self.scene_fixture()
        good = np.eye(12).reshape(12,1,12,1).astype(complex) * 1e-5
        with patch.object(scene, "_trace_tn_candidates", side_effect=[
                (None,good,None), (None,np.zeros_like(good),None)]):
            with self.assertRaisesRegex(RuntimeError, "Final TN channel"):
                scene._compute_sector_tn_paths(None, 0)
        self.assertEqual(scene.tn_drop_diagnostics["status"], "final_validation_failed")


if __name__ == "__main__":
    unittest.main()
