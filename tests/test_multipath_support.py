"""Physical multipath synthesis, coherent DOA recovery, NLOS and regression checks."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import multipath_support as mp
import ntn_music_detection as nmd
import nulling_cdf_utils as ncu


def positions(n=8):
    y, z = np.meshgrid(np.arange(n)*.5, np.arange(n)*.5)
    p = np.column_stack((np.zeros(n*n), y.ravel(), z.ravel()))
    return p-p.mean(axis=0)


def path_fixture(ratio=1., los=True):
    p = positions()*ratio
    angles = [(20., 80.), (45., 105.)]
    u = np.array([nmd.array_position_steering_global(ph, th, p).ravel() for ph, th in angles])
    alpha = np.array([1e-5, 7e-6*np.exp(.7j)])/ratio
    a = (alpha[:, None]*u).T.reshape(1, 1, 1, 64, 2, 1)
    valid = np.ones((1, 1, 1, 64, 2), bool)
    interaction = np.array([0 if los else 1, 1]).reshape(1, 1, 1, 2)
    tau = np.array([1e-6, 1.1e-6]).reshape(1, 1, 2)
    paths = SimpleNamespace(
        valid=valid, interactions=interaction, tau=tau,
        phi_t=np.deg2rad(np.array([20.,45.])).reshape(1,1,2),
        theta_t=np.deg2rad(np.array([80.,105.])).reshape(1,1,2))
    return paths, a, tau


def music_kwargs(**overrides):
    kw = dict(tx_rows=8, tx_cols=8, nsect=1, array_positions_local=positions(),
              tx_orientations_rad=np.zeros((1,3)), detect_covariance_mode='analytic',
              detect_noise_var=1e-13, detect_source_estimation='rank',
              phi_grid_deg=np.arange(0,90,1.), theta_grid_deg=np.arange(60,120,1.),
              covariance_refine=False, peak_refine=True)
    kw.update(overrides)
    return kw


class CoherentToyScene:
    fc = 7e9
    tx_pos = np.array([[0.,0.,30.]])
    ntn_look_pos = np.array([0.,0.,1e6])
    tn_pos = np.array([[100.,-20.,2.]])
    rx_ntn_pos = np.array([[100.,70.,2.]])
    tx_orientation_rad = np.zeros((1,3))

    def __init__(self):
        self.position_calls = self.dl_calls = 0
    def compute_positions(self, **kw):
        self.position_calls += 1
    def compute_paths(self, **kw):
        self.dl_calls += 1
        self.paths_ntn, self.a_ntn, self.tau_ntn = path_fixture(los=kw.get('ntn_los_mode') != 'nlos_only')
        u = nmd.array_position_steering_global(5,90,positions()).ravel()
        self.a_tn = (2e-5*u).reshape(1,1,1,64,1,1)
        self.tau_tn = np.zeros((1,1,1))
    def compute_ntn_ul_paths(self, frequency_hz, **kw):
        return path_fixture(frequency_hz/self.fc)


class MultipathTests(unittest.TestCase):
    def test_static_cir_matches_legacy_sum_exactly(self):
        rng=np.random.default_rng(1)
        a=rng.normal(size=(3,2,4,8,5,1))+1j*rng.normal(size=(3,2,4,8,5,1))
        np.testing.assert_array_equal(mp.collapse_cir_to_narrowband(a), np.sum(a,axis=(4,5)))
        self.assertIs(nmd.collapse_cir_to_narrowband, mp.collapse_cir_to_narrowband)

    def test_coherent_cancellation_not_power_sum(self):
        a=np.array([1.,-1.],complex).reshape(1,1,1,1,2,1)
        self.assertEqual(mp.collapse_cir_to_narrowband(a).item(),0)
        self.assertEqual(np.sum(np.abs(a)**2),2)

    def test_time_axis_is_selected_or_preserved_never_summed(self):
        a=np.array([[1,2,3],[4,5,6]],complex).reshape(1,1,1,1,2,3)
        self.assertEqual(mp.collapse_cir_to_narrowband(a).item(),5)
        self.assertEqual(mp.collapse_cir_to_narrowband(a,time_index=2).item(),9)
        np.testing.assert_array_equal(mp.collapse_cir_to_narrowband(a,time_index=None).ravel(),[5,7,9])
        with self.assertRaises(IndexError):mp.collapse_cir_to_narrowband(a,time_index=3)
        with self.assertRaises(ValueError):mp.collapse_cir_to_narrowband(a,time_index=1.2)

    def test_frequency_offset_uses_delay_and_does_not_duplicate_carrier_phase(self):
        a=np.array([1,1j],complex).reshape(1,1,1,1,2,1)
        tau=np.array([0.,.25]).reshape(1,1,2)
        self.assertEqual(mp.collapse_cir_to_narrowband(a,tau=tau).item(),1+1j)
        self.assertAlmostEqual(mp.collapse_cir_to_narrowband(a,tau=tau,frequency_offset_hz=1).item(),2)
        with self.assertRaises(ValueError):mp.collapse_cir_to_narrowband(a,frequency_offset_hz=1)

    def test_empty_and_invalid_paths(self):
        a=np.zeros((2,1,3,8,0,1),complex)
        np.testing.assert_array_equal(mp.collapse_cir_to_narrowband(a), np.zeros((2,1,3,8)))
        a=np.array([2,np.nan],complex).reshape(1,1,1,1,2,1)
        self.assertEqual(mp.collapse_cir_to_narrowband(a,valid_mask=np.array([True,False])).item(),2)
        with self.assertRaises(ValueError):mp.collapse_cir_to_narrowband(a)
        with self.assertRaises(ValueError):mp.collapse_cir_to_narrowband(np.ones((2,2,2)))

    def test_zero_disables_all_multipath_options(self):
        tn,ntn=mp.resolve_propagation_options(0,2,'nlos_only',dict(diffraction=True,seed=1))
        expected=dict(max_depth=0,los=True,specular_reflection=True,
                      diffuse_reflection=False,refraction=True,synthetic_array=True)
        self.assertEqual(tn,expected);self.assertEqual(ntn,expected)

    def test_nlos_policy_does_not_remove_tn_los(self):
        tn,ntn=mp.resolve_propagation_options(1,2,'nlos_only',dict(seed=123,diffraction=True))
        self.assertTrue(tn['los']);self.assertFalse(ntn['los'])
        self.assertEqual(ntn['seed'],123);self.assertEqual(ntn['max_depth'],2)
        with self.assertRaises(ValueError):mp.resolve_propagation_options(1,0,'natural')
        with self.assertRaises(ValueError):mp.resolve_propagation_options(1,2,'natural',dict(los=False))

    def test_smoothing_restores_two_coherent_directions_and_full_array_powers(self):
        _,a,_=path_fixture()
        h=mp.collapse_cir_to_narrowband(a)
        for mode in ('raw','conj'):
            out=mp.run_multipath_music_pipeline(h,music_kwargs=music_kwargs(channel_mode=mode))
            np.testing.assert_array_equal(out['raw_signal_rank'],[1])
            np.testing.assert_array_equal(out['num_sources_record'],[2])
            np.testing.assert_allclose(out['peak_phi_hat_deg'],[20,45],atol=1e-5)
            np.testing.assert_allclose(out['peak_theta_hat_deg'],[80,105],atol=1e-5)
            np.testing.assert_allclose(out['peak_g_hat'],[1e-10,4.9e-11],rtol=1e-6)
            self.assertEqual(out['peak_u_hat_raw'].shape,(2,64))
            self.assertGreater(abs(out['correlated_source_covariance'][0,0,1]),6e-11)
            self.assertLess(out['covariance_fit_after'][0],1e-6)

    def test_smoothed_fdd_transfer_and_display_angles(self):
        true_vectors=np.array([nmd.array_position_steering_global(ph,th,positions()).ravel()
                               for ph,th in ((20,80),(45,105))])
        for ratio in (.95,1.05):
            paths,a,tau=path_fixture(ratio)
            kw=music_kwargs(array_positions_local=positions()*ratio,
                            phi_offset_deg=13.,phi_mirror_about_sector=True)
            out=mp.run_multipath_music_pipeline(mp.collapse_cir_to_narrowband(a),music_kwargs=kw)
            transferred=ncu.transfer_music_peaks_to_dl(out,dl_array_positions=positions(),music_kwargs=kw,num_tx=1)
            np.testing.assert_allclose(np.abs(np.sum(transferred['peak_u_hat_raw']*true_vectors.conj(),axis=1)),
                                       [1,1],atol=1e-7)
            report=mp.summarize_path_estimates(mp.build_path_catalog(paths,a,tau),out)
            self.assertEqual(report['matched_paths'],2)

    def test_empty_path_catalog(self):
        paths=SimpleNamespace(valid=np.empty((1,1,1,64,0),bool),interactions=np.empty((1,1,1,0),int),
                              tau=np.empty((1,1,0)),phi_t=np.empty((1,1,0)),theta_t=np.empty((1,1,0)))
        catalog=mp.build_path_catalog(paths,np.empty((1,1,1,64,0,1),complex))
        out=mp.run_multipath_music_pipeline(np.zeros((1,1,1,64)),music_kwargs=music_kwargs())
        report=mp.summarize_path_estimates(catalog,out)
        self.assertEqual(report['valid_paths'],0)
        self.assertEqual(report['no_path_links'],1)
        self.assertEqual(catalog['path_channels'].shape,(0,1,1,64))

    def test_correlated_fit_preserves_nonzero_cross_terms(self):
        _,a,_=path_fixture();h=mp.collapse_cir_to_narrowband(a).ravel()
        u=a[0,0,0,:,:,0].T.copy();u/=np.linalg.norm(u,axis=1,keepdims=True)
        r=np.outer(h,h.conj())+1e-13*np.eye(64)
        q,g,residual,_=mp.fit_correlated_path_covariance(r,u,1e-13)
        self.assertGreater(abs(q[0,1]),6e-11)
        np.testing.assert_allclose(u.T@q@u.conj(),np.outer(h,h.conj()),rtol=1e-12,atol=1e-24)
        self.assertLess(residual,1e-12)

    def test_smoothing_is_independent_of_flatten_order_and_preserves_noise(self):
        p=positions();rng=np.random.default_rng(9);order=rng.permutation(64)
        indices,sub=mp.planar_subarrays(p[order],6,6)
        self.assertEqual(indices.shape,(9,36))
        for idx in indices:
            np.testing.assert_allclose(p[order][idx]-p[order][idx].mean(axis=0),sub)
        np.testing.assert_allclose(mp.spatially_smooth_covariance(1e-13*np.eye(64),indices),1e-13*np.eye(36))
        with self.assertRaises(ValueError):mp.planar_subarrays(p,9,6)

    def test_sample_mode_is_reproducible_and_uses_shared_ue_waveform(self):
        _,a,_=path_fixture();h=mp.collapse_cir_to_narrowband(a)
        kw=music_kwargs(detect_covariance_mode='sample',detect_num_snapshots=2000,
                        detect_rng_seed=321,detect_num_sources=2,detect_noise_var=0.)
        one=mp.run_multipath_music_pipeline(h,music_kwargs=kw)
        two=mp.run_multipath_music_pipeline(h,music_kwargs=kw)
        np.testing.assert_array_equal(one['ul_covariance'],two['ul_covariance'])
        self.assertEqual(np.linalg.matrix_rank(one['ul_covariance'][0],tol=1e-18),1)
        self.assertEqual(one['peak_u_hat_raw'].shape,(2,64))

    def test_empty_signal_and_explicit_unsmoothed_comparison(self):
        out=mp.run_multipath_music_pipeline(np.zeros((1,1,1,64),complex),music_kwargs=music_kwargs())
        self.assertEqual(out['peak_u_hat_raw'].shape,(0,64))
        _,a,_=path_fixture();h=mp.collapse_cir_to_narrowband(a)
        out=mp.run_multipath_music_pipeline(h,music_kwargs=music_kwargs(),spatial_smoothing=False)
        np.testing.assert_array_equal(out['num_sources_record'],[1])

    def test_top_k_and_energy_selection_do_not_change_covariance(self):
        _,a,_=path_fixture();h=mp.collapse_cir_to_narrowband(a)
        all_out=mp.run_multipath_music_pipeline(h,music_kwargs=music_kwargs())
        top=mp.run_multipath_music_pipeline(h,music_kwargs=music_kwargs(),top_k=1)
        self.assertEqual(len(top['peak_t_idx']),1)
        np.testing.assert_array_equal(top['ul_covariance'],all_out['ul_covariance'])
        np.testing.assert_array_equal(mp.select_estimated_paths([6,3,1],energy_fraction=.8),[0,1])
        np.testing.assert_array_equal(mp.select_estimated_paths([6,3,1],energy_fraction=.8,top_k=1),[0])

    def test_catalog_keeps_all_paths_and_classifies_los_nlos_links(self):
        paths,a,tau=path_fixture();catalog=mp.build_path_catalog(paths,a,tau)
        self.assertEqual(catalog['valid'].sum(),2);self.assertEqual(catalog['is_los'].sum(),1)
        self.assertEqual(catalog['path_channels'].shape,(2,1,1,64))
        out=mp.run_multipath_music_pipeline(mp.collapse_cir_to_narrowband(a),music_kwargs=music_kwargs())
        report=mp.summarize_path_estimates(catalog,out)
        self.assertEqual(report['matched_paths'],2);self.assertEqual(report['los_links'],1)
        self.assertAlmostEqual(report['matched_path_power_fraction'],1)
        paths.interactions[:]=1
        report=mp.summarize_path_estimates(mp.build_path_catalog(paths,a,tau),out)
        self.assertEqual(report['nlos_links'],1);self.assertEqual(report['los_links'],0)

    def test_covariance_override_validation(self):
        h=np.zeros((1,1,4),complex)
        for bad in (np.eye(3), -np.eye(4), np.ones((4,4))*np.nan, np.triu(np.ones((4,4)))):
            with self.assertRaises(ValueError):nmd.detect_ntn_music_from_hi(h,covariance_override=bad)

    def test_full_fdd_loop_keeps_truth_channels_and_saves_all_rays(self):
        scene=CoherentToyScene()
        with tempfile.TemporaryDirectory() as tmp, patch.object(ncu,'_scene_tx_array_positions_local',return_value=positions()):
            out=ncu.run_nulling_cdf_experiment(
                scene,num_macro_sims=1,compute_positions_kwargs=dict(azimuth=0.,elevation=40.),
                compute_paths_kwargs=dict(fc=7e9,max_depth=1),multipath=1,
                multipath_music_kwargs=dict(top_k=1),ul_frequency_percentages=[-5,0,5],
                h_tn_th=0.,tx_antennas=64,tx_power=1.,snr_noise_power=1e-13,inr_noise_power=1e-13,
                lambda_ranges_music_est=[1e10,1e11],lambda_ranges_music_real=[1e10],
                music_kwargs=music_kwargs(),show_progress=False,print_music_u_corr=False,
                channel_cache_dir=Path(tmp)/'channels')
            self.assertEqual((scene.position_calls,scene.dl_calls),(1,1))
            self.assertEqual(out['num_estimated_curves'],6)
            reference=out['by_percentage'][0]['raw_inr_db']
            for case in out['by_percentage'].values():
                np.testing.assert_array_equal(case['raw_inr_db'],reference)
                self.assertEqual(case['macro_stats'][0]['path_metrics_dl']['valid_paths'],2)
                self.assertEqual(case['macro_stats'][0]['path_metrics_dl']['estimated_paths'],1)
                self.assertEqual(case['est_inr_db'][1e10].size,1)
                self.assertEqual(case['oracle_model'],'true_dl_paths')
            with np.load(Path(tmp)/'channels/paths_dl_0000.npz') as archive:
                np.testing.assert_array_equal(archive['a_ntn'],scene.a_ntn)
            file=ncu.save_experiment_metrics(out,result_dir=tmp)
            with np.load(file,allow_pickle=False) as archive:
                self.assertEqual(archive['multipath'],1)
                report=json.loads(archive['ul_000/path_metrics_dl_json'][0])
                self.assertEqual(report['valid_paths'],2)
                self.assertIn('ul_000/est_inr_db_index_000',archive)


if __name__=='__main__':unittest.main()
