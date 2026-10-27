"""Small hand-calculated checks of the new activity-risk accounting."""
import unittest
import numpy as np
from activity_experiments.analysis import episode_metrics
from listening_experiments.config import TimingConfig


class ActivityMetricsTest(unittest.TestCase):
    def test_partition_keeps_silent_users_and_counts_every_exceedance(self):
        record=dict(windows=np.array([1,4]),ue_ids=np.array([0,1,2]),
            J=np.array([[[0,1,1],[0,2,4]]]),
            detected_ue_mask=np.array([[[False,False,True],[False,False,True]]]),
            inr_linear=np.array([[[[1.,.1,.1]],[[.1,1.,.1]]]]),
            sinr_linear=np.array([[[[1.,3.]],[[1.,3.]]]]),
            raw_inr_linear=np.array([1.,.1,1.]),raw_sinr_linear=np.array([3.,15.]))
        result=episode_metrics(record,TimingConfig(),.5)
        np.testing.assert_allclose(result['q_out'],1/3)
        np.testing.assert_allclose(result['group_fraction'],1/3)
        np.testing.assert_allclose(result['group_contributions'][0,0,0],[1/3,0,0])
        np.testing.assert_allclose(result['group_contributions'][0,1,0],[0,1/3,0])
        np.testing.assert_allclose(result['q_miss_all'],1)
        for li,L in enumerate([1,4]):
            self.assertAlmostEqual(result['q_data'][0,li,0],(L*2/3+(256-2.1*L)/3)/(256-1.1*L))
            self.assertAlmostEqual(result['r_eff'][0,li,0],(L*3+(256-2.1*L)*1.5)/256)

    def test_timing_counts_no_listening_silence_as_protection(self):
        d=TimingConfig().durations(64)
        self.assertAlmostEqual(d['rf'],70.4)
        self.assertAlmostEqual(d['mrt']/d['data'],64/185.6)
        self.assertAlmostEqual(d['mrt']+d['final'],d['data'])


if __name__=='__main__':
    unittest.main()
