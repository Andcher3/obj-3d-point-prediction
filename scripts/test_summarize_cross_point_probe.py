import unittest

import numpy as np

from summarize_cross_point_probe import METRICS, compare, paired_bootstrap


class PairedSummaryChecks(unittest.TestCase):
    def test_same_errors_have_zero_gain(self):
        x=np.arange(1,41,dtype=float)
        r=paired_bootstrap(x,x,1000)
        self.assertEqual(r['relative_gain'],0)
        self.assertEqual(r['descriptive_95pct_interval'],[0,0])

    def test_uniform_gain_and_matched_resampling(self):
        x=np.arange(1,41,dtype=float)
        r=paired_bootstrap(x,.8*x,1000)
        self.assertAlmostEqual(r['relative_gain'],.2)
        np.testing.assert_allclose(r['two_reader_family_97_5pct_interval'],[.2,.2],atol=1e-12)
        self.assertEqual(r['scene_win_fraction'],1)

    def test_mismatched_scenes_rejected(self):
        with self.assertRaises(ValueError):paired_bootstrap(np.ones(40),np.ones(39))

    def test_worse_neighbor_information_is_negative_gain(self):
        r=paired_bootstrap(np.ones(40),np.ones(40)*1.1,1000)
        self.assertLess(r['relative_gain'],0)
        self.assertEqual(r['scene_win_fraction'],0)

    def test_zero_initialization_does_not_bound_information_value(self):
        rows=[dict(scene=f'scene_{i}', **{m:1.0 for m in METRICS}) for i in range(40)]
        runs=[{'reader':'mlp_128_128','condition':condition,'seed':seed,
               'selected_epoch':-1,'test':{'scenes':rows,'scene_macro':{m:1.0 for m in METRICS}}}
              for condition in ['self_shared_no_neighbors','self_shared_true_neighbors']
              for seed in range(3)]
        result=compare({'results':runs},'mlp_128_128')
        self.assertTrue(result['all_runs_selected_zero_correction_initialization'])
        self.assertEqual(result['decision'],
                         'validation_retained_zero_correction_information_value_unresolved')


if __name__=='__main__':unittest.main()
