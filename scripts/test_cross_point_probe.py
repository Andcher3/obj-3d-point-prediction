"""CPU synthetic checks only; no real-scene fitting or model forward."""
import unittest
import numpy as np
from cross_point_probe_features import features,training_standardizer,standardize,check_cache


class ProbeChecks(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(12)
        self.obs=self.rng.normal(size=(7,48,3))
        self.baseline=self.rng.normal(size=(15,48,3))
        self.latent=self.rng.normal(size=(15,48,16))
        self.uv=self.rng.uniform(0,256,size=(48,2))

    def test_matched_dimensions_and_common_features(self):
        a,ma=features(self.obs,self.baseline,self.latent,self.uv,False)
        b,mb=features(self.obs,self.baseline,self.latent,self.uv,True)
        self.assertEqual(a.shape,b.shape)
        width=ma['self_feature_width']
        np.testing.assert_array_equal(a[:,:width],b[:,:width])
        self.assertFalse(a[:,width:].any())
        self.assertTrue(b[:,width:].any())
        self.assertEqual(ma['neighbor_feature_width'],8*7*3+1)

    def test_query_permutation(self):
        perm=self.rng.permutation(48)
        x,_=features(self.obs,self.baseline,self.latent,self.uv)
        y,_=features(self.obs[:,perm],self.baseline[:,perm],self.latent[:,perm],self.uv[perm])
        np.testing.assert_allclose(x,y[np.argsort(perm)],atol=1e-13)

    def test_global_translation_invariance(self):
        a,_=features(self.obs,self.baseline,self.latent,self.uv)
        b,_=features(self.obs+[5,3,-2],self.baseline+[5,3,-2],self.latent,self.uv)
        np.testing.assert_allclose(a,b,atol=1e-13)

    def test_no_self_neighbors(self):
        _,m=features(self.obs,self.baseline,self.latent,self.uv)
        self.assertFalse((m['neighbors']==np.arange(48)[:,None]).any())

    def test_standardizer_train_only_and_constant_channels(self):
        x,_=features(self.obs,self.baseline,self.latent,self.uv,False)
        mean,std=training_standardizer([x])
        self.assertTrue(np.isfinite(standardize(x,mean,std)).all())
        before=mean.copy(); standardize(x+100,mean,std)
        np.testing.assert_array_equal(before,mean)

    def test_leakage_and_time_boundary_rejected(self):
        cache={'observed_xyz':self.obs,'baseline_future':self.baseline,'latent_future':self.latent,
            'query_uv':self.uv,'timestamps_s':np.arange(22)/12,'coordinate_frame':'last_observed_camera',
            'future_rgb_supplied':False,'gt_input_supplied':False}
        x,_=check_cache(cache); self.assertEqual(x.shape[0],48)
        cache['gt_input_supplied']=True
        with self.assertRaises(ValueError): check_cache(cache)
        cache['gt_input_supplied']=False; cache['observed_xyz']=self.obs[:6]
        with self.assertRaises(ValueError): check_cache(cache)

    def test_training_rejects_unaudited_coordinate_scale(self):
        import tempfile
        from pathlib import Path
        from run_cross_point_probe import load_rows
        cache={'observed_xyz':self.obs,'baseline_future':self.baseline,'latent_future':self.latent,
            'query_uv':self.uv,'timestamps_s':np.arange(22)/12,'coordinate_frame':'last_observed_camera',
            'future_rgb_supplied':False,'gt_input_supplied':False}
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            np.savez(folder/'synthetic.npz',**cache)
            with self.assertRaisesRegex(ValueError,'coordinate and scale audit pending'):
                load_rows(folder,folder,['synthetic'],'self_shared_true_neighbors',
                          {'frame_protocol':{'observed_frames':7,'frames':22}})


if __name__=='__main__': unittest.main()
