import unittest
import numpy as np
from audit_uniquery4r_scene5702_true_focal import audit_2d


class Direct2DAuditTests(unittest.TestCase):
    def fixture(self, shift):
        rng=np.random.default_rng(22)
        query=rng.uniform(.1,.8,(48,2))
        source=(query*251+.5)*(256/252)-.5
        m={'processed_resolution_hw':[252,252],'source_resolution_hw':[256,256],
           'query_model_frame_index':6,'query_uv_source_pixels':source.tolist()}
        pairs=[(6,t) for t in range(22) if t!=6]+[(t,t) for t in range(22)]
        gt=np.tile(source,(22,1,1))
        pred=gt+shift
        normalized=((pred+.5)*(252/256)-.5)/251*2-1
        warp=np.zeros((len(pairs),48,2))
        for t in range(22):
            warp[pairs.index((6,t))]=normalized[t]
        raw={'pair_idx':np.array(pairs),'warp2d':warp,'tracks_2d_source_pixels':pred,
             'query_uv_normalized_01':query,'source_frame_indices':np.arange(22),
             'tracks_3d':np.ones((22,48,3)),'warp3d':np.ones((len(pairs),48,3)),
             'tracks_3d_projected_source_pixels':pred.copy()}
        return raw,m,{'tracks_2d':gt,'visibility':np.ones((22,48),bool)}

    def test_identity_and_roundtrip(self):
        r,m,g=self.fixture(np.array([0.,0.]))
        result=audit_2d(r,m,g)
        self.assertEqual(result['t0_pair'],[6,6])
        self.assertLess(result['direct_2d_t0_error_px']['mean'],1e-12)

    def test_constant_bias_can_coexist_with_correct_motion(self):
        r,m,g=self.fixture(np.array([10.,0.]))
        result=audit_2d(r,m,g)
        self.assertAlmostEqual(result['direct_2d_t0_error_px']['mean'],10.)
        self.assertLess(result['direct_2d_future_visible_displacement_error_px']['mean'],1e-12)

    def test_wrong_saved_conversion_rejected(self):
        r,m,g=self.fixture(np.array([0.,0.]))
        r['tracks_2d_source_pixels'][6,0]+=5
        with self.assertRaises(AssertionError):
            audit_2d(r,m,g)


if __name__=='__main__':
    unittest.main()
