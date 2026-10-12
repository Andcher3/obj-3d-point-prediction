import json
import unittest

import numpy as np

from audit_uniquery4r_focal_backprojection import camera_coordinates, lift_keep_camera_depth, scene_diagnosis
from audit_uniquery4r_movif_geometry_camera import projection


class FocalLiftTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(31)
        p = rng.normal(0,.1,(22,48,3))
        p[...,2] += 4
        e = np.tile(np.eye(4)[:3],(22,1,1))
        angle = np.linspace(0,.2,22)
        e[:,0,0],e[:,0,1] = np.cos(angle),-np.sin(angle)
        e[:,1,0],e[:,1,1] = np.sin(angle),np.cos(angle)
        e[:,0,3] = np.arange(22)*.01
        k = np.tile(np.array([[280.,0,128],[0,280.,128],[0,0,1]]),(22,1,1))
        return p,e,k

    def test_matched_intrinsics_roundtrip(self):
        p,e,k = self.fixture()
        np.testing.assert_allclose(lift_keep_camera_depth(p,e,k,k),p,atol=1e-12)

    def test_mismatched_intrinsics_rescale_xy_keep_z(self):
        p,e,k = self.fixture()
        fit = k.copy()
        fit[:,0,0] *= 1.7
        fit[:,1,1] *= .8
        lifted = lift_keep_camera_depth(p,e,k,fit)
        np.testing.assert_allclose(camera_coordinates(lifted,e),camera_coordinates(p,e)*[1.7,.8,1],atol=1e-12)
        np.testing.assert_allclose(projection(lifted,e,k),projection(p,e,fit),atol=1e-10)

    def test_full_gt_correction_and_future_independence(self):
        g,e,k = self.fixture()
        original_camera = camera_coordinates(g,e)
        original_camera[...,:2] /= [1.7,.8]
        p = np.einsum('tji,tqj->tqi',e[:,:3,:3],original_camera-e[:,None,:3,3])
        uv = projection(g,e,k)
        box = np.array([[-1,-1,3],[-1,1,3],[1,-1,3],[1,1,3],[-1,-1,5],[-1,1,5],[1,-1,5],[1,1,5]])
        gt = {'tracks_first_camera':g,'intrinsic_gt':k,'tracks_2d':uv,
              'visibility':np.ones((22,48),bool),'target_bboxes_world':np.tile(box,(22,1,1))}
        raw = {'tracks_3d':p,'extrinsic':e,'intrinsic':k,
               'tracks_3d_projected_source_pixels':projection(p,e,k)}
        m = {'query_model_frame_index':6,'scene_id':'synthetic',
             'source_resolution_hw':[256,256],'processed_resolution_hw':[256,256],
             'query_uv_source_pixels':uv[6].tolist()}
        result = scene_diagnosis(raw,gt,m)
        np.testing.assert_allclose(result['focal_ratio_xy'],[1.7,.8],atol=1e-12)
        self.assertLess(result['lifted_unscaled_diagnostic']['future']['point_error_scene_units']['mean'],1e-11)
        self.assertGreater(result['original_unscaled_diagnostic']['future']['point_error_scene_units']['mean'],.01)
        gt['tracks_2d'][7:] += 100
        other = scene_diagnosis(raw,gt,m)
        self.assertEqual(result['focal_ratio_xy'],other['focal_ratio_xy'])
        json.dumps(result,allow_nan=False)


if __name__ == '__main__':
    unittest.main()
