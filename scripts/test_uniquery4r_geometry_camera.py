import json
import unittest

import numpy as np

from audit_uniquery4r_movif_geometry_camera import diagnose, normalize_poses, projection, source_intrinsics, substitutions


class CameraAttributionTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(42)
        g = rng.normal(0, .1, (22, 48, 3))
        g[..., 2] += 4
        e = np.tile(np.eye(4)[:3], (22, 1, 1))
        e[:, 0, 3] = np.arange(22)*.01
        k = np.tile(np.array([[280.,0,128], [0,280.,128], [0,0,1]]), (22,1,1))
        return g, e, k

    def test_oracle_and_coupled_scale(self):
        g, e, k = self.fixture()
        pe = e.copy()
        pe[:, :3, 3] /= 3
        uv = projection(g, e, k)
        result = substitutions(g/3, g, pe, e, k, k, 3., uv, np.ones((22,48), bool), 6)
        for groups in result.values():
            self.assertLess(groups['future_visible']['mean'], 1e-11)

    def test_point_only_and_camera_only_error(self):
        g, e, k = self.fixture()
        uv = projection(g,e,k)
        vis = np.ones((22,48),bool)
        p = g.copy()
        p[..., 0] += .1
        point = substitutions(p,g,e,e,k,k,1,uv,vis,6)
        self.assertGreater(point['points_pred_pose_gt_intrinsics_gt']['future_visible']['mean'], 1)
        self.assertLess(point['points_gt_pose_pred_intrinsics_pred']['future_visible']['mean'], 1e-11)
        pe = e.copy()
        pe[:, 0, 3] += .1
        camera = substitutions(g,g,pe,e,k,k,1,uv,vis,6)
        self.assertLess(camera['points_pred_pose_gt_intrinsics_gt']['future_visible']['mean'], 1e-11)
        self.assertGreater(camera['points_gt_pose_pred_intrinsics_pred']['future_visible']['mean'], 1)

    def test_first_reference_and_pixel_resizing(self):
        g, e, k = self.fixture()
        relative = normalize_poses(e)
        np.testing.assert_allclose(relative[0], np.eye(4)[:3])
        m = {'processed_resolution_hw':[252,252], 'source_resolution_hw':[256,256]}
        pk = k.copy()
        pk[:,0,0] *= 252/256
        pk[:,1,1] *= 252/256
        pk[:,:2,2] = (k[:,:2,2]+.5)*252/256-.5
        np.testing.assert_allclose(source_intrinsics(pk,m), k)
        identity = np.eye(4)[None,:3]
        np.testing.assert_allclose(projection(g[:1]/3,identity,k[:1]), projection(g[:1],identity,k[:1]))

    def test_full_diagnosis_serializable_and_scales_correct(self):
        g, e, k = self.fixture()
        pe = e.copy()
        pe[:, :3, 3] /= 3
        uv = projection(g,e,k)
        raw = {'tracks_3d':g/3, 'extrinsic':pe, 'intrinsic':k,
               'source_frame_indices':np.arange(22),
               'tracks_2d_source_pixels':uv,
               'tracks_3d_projected_source_pixels':uv}
        box = np.array([[-1,-1,3], [-1,1,3], [1,-1,3], [1,1,3],
                        [-1,-1,5], [-1,1,5], [1,-1,5], [1,1,5]])
        gt = {'tracks_first_camera':g, 'extrinsic_w2c_gt':e, 'intrinsic_gt':k,
              'source_frame_indices':np.arange(22), 'tracks_2d':uv,
              'visibility':np.ones((22,48),bool),
              'target_bboxes_world':np.tile(box,(22,1,1))}
        m = {'scene_id':'test', 'query_model_frame_index':6,
             'source_resolution_hw':[256,256], 'processed_resolution_hw':[256,256]}
        result = diagnose(raw,gt,m)
        self.assertTrue(result['camera_observed_motion_scale_valid'])
        self.assertAlmostEqual(result['camera_observed_motion_scale'],3.)
        json.dumps(result,allow_nan=False)
        for groups in result['primary_point_t0_scale'].values():
            self.assertLess(groups['future_visible']['mean'],1e-11)


if __name__ == '__main__':
    unittest.main()
