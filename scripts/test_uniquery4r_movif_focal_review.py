import unittest
import numpy as np
from render_uniquery4r_movif_focal_review import diagnostic, fit, rays, project


class FocalDiagnosticTests(unittest.TestCase):
    def test_known_focal_recovered_without_xyz_changes(self):
        rng = np.random.default_rng(19)
        xyz = rng.uniform(-.3, .3, (22, 48, 3))
        xyz[..., 2] += 2
        original_xyz = xyz.copy()
        e = np.tile(np.eye(4)[:3], (22, 1, 1))
        k = np.tile(np.array([[150., 0, 126], [0, 170., 126], [0, 0, 1]]), (22, 1, 1))
        truth_k = k.copy()
        truth_k[:, 0, 0], truth_k[:, 1, 1] = 270., 290.
        m = dict(processed_resolution_hw=[252, 252], source_resolution_hw=[256, 256],
                 query_model_frame_index=6)
        expected = project(rays(xyz, e), truth_k, m)
        m['query_uv_source_pixels'] = expected[6].tolist()
        gt_k = truth_k.copy()
        gt_k[:, 0, 0] *= 256/252
        gt_k[:, 1, 1] *= 256/252
        old, new, result = diagnostic(xyz, e, k, m, expected, np.ones((22, 48), bool), gt_k)
        np.testing.assert_allclose(new, expected, atol=1e-11)
        np.testing.assert_array_equal(xyz, original_xyz)
        np.testing.assert_allclose(result['focal_t0_refit_processed_px'], [270, 290], atol=1e-11)
        self.assertLess(result['heldout_t0_query_error_px']['mean'], 1e-10)
        self.assertGreater(result['original']['future_visible_error_px']['mean'], 1)

    def test_future_gt_cannot_change_fit(self):
        rng = np.random.default_rng(20)
        xyz = rng.uniform(.1, 1., (22, 48, 3))
        xyz[..., 2] += 2
        e = np.tile(np.eye(4)[:3], (22, 1, 1))
        k = np.tile(np.array([[150., 0, 126], [0, 170., 126], [0, 0, 1]]), (22, 1, 1))
        m = dict(processed_resolution_hw=[252, 252], source_resolution_hw=[256, 256], query_model_frame_index=6)
        expected = project(rays(xyz, e), k, m)
        m['query_uv_source_pixels'] = expected[6].tolist()
        vis = np.ones((22, 48), bool)
        _, new1, r1 = diagnostic(xyz, e, k, m, expected, vis, k)
        expected[7:] += 1000
        _, new2, r2 = diagnostic(xyz, e, k, m, expected, vis, k*1.2)
        np.testing.assert_array_equal(new1, new2)
        self.assertEqual(r1['fixed_focal_ratio_xy'], r2['fixed_focal_ratio_xy'])

    def test_degenerate_and_negative_focal_rejected(self):
        with self.assertRaises(ValueError):
            fit(np.zeros((48, 2)), np.zeros((48, 2)), np.zeros(2))
        with self.assertRaises(ValueError):
            fit(np.ones((48, 2)), -np.ones((48, 2)), np.zeros(2))


if __name__ == '__main__':
    unittest.main()
