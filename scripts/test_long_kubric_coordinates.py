import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from audit_long_kubric_coordinates import inspect


class CoordinateAuditTests(unittest.TestCase):
    def test_recovers_world_to_camera_with_half_pixel_centres(self):
        H, W, T = 24, 32, 64
        v, u = np.mgrid[:H, :W]
        z = 2 + 0.8 * np.sin(u * 0.5) + 0.5 * np.cos(v * 0.7)
        K = np.array([45, 45, W / 2, H / 2], dtype=np.float32)
        camera = np.stack([(u + 0.5 - K[2]) * z / K[0], (v + 0.5 - K[3]) * z / K[1], z], axis=-1).reshape(-1, 3)
        R = cv2.Rodrigues(np.array([0.2, -0.1, 0.3]))[0]
        translation = np.array([0.1, -0.3, 2.0])
        world = (camera - translation) @ R
        xyz = np.repeat(world[:, None, :], T, axis=1).astype(np.float32)
        with tempfile.TemporaryDirectory(prefix="kubric-coordinate-unit-") as tmp:
            p = Path(tmp) / "test.npz"
            np.savez_compressed(p, world_coords=xyz, occluded=np.zeros((H * W, T), bool),
                                is_bkgd=np.zeros((H * W, 1), bool), W=W, fx_fy_cx_cy=K,
                                extrinsics_w2c_gt=np.repeat(np.eye(4)[None], T, axis=0))
            result = inspect(p)
        self.assertTrue(result["first_frame_coordinate_check_pass"])
        self.assertEqual(result["lowest_error_pixel_offset"], 0.5)
        best = next(x for x in result["convention_trials"] if x["pixel_offset"] == 0.5)
        self.assertLess(best["heldout_p90_px"], 1e-4)
        recovered = np.array(best["derived_absolute_world_to_camera0"])
        np.testing.assert_allclose(recovered[:3, :3], R, atol=1e-5)
        np.testing.assert_allclose(recovered[:3, 3], translation, atol=1e-5)
        self.assertFalse(result["model_input_change"])


if __name__ == "__main__":
    unittest.main()
