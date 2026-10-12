"""CPU sampling tests; no model, download or GPU initialization."""
import unittest

import numpy as np

from prepare_uniquery4r_movif_density_query6 import grid_pixels, use_original48


class DensityChecks(unittest.TestCase):
    def test_full_image_density_and_spacing(self):
        xy, step, total = grid_pixels(np.ones((256,256), bool))
        self.assertAlmostEqual(step, np.sqrt(65536/1000))
        self.assertEqual(total, 1024)
        self.assertEqual(len(xy), total)
        self.assertEqual(len(np.unique(xy, axis=0)), len(xy))
        differences = np.diff(np.unique(xy[:,0]))
        self.assertTrue(np.isin(differences, [8,9]).all())
        self.assertTrue(((xy >= 0) & (xy < 256)).all())

    def test_mask_and_fixed_origin(self):
        full = np.ones((256,256), bool)
        mask = np.zeros_like(full)
        mask[60:150,75:180] = True
        xy, step, _ = grid_pixels(mask)
        all_xy, full_step, _ = grid_pixels(full)
        np.testing.assert_array_equal(xy, all_xy[mask[all_xy[:,1],all_xy[:,0]]])
        self.assertEqual(step, full_step)
        self.assertTrue(mask[xy[:,1],xy[:,0]].all())
        np.testing.assert_array_equal(xy, grid_pixels(mask)[0])

    def test_fallback_boundary(self):
        self.assertTrue(use_original48(0))
        self.assertTrue(use_original48(47))
        self.assertFalse(use_original48(48))
        self.assertFalse(use_original48(80))

    def test_processed_density(self):
        _, step, _ = grid_pixels(np.ones((256,256),bool))
        self.assertAlmostEqual(step*252/256, np.sqrt(252*252/1000))

    def test_invalid_budget(self):
        with self.assertRaises(ValueError):
            grid_pixels(np.ones((10,10),bool), 0)
        with self.assertRaises(ValueError):
            grid_pixels(np.ones((10,10),bool), 10000)


if __name__ == "__main__":
    unittest.main()
