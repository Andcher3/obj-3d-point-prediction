import unittest

import numpy as np

from audit_g2_surface_readiness import rigidity_diagnostic


class SurfaceReadinessChecks(unittest.TestCase):
    def test_rigid_motion_keeps_pair_lengths(self):
        points=np.array([[0.,0.,0.],[1.,0.,0.],[0.,2.,0.]])
        rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        tracks=np.stack([points,points@rotation+[2.,3.,4.]])
        self.assertLess(rigidity_diagnostic(tracks)['pair_length_drift_relative_max'],1e-12)

    def test_shape_change_is_not_rigid(self):
        points=np.array([[0.,0.,0.],[1.,0.,0.],[0.,2.,0.]])
        tracks=np.stack([points,points*2])
        self.assertAlmostEqual(rigidity_diagnostic(tracks)['pair_length_drift_relative_max'],1.)

    def test_invalid_points_rejected(self):
        with self.assertRaises(ValueError):
            rigidity_diagnostic(np.full((2,3,3),np.nan))


if __name__=='__main__':unittest.main()
