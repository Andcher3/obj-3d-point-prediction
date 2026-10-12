"""Controlled tests of coordinate diagnostics; no real-scene fitting/training."""
import unittest

import numpy as np

from audit_cross_point_model_coordinates import similarity, summarize


class CoordinateAuditChecks(unittest.TestCase):
    def test_known_similarity(self):
        a=np.random.default_rng(1).normal(size=(48,3))
        theta=.3
        r=np.array([[np.cos(theta),-np.sin(theta),0],
                    [np.sin(theta),np.cos(theta),0],[0,0,1]])
        b=7*(a@r)+[3,2,1]
        s,got,t=similarity(a,b)
        self.assertAlmostEqual(s,7.)
        np.testing.assert_allclose(s*(a@got)+t,b,atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(got),1.)

    def test_reflection_is_not_accepted_as_rotation(self):
        a=np.random.default_rng(2).normal(size=(48,3))
        b=a*[1,1,-1]
        s,r,t=similarity(a,b)
        self.assertAlmostEqual(np.linalg.det(r),1.)
        self.assertGreater(np.linalg.norm(s*(a@r)+t-b),.1)

    def test_fixed_map_applies_across_time(self):
        a=np.random.default_rng(3).normal(size=(7,48,3))
        b=4*a+[1,2,3]
        s,r,t=similarity(a[0],b[0])
        np.testing.assert_allclose(s*(a@r)+t,b,atol=1e-12)

    def test_normalized_error_summary(self):
        result=summarize(np.array([1.,2.,3.]),2.)
        self.assertAlmostEqual(result['p90_object_radius_fraction'],1.4)

    def test_label_calibration_roundtrip_and_observed_arrays_unchanged(self):
        from calibrate_cross_point_supervision import fit_observed_calibration
        rng=np.random.default_rng(7)
        uv=rng.uniform(10,245,(48,2))
        depth=np.full((392,392),2.,dtype=np.float32)
        intr=np.array([[450.,0,196.],[0,450.,196.],[0,0,1.]])
        poses=np.tile(np.eye(4),(7,1,1))
        xy=uv*392/256
        source=np.column_stack([(xy[:,0]-196)/450*2,(xy[:,1]-196)/450*2,np.full(48,2.)])
        true=3*source+[1.,2.,4.]
        depth_before=depth.copy();uv_before=uv.copy();poses_before=poses.copy()
        s,r,t,_=fit_observed_calibration(depth,intr,poses,uv,true)
        future=rng.normal(size=(15,48,3))
        native=(future-t)@r.T/s
        np.testing.assert_allclose(s*(native@r)+t,future,atol=1e-12)
        np.testing.assert_array_equal(depth,depth_before)
        np.testing.assert_array_equal(uv,uv_before)
        np.testing.assert_array_equal(poses,poses_before)

    def test_evaluation_rotation_translation_and_scale(self):
        from run_cross_point_probe import scene_metrics
        rng=np.random.default_rng(11)
        pred=rng.normal(size=(15,48,3));true=rng.normal(size=(15,48,3))
        r=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        native=scene_metrics(pred,true)
        target=scene_metrics(5*(pred@r)+3,5*(true@r)+3)
        for name in native:self.assertAlmostEqual(5*native[name],target[name])


if __name__=='__main__':unittest.main()
