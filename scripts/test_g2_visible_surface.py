import unittest

import numpy as np

from prepare_g2_visible_surface import surface_relations
from run_g2_visible_conditions import condition_features, neighbor_indices


CFG={'mesh_edge_depth_ratio_limit':.1,'mesh_edge_length_median_multiplier':10.}


class VisibleSurfaceChecks(unittest.TestCase):
    def grid(self):
        x,y=np.meshgrid(np.arange(6),np.arange(5))
        return np.stack([x,y,np.ones_like(x)],-1).astype(float)

    def test_plane_geodesic_matches_diagonal(self):
        xyz=self.grid();mask=np.ones((5,6),bool)
        relation,audit=surface_relations(xyz,mask,[[0,0],[3,3]],np.ones(mask.shape),CFG)
        self.assertAlmostEqual(relation['surface_path'][0,1],relation['straight_distance'][0,1])
        self.assertEqual(audit['query_connected_components'],1)

    def test_hole_requires_longer_path(self):
        xyz=self.grid();mask=np.ones((5,6),bool);mask[1:4,2:4]=False
        relation,_=surface_relations(xyz,mask,[[0,2],[5,2]],np.ones(mask.shape),CFG)
        self.assertGreater(relation['surface_path'][0,1],relation['straight_distance'][0,1])

    def test_disconnected_surface_is_not_bridged(self):
        xyz=self.grid();mask=np.ones((5,6),bool);mask[:,2:4]=False
        relation,audit=surface_relations(xyz,mask,[[0,0],[5,0]],np.ones(mask.shape),CFG)
        self.assertTrue(np.isinf(relation['surface_path'][0,1]))
        self.assertEqual(audit['query_connected_components'],2)

    def test_unreachable_neighbors_fall_back_without_duplicates(self):
        path=np.full((4,4),np.inf);np.fill_diagonal(path,0);path[0,1]=path[1,0]=1
        uv=np.c_[np.arange(4),np.zeros(4)]
        index,fallback=neighbor_indices(path,uv,2,True)
        self.assertGreater(fallback,0)
        for i,row in enumerate(index):
            self.assertNotIn(i,row);self.assertEqual(len(set(row)),2)

    def test_equal_dimensions_and_shared_features(self):
        rng=np.random.default_rng(2);n=12
        cache={'observed_xyz':rng.normal(size=(7,n,3)),
               'baseline_future':rng.normal(size=(15,n,3)),
               'latent_future':rng.normal(size=(15,n,16)),
               'query_uv':rng.normal(size=(n,2))}
        length=np.linalg.norm(cache['observed_xyz'][0,:,None]-cache['observed_xyz'][0,None,:],axis=-1)
        surface={p+'_'+m:(length*1.5 if m=='surface_path' else length)
                 for p in ['estimated','oracle'] for m in ['surface_path','straight_distance']}
        arrays=[]
        for condition in ['image_neighbors_straight','image_neighbors_surface',
                          'surface_neighbors_straight','surface_neighbors_surface','oracle_visible_surface']:
            x,_=condition_features(cache,surface,condition,8);arrays.append(x)
            self.assertEqual(x.shape,(n,797));self.assertTrue(np.isfinite(x).all())
        for x in arrays[1:]:np.testing.assert_array_equal(x[:,:612],arrays[0][:,:612])


if __name__=='__main__':unittest.main()
