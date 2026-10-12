import unittest

import torch

from future_surface_aux_model import FutureSurfaceAuxModel, supervised_losses


class FutureSurfaceAuxChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4);torch.set_num_threads(1)
        self.model=FutureSurfaceAuxModel(input_width=5,hidden=8,pair_hidden=6,steps=2)
        self.x=torch.randn(2,4,5)
        self.pairs=torch.tensor([[0,1],[0,2],[1,3]])

    def test_zero_position_initialization_and_positive_distances(self):
        p,d=self.model(self.x,self.pairs)
        self.assertEqual(p.abs().max().item(),0)
        self.assertTrue((d>0).all())

    def test_auxiliary_gradient_reaches_shared_encoder(self):
        _,d=self.model(self.x,self.pairs)
        loss=(d-3).square().mean();loss.backward()
        norm=sum(p.grad.abs().sum().item() for p in self.model.encoder.parameters())
        self.assertGreater(norm,0)
        self.assertIsNone(self.model.position_head.weight.grad)

    def test_pair_order_symmetric(self):
        _,a=self.model(self.x,self.pairs);_,b=self.model(self.x,self.pairs.flip(-1))
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_joint_query_permutation(self):
        perm=torch.tensor([2,0,3,1]);inverse=torch.argsort(perm)
        p,a=self.model(self.x,self.pairs)
        q,b=self.model(self.x[:,perm],inverse[self.pairs])
        torch.testing.assert_close(p,q[:,inverse],rtol=1e-6,atol=1e-7)
        torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7)

    def test_masked_pairs_have_no_loss_effect(self):
        p,d=self.model(self.x,self.pairs);mask=torch.tensor([[True,False,True]]).expand(2,-1)
        target=torch.ones_like(d)
        _,a=supervised_losses(p,torch.ones_like(p),d,target,mask,1.,1.)
        target[:,1]=1000
        _,b=supervised_losses(p,torch.ones_like(p),d,target,mask,1.,1.)
        torch.testing.assert_close(a,b)

    def test_same_capacity_for_all_supervision_conditions(self):
        a=sum(p.numel() for p in self.model.parameters())
        b=sum(p.numel() for p in FutureSurfaceAuxModel(5,8,6,2).parameters())
        self.assertEqual(a,b)


if __name__=='__main__':unittest.main()
