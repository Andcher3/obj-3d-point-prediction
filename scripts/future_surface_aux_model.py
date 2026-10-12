"""Shared-encoder trajectory and symmetric future-distance auxiliary heads."""
import torch
from torch import nn


class FutureSurfaceAuxModel(nn.Module):
    def __init__(self,input_width=781,hidden=128,pair_hidden=64,steps=15):
        super().__init__()
        self.steps=steps
        self.encoder=nn.Sequential(nn.Linear(input_width,hidden),nn.ReLU(),
                                   nn.Linear(hidden,hidden),nn.ReLU())
        self.position_head=nn.Linear(hidden,steps*3)
        self.distance_head=nn.Sequential(nn.Linear(hidden*2,pair_hidden),nn.ReLU(),
                                         nn.Linear(pair_hidden,steps),nn.Softplus())
        nn.init.zeros_(self.position_head.weight);nn.init.zeros_(self.position_head.bias)

    def forward(self,x,pairs):
        h=self.encoder(x)
        correction=self.position_head(h)
        a=h[:,pairs[:,0]];b=h[:,pairs[:,1]]
        symmetric=torch.cat([a+b,torch.abs(a-b)],dim=-1)
        distance=self.distance_head(symmetric)
        return correction,distance


def supervised_losses(correction,point_target,distance,distance_target,pair_mask,
                       point_moment,aux_moment):
    point=(correction-point_target).square().mean()/point_moment
    if not pair_mask.any():raise ValueError('No valid auxiliary surface pairs')
    # Equal scene weights; valid pair count varies with visible surface components.
    per_pair=(distance-distance_target).square().mean(-1)
    count=pair_mask.sum(-1).clamp_min(1)
    auxiliary=((per_pair*pair_mask).sum(-1)/count).mean()/aux_moment
    return point,auxiliary
