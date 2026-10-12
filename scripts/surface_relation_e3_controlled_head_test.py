#!/usr/bin/env python3
"""Controlled Relation Head engineering tests; no real-video supervision."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from surface_relation_feasibility import solve_component_gauge, incidence_matrix

class EdgeRelationHead(nn.Module):
    def __init__(self, hdim=16, hidden=32):
        super().__init__(); self.net=nn.Sequential(nn.Linear(2*hdim,hidden),nn.Tanh(),nn.Linear(hidden,3))
    def forward(self,h,edges):
        src=torch.tensor([e[0] for e in edges],device=h.device); dst=torch.tensor([e[1] for e in edges],device=h.device)
        return self.net(torch.cat([h[src]+h[dst],torch.abs(h[src]-h[dst])],-1))

class NodeResidual(nn.Module):
    def __init__(self,hdim=16,hidden=32):
        super().__init__(); self.net=nn.Sequential(nn.Linear(hdim,hidden),nn.Tanh(),nn.Linear(hidden,3))
    def forward(self,h): return self.net(h)

def permute_edges(edges,p):
    inv=torch.argsort(p).tolist(); return sorted((min(inv[i],inv[j]),max(inv[i],inv[j])) for i,j in edges)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output-json',type=Path,required=True); a=p.parse_args(); torch.manual_seed(7); np.random.seed(7); random.seed(7)
    n,t,hd=12,4,16; edges=sorted({(i,j) for i in range(n) for j in range(i+1,min(i+3,n))}); h=torch.randn(n,hd,dtype=torch.float64); p0=torch.randn(t,n,3,dtype=torch.float64); target_delta=torch.randn(t,n,3,dtype=torch.float64)*.08; pgt=p0+target_delta
    def integrate(edge_delta):
        corr,_=solve_component_gauge(edge_delta,edges,n); return p0+corr
    head=EdgeRelationHead(hd,32).double(); zero_pred=integrate(torch.zeros(t,len(edges),3,dtype=torch.float64)); zero_error=float(torch.max(torch.abs(zero_pred-p0)))
    pred=head(h,edges); pred_t=pred.unsqueeze(0).expand(t,-1,-1); loss=(integrate(pred_t)-pgt).pow(2).mean(); grad=torch.autograd.grad(loss,head.parameters(),allow_unused=False); grad_finite=all(torch.isfinite(g).all().item() for g in grad)
    opt=torch.optim.Adam(head.parameters(),lr=.05); losses=[]
    for _ in range(1000):
        opt.zero_grad(); out=integrate(head(h,edges).unsqueeze(0).expand(t,-1,-1)); l=(out-pgt).pow(2).mean(); l.backward(); opt.step(); losses.append(float(l.detach()))
    fitted=float(losses[-1]); node=NodeResidual(hd,58).double(); node_count=sum(x.numel() for x in node.parameters()); edge_count=sum(x.numel() for x in head.parameters()); perm=torch.randperm(n); hp=h[perm]; ep=permute_edges(edges,perm); perm_pred=head(hp,ep); orig=head(h,edges); perm_back=torch.empty_like(perm_pred); inv=torch.argsort(perm); 
    # Edge ordering is sorted after remapping; compare as an unordered edge map.
    map_orig={e:i for i,e in enumerate(edges)}; map_perm={e:i for i,e in enumerate(ep)}; vals=[]; inv=torch.argsort(perm)
    for e,i in map_orig.items():
        mapped=tuple(sorted((int(inv[e[0]]),int(inv[e[1]]))))
        vals.append(torch.max(torch.abs(orig[i]-perm_pred[map_perm[mapped]])))
    out={'scope':'synthetic/oracle Relation Head engineering test; no real-video claim','node_count':n,'edge_count':len(edges),'zero_correction_max_abs':zero_error,'gradient_finite':bool(grad_finite),'initial_loss':float(loss.detach()),'overfit_final_loss':fitted,'overfit_loss_ratio':fitted/max(float(loss.detach()),1e-12),'edge_head_parameters':edge_count,'node_residual_parameters':node_count,'parameter_match_ratio':edge_count/max(node_count,1),'query_permutation_max_abs':float(torch.max(torch.stack(vals))), 'constrained_l2_test':'PASS' if zero_error==0.0 and grad_finite else 'FAIL','overfit_test':'PASS' if fitted<float(loss.detach())*.05 else 'FAIL','permutation_test':'PASS' if float(torch.max(torch.stack(vals)))<1e-10 else 'FAIL'}
    a.output_json.parent.mkdir(parents=True,exist_ok=True); a.output_json.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
