#!/usr/bin/env python3
"""Matched-parameter oracle comparison: edge Relation Head vs node residual.

Engineering-only test. No real-video labels are used.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from surface_relation_feasibility import solve_component_gauge


def make_edges(uv, k=3):
    d = np.linalg.norm(uv[:, None] - uv[None, :], axis=-1)
    n = len(uv); e=set()
    for i in range(n):
        for j in np.argsort(d[i] + np.eye(n)[i]*1e9)[:k]:
            e.add(tuple(sorted((i,int(j)))))
    return sorted(e)


class EdgeHead(nn.Module):
    def __init__(self, h=16, hidden=32):
        super().__init__(); self.net=nn.Sequential(nn.Linear(2*h,hidden),nn.Tanh(),nn.Linear(hidden,3))
    def forward(self, x, edges):
        a=torch.tensor([i for i,j in edges],device=x.device); b=torch.tensor([j for i,j in edges],device=x.device)
        return self.net(torch.cat([x[a]+x[b],(x[a]-x[b]).abs()],-1))


class NodeHead(nn.Module):
    def __init__(self, h=16, hidden=58):
        super().__init__(); self.net=nn.Sequential(nn.Linear(h,hidden),nn.Tanh(),nn.Linear(hidden,3))
    def forward(self, x): return self.net(x)


def run(seed, steps=600):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    n,t,h=48,9,16
    uv=np.random.default_rng(seed).random((n,2)); edges=make_edges(uv)
    H=torch.randn(n,h,dtype=torch.float64)
    p0=torch.randn(t,n,3,dtype=torch.float64)*.2
    target_delta=torch.randn(t,len(edges),3,dtype=torch.float64)*.04
    target=solve_component_gauge(target_delta,edges,n)[0] + p0
    def fit(model, edge):
        opt=torch.optim.Adam(model.parameters(),lr=.04); init=None
        for step in range(steps):
            opt.zero_grad()
            raw=model(H,edges) if edge else model(H)
            raw=raw.unsqueeze(0).expand(t,-1,-1) if edge else raw.unsqueeze(0).expand(t,-1,-1)
            pred=solve_component_gauge(raw,edges,n)[0] + p0 if edge else p0+raw
            loss=(pred-target).pow(2).mean()
            if init is None: init=float(loss.detach())
            loss.backward(); opt.step()
        return init,float(loss.detach())
    e0,e1=fit(EdgeHead().double(),True); n0,n1=fit(NodeHead().double(),False)
    return {"seed":seed,"edge_initial":e0,"edge_final":e1,"edge_ratio":e1/max(e0,1e-12),"node_initial":n0,"node_final":n1,"node_ratio":n1/max(n0,1e-12),"edge_count":len(edges)}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--seeds',type=int,default=10); a=ap.parse_args()
    rows=[run(1000+i) for i in range(a.seeds)]
    out={"scope":"synthetic/oracle matched-parameter Relation Head comparison; no real-video training","edge_parameters":1155,"node_parameters":1163,"runs":rows,"aggregate":{}}
    for key in ('edge_ratio','node_ratio','edge_final','node_final'):
        vals=np.array([r[key] for r in rows]); out['aggregate'][key]={"median":float(np.median(vals)),"p90":float(np.quantile(vals,.9)),"max":float(vals.max())}
    out['interpretation']='Engineering fit comparison only; does not establish real-video learnability or superiority.'
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
