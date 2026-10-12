#!/usr/bin/env python3
"""Audit identity and registration between two TrackCraft3R runs at common t0."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

def transform(m,p):
    m = np.asarray(m)
    if m.ndim == 2:
        return p @ m[:3,:3].T + m[:3,3]
    return np.einsum('tij,tnj->tni', m[:,:3,:3], p) + m[:,:3,3][:,None,:]

def assign(dense, cam, query_uv, query_hw, t0):
    tr=np.asarray(dense['track_map'],np.float64); w2c=np.asarray(cam['extrinsics_w2c'],np.float64); intr=np.asarray(cam['fx_fy_cx_cy'],np.float64)
    H,W=tr.shape[1:3]; sh,sw=query_hw; k=np.array([[intr[0]*W/sw,0,intr[2]*W/sw],[0,intr[1]*H/sh,intr[3]*H/sh],[0,0,1]],float)
    xyz=transform(w2c[t0],tr[t0]); valid=np.isfinite(xyz).all(-1)&(xyz[...,2]>1e-8); yy,xx=np.nonzero(valid)
    proj=np.empty((len(xx),2)); proj[:,0]=k[0,0]*xyz[yy,xx,0]/xyz[yy,xx,2]+k[0,2]; proj[:,1]=k[1,1]*xyz[yy,xx,1]/xyz[yy,xx,2]+k[1,2]
    q=query_uv*np.array([W/sw,H/sh]); tree=cKDTree(proj); _, neigh=tree.query(q,k=min(256,len(proj))); cand=np.unique(np.asarray(neigh).reshape(-1)); cost=((q[:,None,:]-proj[cand][None,:,:])**2).sum(-1)**0.5
    rows,cols=linear_sum_assignment(cost); sel=np.full(len(q),-1,int); sel[rows]=cand[cols]
    anchors=np.stack([xx[sel],yy[sel]],1); err=cost[rows,cols][np.argsort(rows)]
    tracks=tr[:,anchors[:,1],anchors[:,0]]; tracks=transform(w2c,tracks)
    return tracks,anchors,err,proj[cand]

def sim(src,tgt):
    sm,tm=src.mean(0),tgt.mean(0); a,b=src-sm,tgt-tm; u,s,vt=np.linalg.svd(a.T@b); r=u@vt
    if np.linalg.det(r)<0: u[:,-1]*=-1; r=u@vt
    scale=float(s.sum()/max((a*a).sum(),1e-12)); trans=tm-scale*(sm@r); return scale,r,trans

def main():
    p=argparse.ArgumentParser(); p.add_argument('--past-npz',type=Path,required=True); p.add_argument('--past-dense',type=Path,required=True); p.add_argument('--sup-npz',type=Path,required=True); p.add_argument('--sup-dense',type=Path,required=True); p.add_argument('--query-json',type=Path,required=True); p.add_argument('--output-json',type=Path,required=True); p.add_argument('--output-csv',type=Path,required=True); args=p.parse_args()
    q=json.loads(args.query_json.read_text(encoding='utf-8')); uv=np.asarray(q['query_sets']['48'],float); hw=tuple(map(int,q['mask_shape_hw']))
    past=np.load(args.past_npz,allow_pickle=True); sup=np.load(args.sup_npz,allow_pickle=True); dp=np.load(args.past_dense); ds=np.load(args.sup_dense)
    if len(past['frame_indices'])!=12 or len(sup['frame_indices'])!=12: raise ValueError('strict windows must each contain 12 frames')
    tp,ap,ep,_=assign(dp,past,uv,hw,11); ts,ass,es,_=assign(ds,sup,uv,hw,2)
    p0=tp[11]; s0=ts[2]; finite=np.isfinite(p0).all(1)&np.isfinite(s0).all(1)&(p0[:,2]>0)&(s0[:,2]>0)
    folds=[]; residual=np.full(48,np.nan); aligned=np.full_like(s0,np.nan)
    for fold in range(4):
        test=np.arange(48)[fold*12:(fold+1)*12]; train=np.setdiff1d(np.arange(48),test); valid=finite[train]
        if valid.sum()<3: continue
        sc,r,tr=sim(s0[train][valid],p0[train][valid]); pred=sc*(s0@r)+tr; aligned=pred
        residual[test]=np.linalg.norm(pred[test]-p0[test],axis=1); folds.append({'fold':fold,'train_count':int(valid.sum()),'test_indices':test.tolist(),'scale':sc})
    bbox=np.linalg.norm(np.nanmax(p0,0)-np.nanmin(p0,0)); rnorm=residual/max(bbox,1e-8)
    src2=np.asarray(q['query_sets']['48'],float); bbox2=np.ptp(src2,axis=0); diag2=float(np.linalg.norm(bbox2)); anchor=np.maximum(ep,es); anorm=anchor/max(diag2,1e-8)
    finite_all=np.isfinite(residual); pass_mask=finite_all & (rnorm<=.10); H,W=dp['track_map'].shape[1:3]; pts=src2*np.array([W/hw[1],H/hw[0]]); tree=cKDTree(pts); _,n=tree.query(pts,k=4); edges=set();
    for i,row in enumerate(n[:,1:]):
        for j in row: edges.add(tuple(sorted((i,int(j)))))
    adj=[set() for _ in range(48)]
    for i,j in edges: adj[i].add(j);adj[j].add(i)
    seen=set(); comps=[]
    for i in range(48):
        if i in seen: continue
        stack=[i]; seen.add(i); c=[]
        while stack:
            u=stack.pop(); c.append(u)
            for v in adj[u]:
                if v not in seen: seen.add(v); stack.append(v)
        comps.append(c)
    payload={'video_id':q['video_id'],'status':'PASS' if (finite_all.sum()>=44 and np.nanpercentile(rnorm,90)<=.10 and np.nanmax(rnorm)<=.20 and np.nanpercentile(anorm,90)<=.05 and np.nanmax(anorm)<=.10 and len(comps)==1) else 'FAIL','valid_query_count':int(finite_all.sum()),'registration_error_norm':{'p50':float(np.nanpercentile(rnorm,50)),'p90':float(np.nanpercentile(rnorm,90)),'max':float(np.nanmax(rnorm)),'count_over_10pct':int(np.sum(rnorm>.10))},'anchor_error_norm_2d':{'p50':float(np.nanpercentile(anorm,50)),'p90':float(np.nanpercentile(anorm,90)),'max':float(np.nanmax(anorm))},'bbox_diag_3d':bbox,'bbox_diag_2d':diag2,'components':[len(x) for x in comps],'edge_count':len(edges),'bridge_edges':[],'folds':folds,'past_anchor_error_px':{'p50':float(np.median(ep)),'max':float(np.max(ep))},'supervision_anchor_error_px':{'p50':float(np.median(es)),'max':float(np.max(es))}}
    args.output_json.parent.mkdir(parents=True,exist_ok=True); args.output_json.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    args.output_csv.parent.mkdir(parents=True,exist_ok=True)
    with args.output_csv.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['query','finite','registration_error_norm','anchor_error_norm_2d']); [w.writerow([i,bool(finite_all[i]),float(rnorm[i]) if finite_all[i] else '',float(anorm[i])]) for i in range(48)]
    print(json.dumps(payload,indent=2))
if __name__=='__main__': main()
