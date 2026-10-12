#!/usr/bin/env python3
"""Quality gate for single-context relation targets; no model training."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

def tf(m,p):
    m=np.asarray(m)
    if m.ndim==2: return p @ m[:3,:3].T + m[:3,3]
    return np.einsum('tij,tnj->tni',m[:,:3,:3],p)+m[:,:3,3][:,None,:]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dense-npz',type=Path,required=True); p.add_argument('--camera-npz',type=Path,required=True); p.add_argument('--query-json',type=Path,required=True); p.add_argument('--output-json',type=Path,required=True); a=p.parse_args()
    d=np.load(a.dense_npz); c=np.load(a.camera_npz,allow_pickle=True); q=json.loads(a.query_json.read_text()); tr=np.asarray(d['track_map'],float); w=np.asarray(c['extrinsics_w2c'],float); intr=np.asarray(c['fx_fy_cx_cy'],float); uv=np.asarray(q['query_sets']['48'],float); sh,sw=map(int,q['mask_shape_hw']); H,W=tr.shape[1:3]
    timestamps=np.asarray(c['timestamps_s'],float); t0_idx=int(np.argmin(np.abs(timestamps-float(q['t0_seconds'])))); time_error=float(abs(timestamps[t0_idx]-float(q['t0_seconds']))); protocol_16=(len(tr)==16 and t0_idx==6 and len(tr)-t0_idx-1==9); future_slice=slice(t0_idx+1,t0_idx+10)
    k=np.array([[intr[0]*W/sw,0,intr[2]*W/sw],[0,intr[1]*H/sh,intr[3]*H/sh],[0,0,1]],float); xyz=tf(w[t0_idx],tr[t0_idx]); valid=np.isfinite(xyz).all(-1)&(xyz[...,2]>0); yy,xx=np.nonzero(valid); proj=np.stack([k[0,0]*xyz[yy,xx,0]/xyz[yy,xx,2]+k[0,2],k[1,1]*xyz[yy,xx,1]/xyz[yy,xx,2]+k[1,2]],1); qmodel=uv*np.array([W/sw,H/sh]); tree=cKDTree(proj); near,nn=tree.query(qmodel,k=min(128,len(proj))); cand=np.unique(np.asarray(nn).reshape(-1)); cost=np.linalg.norm(qmodel[:,None]-proj[cand][None],axis=-1); rows,cols=linear_sum_assignment(cost); match=cost[rows,cols][np.argsort(rows)]; sel=cand[cols[np.argsort(rows)]]; tracks=tf(w,tr[:,yy[sel],xx[sel]]); p0=tracks[t0_idx]; future=tracks[future_slice]; finite=np.isfinite(tracks).all((0,2))&(tracks[:,:,2]>0).all(0)
    bbox=q.get('bbox_xyxy'); diag2=float(np.linalg.norm([bbox[2]-bbox[0],bbox[3]-bbox[1]])); norm=match/diag2; keep=finite&(norm<=.10); graph_pts=qmodel; _,neigh=cKDTree(graph_pts).query(graph_pts,k=4); edges=set(); adj=[set() for _ in range(48)]
    for i,row in enumerate(neigh[:,1:]):
        for j in row: edges.add(tuple(sorted((i,int(j)))))
    for i,j in edges: adj[i].add(j); adj[j].add(i)
    seen=set(); stack=[0]; seen.add(0)
    while stack:
        u=stack.pop()
        for v in adj[u]:
            if v not in seen: seen.add(v); stack.append(v)
    drift=[]
    for frame in future:
        vals=[]
        for i in range(48):
            for j in range(i+1,48):
                d0=np.linalg.norm(p0[i]-p0[j]); vals.append(abs(np.linalg.norm(frame[i]-frame[j])-d0)/max(d0,1e-8))
        drift.append(vals)
    out={'video_id':q['video_id'],'scope':'single-context relation-target quality audit; no Relation Head training','frame_count':int(len(tr)),'t0_dense_frame_index':t0_idx,'t0_timestamp_error_s':time_error,'formal_16_frame_protocol':protocol_16,'valid_query_count':int(finite.sum()),'anchor_valid_query_count':int(keep.sum()),'anchor_error_norm_2d':{'p50':float(np.median(norm)),'p90':float(np.quantile(norm,.9)),'max':float(np.max(norm)),'count_over_5pct':int(np.sum(norm>.05)),'count_over_10pct':int(np.sum(norm>.10))},'bbox_xyxy':bbox,'bbox_diag_2d':diag2,'graph':{'edge_count':len(edges),'connected':len(seen)==48,'component_count':1 if len(seen)==48 else 2},'future_pairwise_distance_drift':{'p50':float(np.median(drift)),'p90':float(np.quantile(drift,.9)),'max':float(np.max(drift))},'gate':'PASS' if protocol_16 and finite.sum()>=44 and keep.sum()>=44 and np.quantile(norm,.9)<=.05 and np.max(norm)<=.10 and len(seen)==48 else 'FAIL'}
    a.output_json.parent.mkdir(parents=True,exist_ok=True); a.output_json.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
