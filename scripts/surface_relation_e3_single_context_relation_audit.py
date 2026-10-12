#!/usr/bin/env python3
"""Audit relation targets from one TrackCraft3R context; no model training."""
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
    p=argparse.ArgumentParser()
    p.add_argument('--dense-npz',type=Path,required=True); p.add_argument('--camera-npz',type=Path,required=True); p.add_argument('--query-json',type=Path,required=True); p.add_argument('--output-json',type=Path,required=True)
    a=p.parse_args(); d=np.load(a.dense_npz); c=np.load(a.camera_npz,allow_pickle=True); q=json.loads(a.query_json.read_text())
    tr=np.asarray(d['track_map'],float); w=np.asarray(c['extrinsics_w2c'],float); intr=np.asarray(c['fx_fy_cx_cy'],float); uv=np.asarray(q['query_sets']['48'],float); sh,sw=map(int,q['mask_shape_hw']); H,W=tr.shape[1:3]
    k=np.array([[intr[0]*W/sw,0,intr[2]*W/sw],[0,intr[1]*H/sh,intr[3]*H/sh],[0,0,1]],float); xyz=tf(w[6],tr[6]); valid=np.isfinite(xyz).all(-1)&(xyz[...,2]>0); yy,xx=np.nonzero(valid); proj=np.stack([k[0,0]*xyz[yy,xx,0]/xyz[yy,xx,2]+k[0,2],k[1,1]*xyz[yy,xx,1]/xyz[yy,xx,2]+k[1,2]],1); qmodel=uv*np.array([W/sw,H/sh]); tree=cKDTree(proj); _,nn=tree.query(qmodel,k=min(128,len(proj))); cand=np.unique(np.asarray(nn).reshape(-1)); cost=np.linalg.norm(qmodel[:,None]-proj[cand][None],axis=-1); rows,cols=linear_sum_assignment(cost); sel=cand[cols[np.argsort(rows)]]; anchors=np.stack([xx[sel],yy[sel]],1); tracks=tf(w,tr[:,anchors[:,1],anchors[:,0]])
    p0=tracks[6]; future=tracks[7:16]; valid_q=np.isfinite(tracks).all((0,2))&(tracks[:,:,2]>0).all(0); edges=[]; tree3=cKDTree(p0); _,nei=tree3.query(p0,k=4)
    for i,row in enumerate(nei[:,1:]):
        for j in row:
            e=tuple(sorted((i,int(j))));
            if e not in edges: edges.append(e)
    edge0=np.array([p0[j]-p0[i] for i,j in edges]); rel=[]
    for frame in future:
        edge=np.array([frame[j]-frame[i] for i,j in edges]); rel.append(np.linalg.norm(edge-edge0,axis=1))
    rel=np.asarray(rel); bbox=np.linalg.norm(np.nanmax(p0,0)-np.nanmin(p0,0)); shape_change=[]
    for frame in future:
        vals=[]
        for i,j in edges: vals.append(abs(np.linalg.norm(frame[j]-frame[i])-np.linalg.norm(p0[j]-p0[i]))/max(np.linalg.norm(p0[j]-p0[i]),1e-8))
        shape_change.append(vals)
    out={'video_id':q['video_id'],'scope':'single-context offline relation-target audit; no Relation Head training','frame_indices':np.asarray(c['frame_indices']).tolist(),'t0_dense_frame_index':6,'query_count':48,'valid_query_count':int(valid_q.sum()),'global_match_error_px':{'p50':float(np.median(cost[rows,cols])),'max':float(np.max(cost[rows,cols]))},'edge_count':len(edges),'bbox_diag_3d':float(bbox),'edge_displacement_change_m':{'p50':float(np.median(rel)),'p90':float(np.quantile(rel,.9)),'max':float(np.max(rel))},'normalized_shape_change':{'p50':float(np.median(shape_change)),'p90':float(np.quantile(shape_change,.9)),'max':float(np.max(shape_change))},'future_horizon_count':9,'coordinate_frame':'single_trackcraft_context_last_observed_camera'}
    a.output_json.parent.mkdir(parents=True,exist_ok=True); a.output_json.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
