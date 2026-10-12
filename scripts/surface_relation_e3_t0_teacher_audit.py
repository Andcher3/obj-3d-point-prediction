#!/usr/bin/env python3
"""Audit a TrackCraft3R teacher window whose first frame is formal t0."""
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

def sim(src,tgt):
    sm,tm=src.mean(0),tgt.mean(0); a,b=src-sm,tgt-tm; u,s,vt=np.linalg.svd(a.T@b); r=u@vt
    if np.linalg.det(r)<0: u[:,-1]*=-1; r=u@vt
    scale=float(s.sum()/max((a*a).sum(),1e-12)); return scale,r,tm-scale*(sm@r)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--teacher-dense',type=Path,required=True); p.add_argument('--teacher-camera',type=Path,required=True); p.add_argument('--observed-cache',type=Path,required=True); p.add_argument('--query-json',type=Path,required=True); p.add_argument('--output-json',type=Path,required=True); a=p.parse_args()
    d=np.load(a.teacher_dense); c=np.load(a.teacher_camera,allow_pickle=True); cache=np.load(a.observed_cache,allow_pickle=True); q=json.loads(a.query_json.read_text()); tr=np.asarray(d['track_map'],float); w=np.asarray(c['extrinsics_w2c'],float); intr=np.asarray(c['fx_fy_cx_cy'],float); uv=np.asarray(q['query_sets']['48'],float); sh,sw=map(int,q['mask_shape_hw']); H,W=tr.shape[1:3]
    k=np.array([[intr[0]*W/sw,0,intr[2]*W/sw],[0,intr[1]*H/sh,intr[3]*H/sh],[0,0,1]],float); qmodel=uv*np.array([W/sw,H/sh]); xyz=tf(w[0],tr[0]); valid=np.isfinite(xyz).all(-1)&(xyz[...,2]>0); yy,xx=np.nonzero(valid); proj=np.stack([k[0,0]*xyz[yy,xx,0]/xyz[yy,xx,2]+k[0,2],k[1,1]*xyz[yy,xx,1]/xyz[yy,xx,2]+k[1,2]],1); anchors=np.rint(qmodel).astype(int); anchors[:,0]=np.clip(anchors[:,0],0,W-1); anchors[:,1]=np.clip(anchors[:,1],0,H-1); tracks=tf(w,tr[:,anchors[:,1],anchors[:,0]]); teacher_t0=tracks[0]
    obs=np.asarray(cache['observed_only_trackcraft_t0'],float); finite=np.isfinite(teacher_t0).all(1)&np.isfinite(obs).all(1)&(teacher_t0[:,2]>0)&(obs[:,2]>0); residual=np.full(48,np.nan); folds=[]
    for fold in range(4):
        test=np.arange(48)[fold*12:(fold+1)*12]; train=np.setdiff1d(np.arange(48),test); trn=finite[train]; sc,r,t=sim(teacher_t0[train][trn],obs[train][trn]); pred=sc*(teacher_t0@r)+t; residual[test]=np.linalg.norm(pred[test]-obs[test],axis=1); fit=np.linalg.norm(pred[train[trn]]-obs[train[trn]],axis=1); folds.append({'fold':fold,'train_fit_p90_m':float(np.quantile(fit,.9)),'holdout_p90_m':float(np.quantile(residual[test],.9)),'scale':sc})
    bbox=np.linalg.norm(np.nanmax(obs,0)-np.nanmin(obs,0)); norm=residual/max(bbox,1e-8); out={'video_id':q['video_id'],'scope':'t0-first-frame TrackCraft3R teacher audit; no Relation Head training','teacher_frame_count':int(len(tr)),'formal_t0_dense_frame_index':0,'future_slice':'1:10','fixed_query_frame0_anchor':True,'valid_query_count':int(finite.sum()),'match_error_px':{'p50':float(np.median(np.linalg.norm(qmodel-anchors,axis=1))),'max':float(np.max(np.linalg.norm(qmodel-anchors,axis=1)))},'crossfit_registration_norm':{'p50':float(np.nanpercentile(norm,50)),'p90':float(np.nanpercentile(norm,90)),'max':float(np.nanmax(norm)),'count_over_10pct':int(np.sum(norm>.10))},'folds':folds,'gate':'PASS' if finite.sum()>=44 and np.nanpercentile(norm,90)<=.10 and np.nanmax(norm)<=.20 and np.sum(norm>.10)<=int(.1*48) else 'FAIL'}
    a.output_json.parent.mkdir(parents=True,exist_ok=True); a.output_json.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
