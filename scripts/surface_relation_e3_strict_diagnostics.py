#!/usr/bin/env python3
"""Supplementary diagnostics for completed strict E3 audits."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

def transform(m,p): return p @ m[:3,:3].T + m[:3,3]
def main():
    p=argparse.ArgumentParser(); p.add_argument('--past-npz',type=Path,required=True); p.add_argument('--past-dense',type=Path,required=True); p.add_argument('--sup-npz',type=Path,required=True); p.add_argument('--sup-dense',type=Path,required=True); p.add_argument('--query-json',type=Path,required=True); p.add_argument('--audit-json',type=Path,required=True); p.add_argument('--output-json',type=Path,required=True); a=p.parse_args()
    q=json.loads(a.query_json.read_text()); old=json.loads(a.audit_json.read_text()); past=np.load(a.past_npz); sup=np.load(a.sup_npz); dp=np.load(a.past_dense); ds=np.load(a.sup_dense); uv=np.asarray(q['query_sets']['48'],float); hw=np.asarray(q['mask_shape_hw']);
    def direct(d,c,t):
        tr=np.asarray(d['track_map'],float); w=np.asarray(c['extrinsics_w2c'],float); H,W=tr.shape[1:3]; uv2=uv*np.array([W/hw[1],H/hw[0]]); x=np.rint(uv2[:,0]).astype(int); y=np.rint(uv2[:,1]).astype(int); return transform(w[t],tr[t,y,x])
    p0=direct(dp,past,11); s0=direct(ds,sup,2); finite=np.isfinite(p0).all(1)&np.isfinite(s0).all(1); sm,tm=s0[finite].mean(0),p0[finite].mean(0); aa=s0[finite]-sm; bb=p0[finite]-tm; u,sv,vt=np.linalg.svd(aa.T@bb); r=u@vt; sc=float(sv.sum()/max((aa*aa).sum(),1e-12));
    if np.linalg.det(r)<0: u[:,-1]*=-1; r=u@vt
    pred=sc*(s0@r)+(tm-sc*(sm@r)); pair=[]
    for i in range(48):
        for j in range(i+1,48): pair.append(abs(np.linalg.norm(pred[i]-pred[j])-np.linalg.norm(p0[i]-p0[j]))/max(np.linalg.norm(p0[i]-p0[j]),1e-8))
    bbox=q.get('bbox_xyxy'); diag2=float(np.linalg.norm([bbox[2]-bbox[0],bbox[3]-bbox[1]])) if bbox else float(np.linalg.norm(np.ptp(uv,0))); tree=cKDTree(uv); dist,_=tree.query(uv,k=3)
    out={'video_id':q['video_id'],'direct_uv_self_consistency':{'finite_count':int(finite.sum()),'similarity_fit_p90_m':float(np.quantile(np.linalg.norm(pred[finite]-p0[finite],axis=1),.9))},'pairwise_distance_relative_error':{'p50':float(np.median(pair)),'p90':float(np.quantile(pair,.9)),'max':float(np.max(pair))},'formal_bbox_xyxy':bbox,'formal_bbox_diag_2d':diag2,'query_pairwise_spacing_px':{'p50':float(np.median(dist[:,1])),'p90':float(np.quantile(dist[:,1],.9)),'max':float(np.max(dist[:,1]))},'query_pairwise_second_spacing_gap_px':{'p50':float(np.median(dist[:,2]-dist[:,1])),'p90':float(np.quantile(dist[:,2]-dist[:,1],.9)),'min':float(np.min(dist[:,2]-dist[:,1]))},'note':'These spacing fields describe distances among query UV points, not distances to dense-track candidates.','prior_audit_status':old['status']}
    a.output_json.parent.mkdir(parents=True,exist_ok=True); a.output_json.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
