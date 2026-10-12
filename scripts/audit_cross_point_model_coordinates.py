"""Read-only raw coordinate diagnostics; never approves or rewrites model caches.

All correspondence fits use known query identities in observed frames only.
Fit parameters/GT arrays are never written as prediction-time model features.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def similarity(source, target):
    a, b = source.mean(0), target.mean(0)
    x, y = source-a, target-b
    u, singular, vt = np.linalg.svd(x.T @ y)
    signs = np.ones(3)
    signs[-1] = np.linalg.det(u @ vt)
    rotation = u @ np.diag(signs) @ vt
    scale = float((singular*signs).sum() / max(float((x*x).sum()), 1e-12))
    translation = b - scale*(a @ rotation)
    return scale, rotation, translation


def summarize(errors, size):
    return {'median': float(np.median(errors)), 'p90': float(np.percentile(errors,90)),
            'maximum': float(np.max(errors)),
            'p90_object_radius_fraction': float(np.percentile(errors,90)/size)}


def audit_scene(base, scene):
    folder = base/'inputs'/scene
    with np.load(base/'model_cache_raw'/(scene+'.npz'), allow_pickle=False) as cache:
        observed = np.asarray(cache['observed_xyz'], dtype=np.float64)
        if cache['gt_input_supplied'] or cache['future_rgb_supplied']:
            raise ValueError('Prediction inputs contain GT/future data')
        units = str(cache['coordinate_units'])
    with np.load(base/'observed_geometry'/(scene+'.npz'), allow_pickle=False) as geometry:
        depth = geometry['depth'][0].copy()
        intr = geometry['intrinsics'][0].copy()
        poses = geometry['extrinsics_w2c'].copy()
    with np.load(folder/'gt_tracks.npz', allow_pickle=False) as gt:
        true = np.asarray(gt['tracks_last_observed_camera'][:7], dtype=np.float64)
        gt_intr = gt['intrinsic_gt'][:7].copy()
        gt_poses = gt['extrinsic_w2c_gt'][:7].copy()
    uv = np.asarray(json.loads((folder/'queries.json').read_text())['query_uv_source_pixels'])
    h,w=256,256
    xy = (uv * [depth.shape[1]/w, depth.shape[0]/h]).astype(np.float32)
    z = cv2.remap(depth, xy[:,0].reshape(1,-1), xy[:,1].reshape(1,-1),
                  cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()
    camera0 = np.stack([(xy[:,0]-intr[0,2])/intr[0,0]*z,
                        (xy[:,1]-intr[1,2])/intr[1,1]*z,z],-1)
    relative = poses[-1] @ np.linalg.inv(poses[0])
    pj0 = camera0 @ relative[:3,:3].T + relative[:3,3]
    radius = float(np.sqrt(np.mean(np.sum((true[0]-true[0].mean(0))**2,-1))))
    radius = max(radius,1e-8)
    rows = {}
    for name, source, target in [('DA3_query_frame0',pj0,true[0]),
                                 ('MF_observed_frame0',observed[0],true[0]),
                                 ('MF_observed_last',observed[-1],true[-1])]:
        rows[name] = {'raw_error':summarize(np.linalg.norm(source-target,axis=-1),radius),
                      'raw_object_radius':float(np.sqrt(np.mean(np.sum((source-source.mean(0))**2,-1))))}
        s,r,t=similarity(source,target)
        rows[name]['all_query_similarity_diagnostic']={'scale':s,
            'rotation_angle_deg':float(np.degrees(np.arccos(np.clip((np.trace(r)-1)/2,-1,1)))),
            'translation':t.tolist(), 'error':summarize(np.linalg.norm(s*(source@r)+t-target,axis=-1),radius)}
        # Four held-out query groups expose non-global errors hidden by a fit.
        heldout=np.zeros(48)
        for fold in range(4):
            test=np.arange(48)%4==fold
            scale,rotation,translation=similarity(source[~test],target[~test])
            heldout[test]=np.linalg.norm(scale*(source[test]@rotation)+translation-target[test],axis=-1)
        rows[name]['heldout_similarity_error']=summarize(heldout,radius)
    # One frame-0 map is held fixed across the remaining observed times.
    s,r,t=similarity(observed[0],true[0])
    aligned=s*(observed@r)+t
    rows['fixed_frame0_mapping_past'] = summarize(np.linalg.norm(aligned-true,axis=-1),radius)
    gt_relative=gt_poses[-1]@np.linalg.inv(gt_poses[0])
    scale_obs=rows['DA3_query_frame0']['all_query_similarity_diagnostic']['scale']
    inferred_k=intr.copy(); inferred_k[0]*=w/depth.shape[1]; inferred_k[1]*=h/depth.shape[0]
    return {'scene':scene,'raw_cache_units':units,'estimated_camera_vs_GT_not_preapproved':True,
        'GT_observed_radius':radius,'diagnostics':rows,
        'first_frame_fx_ratio':float(inferred_k[0,0]/gt_intr[0,0,0]),
        'first_frame_fy_ratio':float(inferred_k[1,1]/gt_intr[0,1,1]),
        'camera_relative_rotation_difference_deg':float(np.degrees(np.arccos(np.clip(
            (np.trace(relative[:3,:3]@gt_relative[:3,:3].T)-1)/2,-1,1)))),
        'camera_relative_translation_difference_after_query_scale':float(np.linalg.norm(
            scale_obs*relative[:3,3]-gt_relative[:3,3])),
        'training_coordinate_audit_pass':False, 'future_GT_used_to_fit':False,
        'GT_provided_to_forecaster':False,'cache_rewritten':False}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve existing audit')
    split=json.loads((args.base/'split-manifest.json').read_text())['splits']
    scenes=[s for s in sum(split.values(),[]) if (args.base/'model_cache_raw'/(s+'.npz')).exists()]
    if not scenes:
        raise ValueError('No raw model output available')
    results={'scenes':[audit_scene(args.base,s) for s in scenes],
             'audit_only':True,'performance_evaluation':False,'training_authorized_by_audit':False}
    args.output.write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
