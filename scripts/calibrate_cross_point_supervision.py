"""User-approved past-only coordinate calibration of labels/evaluation.

Raw predictor inputs/outputs are never calibrated with GT. Fit a single similarity
from DA3 frame-0 query geometry to simulator frame-0 points in reference camera 6.
Apply its inverse to future LABELS only. No future labels choose fit parameters.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from audit_cross_point_model_coordinates import similarity, summarize


def sha(path):
    with path.open('rb') as stream:
        h=hashlib.sha256()
        for chunk in iter(lambda:stream.read(1024**2),b''):
            h.update(chunk)
    return h.hexdigest()


def fit_observed_calibration(depth, intr, poses, query_uv, gt_frame0):
    xy=(np.asarray(query_uv)*[depth.shape[1]/256,depth.shape[0]/256]).astype(np.float32)
    z=cv2.remap(depth,xy[:,0].reshape(1,-1),xy[:,1].reshape(1,-1),
                cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).ravel()
    pts=np.stack([(xy[:,0]-intr[0,2])/intr[0,0]*z,
                  (xy[:,1]-intr[1,2])/intr[1,1]*z,z],-1)
    relative=poses[6]@np.linalg.inv(poses[0])
    source=pts@relative[:3,:3].T+relative[:3,3]
    target=np.asarray(gt_frame0,dtype=np.float64)
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError('Nonfinite observed calibration inputs')
    scale,rotation,translation=similarity(source,target)
    if not np.isfinite(scale) or scale<=1e-8:
        raise ValueError('Invalid calibration scale')
    source_radius=float(np.sqrt(np.mean(np.sum((source-source.mean(0))**2,-1))))
    if source_radius<1e-8:
        raise ValueError('Degenerate observed point set')
    return scale,rotation,translation,source


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base',type=Path,required=True)
    ap.add_argument('--allow-observed-gt-calibration',action='store_true')
    args=ap.parse_args()
    if not args.allow_observed_gt_calibration:
        raise PermissionError('Observed-GT label/evaluation calibration requires user approval')
    base=args.base.resolve()
    split=json.loads((base/'split-manifest.json').read_text())['splits']
    scenes=sum(split.values(),[])
    out_cache=base/'model_cache_audited'
    out_labels=base/'calibrated_labels'
    if out_cache.exists() or out_labels.exists():
        raise FileExistsError('Preserve previous calibrated data')
    out_cache.mkdir();out_labels.mkdir()
    results=[]
    for scene in scenes:
        folder=base/'inputs'/scene
        query=json.loads((folder/'queries.json').read_text())
        if query['query_source_frame']!=0 or query['last_observed_source_frame']!=6:
            raise ValueError('Calibration frame protocol mismatch')
        raw=base/'model_cache_raw'/(scene+'.npz')
        raw_sha=sha(raw)
        with np.load(raw,allow_pickle=False) as cache:
            arrays={k:cache[k].copy() for k in cache.files}
        if bool(arrays['gt_input_supplied']) or bool(arrays['future_rgb_supplied']):
            raise ValueError('Forbidden GT/future predictor input')
        with np.load(base/'observed_geometry'/(scene+'.npz'),allow_pickle=False) as geometry:
            depth=geometry['depth'][0].copy()
            intr=geometry['intrinsics'][0].copy()
            poses=geometry['extrinsics_w2c'].copy()
        with np.load(folder/'gt_tracks.npz',allow_pickle=False) as gt:
            # Only known observed frame-0 point identities fit the transform.
            observed_true=gt['tracks_last_observed_camera'][0].copy()
        scale,rotation,translation,source=fit_observed_calibration(
            depth,intr,poses,np.asarray(query['query_uv_source_pixels']),observed_true)
        radius=max(float(np.sqrt(np.mean(np.sum((observed_true-observed_true.mean(0))**2,-1)))),1e-8)
        fit_error=np.linalg.norm(scale*(source@rotation)+translation-observed_true,axis=-1)
        heldout=np.zeros(48)
        for fold in range(4):
            test=np.arange(48)%4==fold
            s,r,t=similarity(source[~test],observed_true[~test])
            heldout[test]=np.linalg.norm(s*(source[test]@r)+t-observed_true[test],axis=-1)
        # Future GT is opened only after calibration is completely determined.
        with np.load(base/'cpu_cache'/(scene+'-labels.npz'),allow_pickle=False) as original:
            label={k:original[k].copy() for k in original.files}
        native=(label['future_xyz']-translation)@rotation.T/scale
        np.testing.assert_allclose(scale*(native@rotation)+translation,label['future_xyz'],atol=1e-10)
        if not np.isfinite(native).all():raise ValueError('Invalid transformed labels')
        np.savez_compressed(out_labels/(scene+'-labels.npz'),
            future_xyz=native,validity=label['validity'],visibility=label['visibility'],
            timestamps_s=label['timestamps_s'],evaluation_to_gt_scale=scale,
            evaluation_to_gt_rotation=rotation,evaluation_to_gt_translation=translation,
            coordinate_frame='last_observed_camera',coordinate_units='DA3_estimated_units',
            calibration_fit_frame=0,reference_camera_frame=6,future_used_to_fit=False,
            raw_cache_sha256=raw_sha,query_sha256=arrays['query_sha256'])
        arrays['training_coordinate_audit_pass']=np.asarray(True)
        arrays['label_coordinate_calibration']='observed_DA3_frame0_to_simulator_frame0'
        arrays['GT_assisted_label_evaluation_calibration']=np.asarray(True)
        arrays['raw_cache_sha256']=np.asarray(raw_sha)
        # The only changed fields are audit metadata. Features remain identical.
        np.savez_compressed(out_cache/(scene+'.npz'),**arrays)
        with np.load(out_cache/(scene+'.npz'),allow_pickle=False) as copy:
            for key in ['observed_xyz','baseline_future','latent_future','query_uv','timestamps_s']:
                np.testing.assert_array_equal(copy[key],arrays[key])
        if sha(raw)!=raw_sha:raise ValueError('Raw predictor cache changed')
        results.append({'scene':scene,'model_to_GT_scale':scale,
            'rotation_row_matrix':rotation.tolist(),'translation':translation.tolist(),
            'fit_error':summarize(fit_error,radius),'heldout_error':summarize(heldout,radius),
            'raw_cache_sha256':raw_sha,'prediction_inputs_unchanged':True,
            'future_GT_used_to_fit':False,'calibration_frame':0,'reference_camera_frame':6,
            'large_geometric_residuals_are_observation_error_not_hidden':True})
    cfg=json.loads((base/'logs/effective-protocol.json').read_text())
    cfg.update(labels_directory='calibrated_labels',
        supervision_coordinate_protocol='user-approved observed-frame0 similarity; fixed for future labels and all groups',
        evaluation='GT-assisted observed coordinate calibration; Euclidean errors scaled to simulator units',
        gpu_forward_authorized=True)
    (base/'logs/calibrated-protocol.json').write_text(json.dumps(cfg,indent=2))
    report={'calibration_complete':True,'scenes':results,'counts':{k:len(v) for k,v in split.items()},
        'features_unchanged':True,'raw_caches_unchanged':True,'future_GT_used_to_fit':False,
        'GT_assisted_label_evaluation_calibration':True,
        'geometric_accuracy_pass_claimed':False,'main_predictor_inputs':'observed RGB and query only'}
    (base/'logs/observed-calibration-complete.json').write_text(json.dumps(report,indent=2))
    print('OBSERVED_CALIBRATION_COMPLETE',len(results),'scenes; GT-assisted label/evaluation coordinates',flush=True)


if __name__=='__main__':main()
