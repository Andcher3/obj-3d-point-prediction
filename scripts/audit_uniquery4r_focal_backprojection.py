"""CPU diagnosis of fitted-K projection -> original-K lifting, keep camera Z.

Corrected XYZ is an in-memory diagnostic candidate, not a new teacher label.
No future fit, no per-point translation, no edits to original model arrays.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_uniquery4r_movif_geometry_camera import projection, source_intrinsics, stats
from render_uniquery4r_movif_focal_review import diagnostic, sha


def camera_coordinates(points, e):
    return np.einsum('tij,tqj->tqi', e[:, :3, :3], points) + e[:, None, :3, 3]


def inverse_camera(points, e):
    return np.einsum('tij,tqj->tqi', np.linalg.inv(e[:, :3, :3]), points-e[:, None, :3, 3])


def lift_keep_camera_depth(points, e, original_k, projection_k):
    camera = camera_coordinates(points, e)
    if not np.isfinite(camera).all() or np.any(camera[..., 2] <= 0):
        raise ValueError('Non-finite/behind-camera depth; refuse silent masking')
    fitted_uv = projection(points, e, projection_k)
    uv1 = np.concatenate((fitted_uv, np.ones((*fitted_uv.shape[:2], 1))), axis=-1)
    ray = np.einsum('tij,tqj->tqi', np.linalg.inv(original_k), uv1)
    lifted_camera = ray/ray[..., 2:3]*camera[..., 2:3]
    corrected = inverse_camera(lifted_camera, e)
    if not np.isfinite(corrected).all():
        raise ValueError('Non-finite lifted XYZ')
    np.testing.assert_allclose(projection(corrected, e, original_k), fitted_uv, atol=2e-8)
    np.testing.assert_allclose(camera_coordinates(corrected,e)[...,2], camera[...,2], atol=2e-10)
    return corrected


def scale_from_t0_pairs(p, g, t0):
    i, j = np.triu_indices(p.shape[1], 1)
    pl = np.linalg.norm(p[t0,i]-p[t0,j], axis=-1)
    gl = np.linalg.norm(g[t0,i]-g[t0,j], axis=-1)
    if np.any(pl <= 1e-12) or np.any(gl <= 1e-12):
        raise ValueError('Degenerate t0 pair')
    return float(np.median(gl/pl))


def metrics(p, g, scale, diagonal, t0, vis):
    p = p*scale
    point = np.linalg.norm(p-g, axis=-1)
    centered = np.linalg.norm((p-p.mean(1,keepdims=True))-(g-g.mean(1,keepdims=True)), axis=-1)/diagonal
    i, j = np.triu_indices(p.shape[1], 1)
    pe, ge = p[:,i]-p[:,j], g[:,i]-g[:,j]
    pl, gl = np.linalg.norm(pe,axis=-1), np.linalg.norm(ge,axis=-1)
    edge = np.linalg.norm(pe-ge,axis=-1)/diagonal
    length = np.abs(pl-gl)/np.maximum(gl, diagonal*1e-6)
    change = np.linalg.norm((pe-pe[t0])-(ge-ge[t0]),axis=-1)/diagonal
    groups = {}
    for label, sl in [('t0', slice(t0,t0+1)), ('future',slice(t0+1,None))]:
        groups[label] = {
            'point_error_scene_units': stats(point[sl]),
            'point_error_bbox_fraction': stats(point[sl]/diagonal),
            'centered_point_error_bbox_fraction': stats(centered[sl]),
            'relative_vector_error_bbox_fraction': stats(edge[sl]),
            'relative_pair_length_error_fraction': stats(length[sl]),
            'relative_vector_change_error_bbox_fraction': stats(change[sl])}
    groups['future_visible_point_error_scene_units'] = stats(point[t0+1:][vis[t0+1:]])
    groups['future_visible_point_error_bbox_fraction'] = stats(point[t0+1:][vis[t0+1:]]/diagonal)
    return groups


def scene_diagnosis(raw, gt, m):
    p = raw['tracks_3d'].astype(np.float64)
    g = gt['tracks_first_camera'].astype(np.float64)
    e, k = raw['extrinsic'].astype(np.float64), raw['intrinsic'].astype(np.float64)
    t0, vis = m['query_model_frame_index'], gt['visibility'].astype(bool)
    old_uv, fit_uv, fit_info = diagnostic(p, e, k, m, gt['tracks_2d'],vis,gt['intrinsic_gt'])
    ratio = np.array(fit_info['fixed_focal_ratio_xy'])
    fitted_k = k.copy()
    fitted_k[:,0,0] *= ratio[0]
    fitted_k[:,1,1] *= ratio[1]
    changed = lift_keep_camera_depth(p, e, k, fitted_k)
    # Independent algebra check: the operation rescales camera X/Y, not Z.
    camera_expected = camera_coordinates(p,e)*[ratio[0],ratio[1],1.]
    np.testing.assert_allclose(changed, inverse_camera(camera_expected,e), atol=2e-10)
    same_k = lift_keep_camera_depth(p,e,fitted_k,fitted_k)
    np.testing.assert_allclose(same_k, p, atol=2e-10)
    np.testing.assert_allclose(projection(changed,e,source_intrinsics(k,m)),fit_uv,atol=2e-8)
    np.testing.assert_allclose(old_uv,raw['tracks_3d_projected_source_pixels'],atol=.003,equal_nan=True)
    scale = scale_from_t0_pairs(p,g,t0)
    changed_scale = scale_from_t0_pairs(changed,g,t0)
    box = gt['target_bboxes_world'][t0]
    diagonal = float(np.max(np.linalg.norm(box[:,None]-box[None],axis=-1)))
    result = {'scene_id':m['scene_id'], 'focal_ratio_xy':ratio.tolist(),
              'original_t0_pair_scale':scale, 'lifted_independent_t0_pair_scale':changed_scale,
              'target_bbox_diagonal_scene_units':diagonal,
              'original_common_scale':metrics(p,g,scale,diagonal,t0,vis),
              'lifted_common_scale':metrics(changed,g,scale,diagonal,t0,vis),
              'lifted_independent_t0_scale_diagnostic':metrics(changed,g,changed_scale,diagonal,t0,vis),
              'original_unscaled_diagnostic':metrics(p,g,1.,diagonal,t0,vis),
              'lifted_unscaled_diagnostic':metrics(changed,g,1.,diagonal,t0,vis),
              'original_projection_future_visible_px':fit_info['original']['future_visible_error_px'],
              'fit_projection_future_visible_px':fit_info['refit']['future_visible_error_px'],
              'depth_preserved_max_error':float(np.max(np.abs(camera_coordinates(changed,e)[...,2]-camera_coordinates(p,e)[...,2]))),
              'matching_fitted_k_inverse_xyz_max_error':float(np.max(np.abs(same_k-p))),
              'lifted_xyz_not_saved_as_pseudo_gt':True, 'future_fit':False}
    return result


def summarize(rows):
    labels = ['original_common_scale','lifted_common_scale','lifted_independent_t0_scale_diagnostic',
              'original_unscaled_diagnostic','lifted_unscaled_diagnostic']
    macro = {}
    for label in labels:
        macro[label] = {metric:float(np.mean([r[label]['future'][metric]['mean'] for r in rows]))
                        for metric in rows[0][label]['future']}
    counts = {}
    for metric in macro['original_common_scale']:
        pairs = [(r['original_common_scale']['future'][metric]['mean'], r['lifted_common_scale']['future'][metric]['mean']) for r in rows]
        counts[metric] = {'improved':sum(b<a for a,b in pairs), 'worsened':sum(b>a for a,b in pairs)}
    return {'scene_count':len(rows), 'future_scene_macro':macro,
            'common_scale_improvement_scene_counts':counts,
            'future_visible_projection_macro_px': {label:float(np.mean([r[key]['mean'] for r in rows])) for label,key in [('original','original_projection_future_visible_px'),('fit_then_lift','fit_projection_future_visible_px')]},
            'protocol':'fit_K_project_then_original_K_inverse_with_original_camera_Z',
            'primary_scale':'both variants use original t0 pair-distance scale',
            'separate_scale_comparison_is_gt_t0_assisted_sensitivity_only':True,
            'camera_pose_unchanged':True, 'camera_depth_unchanged':True,
            'model_forward_called':False, 'raw_arrays_changed':False,
            'scenes':rows}


def main(base):
    target = base/'logs/focal-backprojection-3d-comparison.json'
    if target.exists():
        raise FileExistsError(target)
    status = json.loads((base/'logs/gt-preparation-status.json').read_text())
    rows = []
    for scene in status['scenes']:
        if scene['status'] != 'READY':
            continue
        name = scene['scene_id']
        paths = [base/f'outputs/{name}/{name}_22frames_raw.npz', base/f'inputs/{name}/gt_tracks.npz', base/f'inputs/{name}/queries.json']
        hashes = [sha(path) for path in paths]
        m = json.loads(paths[2].read_text())
        with np.load(paths[0],allow_pickle=False) as raw, np.load(paths[1],allow_pickle=False) as gt:
            row = scene_diagnosis(raw,gt,m)
        if hashes != [sha(path) for path in paths]:
            raise ValueError('Raw input changed')
        row['source_sha256'] = dict(zip(('raw','gt','query'),hashes))
        rows.append(row)
        print(json.dumps({'scene':name,'completed':True}),flush=True)
    report = summarize(rows)
    target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'scenes'},indent=2))
    digest = sha(target)
    print(digest[:32],flush=True)
    print(digest[32:],flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--base',type=Path,required=True)
    args = ap.parse_args()
    main(args.base)
