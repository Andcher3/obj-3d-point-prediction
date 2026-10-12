"""CPU cache diagnosis: oracle point/camera substitutions, no model or raw edits.

All substitutions are GT-assisted diagnostics, never inference-time corrections.
World coordinates refer to input camera 0. Points and camera translations must
share the same scale. Pixel errors are measured in the original source image.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from render_uniquery4r_movif_focal_review import sha


def normalize_poses(e):
    e = np.asarray(e, np.float64)
    if e.shape[-2:] == (3, 4):
        full = np.tile(np.eye(4), (len(e), 1, 1))
        full[:, :3] = e
        e = full
    return (e @ np.linalg.inv(e[0]))[:, :3]


def camera_centers(e):
    return -np.einsum('tji,tj->ti', e[:, :3, :3], e[:, :3, 3])


def source_intrinsics(k, m):
    h, w = m['processed_resolution_hw']
    oh, ow = m['source_resolution_hw']
    a = np.array([[ow/w, 0, .5*ow/w-.5],
                  [0, oh/h, .5*oh/h-.5], [0, 0, 1.]])
    return a @ k


def projection(p, e, k):
    c = np.einsum('tij,tqj->tqi', e[:, :3, :3], p) + e[:, None, :3, 3]
    h = np.einsum('tij,tqj->tqi', k, c)
    with np.errstate(divide='ignore', invalid='ignore'):
        uv = h[..., :2] / h[..., 2:3]
    uv[c[..., 2] <= 0] = np.nan
    return uv


def stats(x):
    x = np.asarray(x).ravel()
    f = x[np.isfinite(x)]
    return {'selected_count': len(x), 'finite_count': len(f),
            'invalid_count': len(x)-len(f),
            'mean': float(f.mean()) if len(f) else None,
            'p90': float(np.quantile(f, .9)) if len(f) else None,
            'max': float(f.max()) if len(f) else None}


def image_errors(uv, truth_uv, visible, t0):
    err = np.linalg.norm(uv-truth_uv, axis=-1)
    return {'frame0_all': stats(err[0]), 'frame0_visible': stats(err[0][visible[0]]),
            't0_all': stats(err[t0]), 't0_visible': stats(err[t0][visible[t0]]),
            'future_all': stats(err[t0+1:]),
            'future_visible': stats(err[t0+1:][visible[t0+1:]])}


def substitutions(p, g, ep, eg, kp, kg, scale, uv, visible, t0):
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError('Invalid global scale')
    sp = p*scale
    se = ep.copy()
    se[:, :3, 3] *= scale
    result = {}
    for pt, pose, intr in itertools.product(('pred', 'gt'), repeat=3):
        projected = projection(sp if pt == 'pred' else g,
                               se if pose == 'pred' else eg,
                               kp if intr == 'pred' else kg)
        key = f'points_{pt}_pose_{pose}_intrinsics_{intr}'
        result[key] = image_errors(projected, uv, visible, t0)
    return result


def shape_error(p, g, scale, diagonal, t0):
    i, j = np.triu_indices(p.shape[1], 1)
    pl = np.linalg.norm(p[:, i]-p[:, j], axis=-1)
    gl = np.linalg.norm(g[:, i]-g[:, j], axis=-1)
    denom = np.maximum(gl, diagonal*1e-6)
    lengths = np.abs(pl*scale-gl)/denom
    optimal_scale_each_frame = np.sum(pl*gl, axis=1)/np.sum(pl*pl, axis=1)
    best_scale_lengths = np.abs(pl*optimal_scale_each_frame[:, None]-gl)/denom
    rigid_residual = []
    for a, b in zip(p*scale, g):
        a, b = a-a.mean(0), b-b.mean(0)
        u, _, vt = np.linalg.svd(a.T @ b)
        correction = np.eye(3)
        correction[2, 2] = np.linalg.det(u @ vt)
        rotation = u @ correction @ vt
        rigid_residual.append(np.linalg.norm(a @ rotation-b, axis=-1)/diagonal)
    return {'pair_length_error_fixed_t0_scale_future': stats(lengths[t0+1:]),
            'pair_length_error_oracle_framewise_scale_future': stats(best_scale_lengths[t0+1:]),
            'point_shape_residual_oracle_framewise_rigid_bbox_fraction_future': stats(np.array(rigid_residual)[t0+1:]),
            'oracle_per_frame_operations_are_diagnostic_only': True}


def diagnose(raw, gt, m):
    p = raw['tracks_3d'].astype(np.float64)
    g = gt['tracks_first_camera'].astype(np.float64)
    ep = raw['extrinsic'].astype(np.float64)
    eg = normalize_poses(gt['extrinsic_w2c_gt'])
    kp = source_intrinsics(raw['intrinsic'].astype(np.float64), m)
    kg = gt['intrinsic_gt'].astype(np.float64)
    uv = gt['tracks_2d'].astype(np.float64)
    vis = gt['visibility'].astype(bool)
    t0 = m['query_model_frame_index']
    if p.shape != g.shape or not np.isfinite(p).all():
        raise ValueError('Bad cached XYZ')
    np.testing.assert_allclose(ep[0], np.eye(4)[:3], atol=2e-5)
    np.testing.assert_allclose(eg[0], np.eye(4)[:3], atol=2e-12)
    np.testing.assert_array_equal(raw['source_frame_indices'], gt['source_frame_indices'])
    np.testing.assert_allclose(projection(g, eg, kg), uv, atol=2e-10)
    np.testing.assert_allclose(projection(p, ep, kp), raw['tracks_3d_projected_source_pixels'], atol=.003, equal_nan=True)
    i, j = np.triu_indices(p.shape[1], 1)
    pl = np.linalg.norm(p[t0, i]-p[t0, j], axis=-1)
    gl = np.linalg.norm(g[t0, i]-g[t0, j], axis=-1)
    if np.any(pl <= 1e-12):
        raise ValueError('Degenerate predicted pair')
    scale = float(np.median(gl/pl))
    primary = substitutions(p, g, ep, eg, kp, kg, scale, uv, vis, t0)
    cp, cg = camera_centers(ep), camera_centers(eg)
    denom = float(np.sum(cp[1:t0+1]**2))
    camera_scale = float(np.sum(cp[1:t0+1]*cg[1:t0+1])/denom) if denom > 1e-14 else None
    camera_scale_valid = bool(camera_scale is not None and camera_scale > 0 and np.isfinite(camera_scale))
    alternative = substitutions(p, g, ep, eg, kp, kg, camera_scale, uv, vis, t0) if camera_scale_valid else None
    relative = ep[:, :3, :3] @ np.swapaxes(eg[:, :3, :3], 1, 2)
    angles = np.degrees(np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2)-1)/2, -1, 1)))
    boxes = gt['target_bboxes_world'][t0]
    diagonal = float(np.max(np.linalg.norm(boxes[:, None]-boxes[None], axis=-1)))
    # At camera 0, translation is zero and scale cancels from projection. This
    # separately tests XYZ ray directions without predicted pose or focal.
    p0 = projection(p[:1], np.eye(4)[None, :3], kg[:1])[0]
    identity_error = np.linalg.norm(p0-uv[0], axis=-1)
    return {'scene_id': m['scene_id'], 'query_time': t0, 'frames': len(p),
            'point_t0_pair_scale': scale, 'camera_observed_motion_scale': camera_scale,
            'camera_observed_motion_scale_valid': camera_scale_valid,
            'primary_point_t0_scale': primary, 'alternative_camera_scale': alternative,
            'camera_rotation_error_degrees_observed': stats(angles[:t0+1]),
            'camera_rotation_error_degrees_future': stats(angles[t0+1:]),
            'camera_center_error_point_scale_future_scene_units': stats(np.linalg.norm(cp[t0+1:]*scale-cg[t0+1:], axis=-1)),
            'camera_center_error_camera_scale_future_scene_units': stats(np.linalg.norm(cp[t0+1:]*camera_scale-cg[t0+1:], axis=-1)) if camera_scale_valid else None,
            'point_shape': shape_error(p, g, scale, diagonal, t0),
            'frame0_identity_pose_true_intrinsics_all_px': stats(identity_error),
            'frame0_identity_pose_true_intrinsics_visible_px': stats(identity_error[vis[0]]),
            'gt_full_projection_roundtrip_max_px': float(np.max(np.linalg.norm(projection(g, eg, kg)-uv, axis=-1))),
            'reference_identity_error_pred': float(np.max(np.abs(ep[0]-np.eye(4)[:3]))),
            'raw_xyz_unchanged': True, 'future_gt_used_for_evaluation_only': True,
            'future_pose_or_point_fit_used_in_substitution': False,
            'direct_2d_future_visible': image_errors(raw['tracks_2d_source_pixels'], uv, vis, t0)['future_visible']}


def summarize(rows):
    def macro(records):
        means = [r['mean'] for r in records if r is not None and r['mean'] is not None]
        return {'scene_count': len(means), 'mean': float(np.mean(means)) if means else None,
                'invalid_selected_count': sum(r['invalid_count'] for r in records if r is not None)}
    cases = {}
    for name in rows[0]['primary_point_t0_scale']:
        cases[name] = {group: macro([r['primary_point_t0_scale'][name][group] for r in rows])
                       for group in ('frame0_visible', 't0_visible', 'future_visible')}
    alt = [r for r in rows if r['alternative_camera_scale'] is not None]
    return {'scene_count': len(rows), 'primary_projection_macro_px': cases,
            'alternative_projection_macro_future_px': {name: macro([r['alternative_camera_scale'][name]['future_visible'] for r in alt]) for name in cases},
            'camera_rotation_macro_future_degrees': macro([r['camera_rotation_error_degrees_future'] for r in rows]),
            'frame0_identity_true_camera_visible_macro_px': macro([r['frame0_identity_pose_true_intrinsics_visible_px'] for r in rows]),
            'pair_length_fixed_scale_macro_future': macro([r['point_shape']['pair_length_error_fixed_t0_scale_future'] for r in rows]),
            'pair_length_oracle_framewise_scale_macro_future': macro([r['point_shape']['pair_length_error_oracle_framewise_scale_future'] for r in rows]),
            'shape_rigid_residual_macro_future_bbox_fraction': macro([r['point_shape']['point_shape_residual_oracle_framewise_rigid_bbox_fraction_future'] for r in rows]),
            'scenes': rows, 'model_forward_called': False, 'raw_arrays_changed': False,
            'substitutions_are_gt_assisted_diagnosis_not_corrected_labels': True,
            'projection_errors_are_not_additive_error_contributions': True}


def main(base):
    status = json.loads((base/'logs/gt-preparation-status.json').read_text())
    rows = []
    for scene in status['scenes']:
        if scene['status'] != 'READY':
            continue
        name = scene['scene_id']
        paths = [base/f'outputs/{name}/{name}_22frames_raw.npz', base/f'inputs/{name}/gt_tracks.npz', base/f'inputs/{name}/queries.json']
        before = [sha(p) for p in paths]
        m = json.loads(paths[2].read_text())
        with np.load(paths[0], allow_pickle=False) as raw, np.load(paths[1], allow_pickle=False) as gt:
            row = diagnose(raw, gt, m)
        if before != [sha(p) for p in paths]:
            raise ValueError('Raw inputs changed')
        row['source_sha256'] = dict(zip(('raw', 'gt', 'query'), before))
        rows.append(row)
        print(json.dumps({'scene_id': name, 'diagnosed': True}), flush=True)
    report = summarize(rows)
    output = base/'logs/geometry-camera-attribution.json'
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'scenes'}, indent=2, allow_nan=False))
    digest = sha(output)
    print(json.dumps({'output_sha256_parts': [digest[:32], digest[32:]]}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', type=Path, required=True)
    args = ap.parse_args()
    main(args.base)
