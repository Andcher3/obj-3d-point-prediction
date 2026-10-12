"""CPU diagnostic only: source integrity, independent projection, focal cross-fit.

Does not import/load a model, change raw outputs, or create replacement labels.
Focal fits use known source pixel locations in self-pairs as a diagnostic, not GT
camera calibration. No metric 3D ground truth is available for these clips.
"""
import argparse
import colorsys
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def stats(values):
    a = np.asarray(values).reshape(-1)
    finite = a[np.isfinite(a)]
    return {'finite_count': int(len(finite)), 'total_count': int(len(a)),
            'median': float(np.median(finite)) if len(finite) else None,
            'p90': float(np.percentile(finite, 90)) if len(finite) else None,
            'max': float(np.max(finite)) if len(finite) else None}


def pixel_error(a, b):
    return stats(np.linalg.norm(a - b, axis=-1))


def to_original(uv, m):
    h, w = m['processed_resolution_hw']
    oh, ow = m['source_resolution_hw']
    return (uv + 0.5) * [ow / w, oh / h] - 0.5


def camera_rays(points, E):
    C = np.einsum('tij,tqj->tqi', E[:, :3, :3], points) + E[:, None, :3, 3]
    with np.errstate(divide='ignore', invalid='ignore'):
        ray = C[..., :2] / C[..., 2:3]
    ray[C[..., 2] <= 0] = np.nan
    return ray


def project(points, E, K, m):
    return to_original(camera_rays(points, E) * np.stack((K[:, 0, 0], K[:, 1, 1]), -1)[:, None]
                       + K[:, None, :2, 2], m)


def fit_focal(rays, expected, center):
    # Two unknowns only. No point offsets, camera rotation/translation, principal
    # point, query identity, or future annotations are optimized.
    return np.sum(rays * (expected - center), axis=0) / np.sum(rays * rays, axis=0)


def focal_crossfit(points, E, K, expected, m):
    rays = camera_rays(points[None], E[None])[0]
    out = np.full((48, 2), np.nan)
    fitted = []
    for fold in range(4):
        test = np.arange(48) % 4 == fold
        train = ~test
        focal = fit_focal(rays[train], expected[train], K[:2, 2])
        fitted.append(focal.tolist())
        out[test] = rays[test] * focal + K[:2, 2]
    all_focal = fit_focal(rays, expected, K[:2, 2])
    return {'heldout_pixel_error': pixel_error(to_original(out, m), to_original(expected, m)),
            'fold_focals_processed_px': fitted, 'all_points_focal_processed_px': all_focal.tolist(),
            'fitted_to_model_focal_ratio': (all_focal / [K[0, 0], K[1, 1]]).tolist()}, all_focal


def mask_hit(uv, mask):
    h, w = mask.shape
    finite = np.isfinite(uv).all(-1)
    p = np.zeros_like(uv, dtype=int)
    p[finite] = np.rint(uv[finite]).astype(int)
    on_screen = finite & (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
    hits = np.zeros(len(uv), dtype=bool)
    hits[on_screen] = mask[p[on_screen, 1], p[on_screen, 0]]
    return {'hits': int(hits.sum()), 'on_screen': int(on_screen.sum()),
            'finite': int(finite.sum()), 'queries': len(uv)}


def draw(image, points, title):
    panel = cv2.copyMakeBorder(image, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
    cv2.putText(panel, title, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .47, (255, 255, 255), 1)
    h, w = image.shape[:2]
    for i, p in enumerate(points):
        if np.isfinite(p).all() and 0 <= p[0] < w and 0 <= p[1] < h:
            color = tuple(round(x*255) for x in colorsys.hsv_to_rgb(i/48, .8, 1)[::-1])
            xy = (int(round(p[0])), int(round(p[1])) + 40)
            cv2.circle(panel, xy, 4, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(panel, xy, 3, color, -1, cv2.LINE_AA)
    return panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', type=Path, required=True)
    args = ap.parse_args()
    base = args.base.resolve()
    output = base / 'drift-diagnosis-cpu'
    output.mkdir(exist_ok=True)
    hashes = json.loads((base / 'input_sha256.json').read_text())
    summary = {'mode': 'CPU_CACHE_DIAGNOSTIC', 'model_forward_called': False,
               'raw_outputs_modified': False, 'camera_fit_is_diagnostic_only': True,
               'no_3d_or_camera_ground_truth': True, 'videos': []}
    for video in ('car-turn', 'drift-turn', 'boat', 'soccerball'):
        directory = base / 'inputs' / video
        m = json.loads((directory / 'queries.json').read_text())
        h, w = m['processed_resolution_hw']
        q = np.asarray(m['query_uv_processed_pixels'])
        q_original = np.asarray(m['query_uv_source_pixels'])
        t0 = m['query_model_frame_index']
        audit = {'video': video, 'source_frame': m['query_source_frame_index'], 'contexts': []}
        for rel, expected_hash in hashes.items():
            if not rel.startswith('inputs/' + video + '/'):
                continue
            assert hashlib.sha256((base / rel).read_bytes()).hexdigest() == expected_hash, rel
        mask0 = np.asarray(Image.open(directory / 'mask' / f"{m['query_source_frame_index']:05d}.png")) == 1
        audit['approved_query_mask_hits'] = mask_hit(q_original, mask0)
        audit['query_pixel_roundtrip_max'] = float(np.max(abs(to_original(q, m)-q_original)))
        for count in (22, 7):
            path = base / 'outputs' / video / f'{video}_{count}frames_raw.npz'
            digest_before = hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path) as loaded:
                raw = {k: loaded[k].astype(float) for k in loaded.files}
            pairs = [tuple(map(int, p)) for p in raw['pair_idx']]
            cross_ids = [pairs.index((t0, t)) for t in range(count)]
            self_ids = [pairs.index((t, t)) for t in range(count)]
            assert len(pairs) == len(set(pairs)) == 2 * count - 1
            assert np.array_equal(raw['source_frame_indices'], m['source_frame_indices'][:count])
            assert np.array_equal(raw['tracks_3d'], raw['warp3d'][cross_ids])
            np.testing.assert_allclose(raw['query_uv_normalized_01'], q/[w-1, h-1], atol=3e-8, rtol=0)
            # Independent pose decoder, using official XYZW quaternion and FoV order.
            pose = raw['pose_enc']
            E_raw = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
            E_raw[:, :3, :3] = Rotation.from_quat(pose[:, 3:7]).as_matrix()
            E_raw[:, :3, 3] = pose[:, :3]
            E_rebuilt = E_raw @ np.linalg.inv(E_raw[0])
            E, K = raw['extrinsic'], raw['intrinsic']
            focal_rebuilt = np.stack((w/2 / np.tan(pose[:, 8]/2), h/2 / np.tan(pose[:, 7]/2)), -1)
            assert np.max(abs(E_rebuilt[:, :3]-E)) < 1e-6
            assert np.max(abs(focal_rebuilt-np.stack((K[:, 0, 0], K[:, 1, 1]), -1))) < .002
            np.testing.assert_allclose(E[0], np.eye(4)[:3], atol=2e-6)
            uv2 = to_original((raw['warp2d'][cross_ids]+1)*.5*[w-1, h-1], m)
            uv3 = project(raw['warp3d'][cross_ids], E, K, m)
            uv_self2 = to_original((raw['warp2d'][self_ids]+1)*.5*[w-1, h-1], m)
            uv_self3 = project(raw['warp3d'][self_ids], E, K, m)
            np.testing.assert_allclose(uv2, raw['tracks_2d_source_pixels'], atol=1e-6)
            np.testing.assert_allclose(uv3, raw['tracks_3d_projected_source_pixels'], atol=.003, equal_nan=True)
            cf, focal = focal_crossfit(raw['tracks_3d'][t0], E[t0], K[t0], q, m)
            K_test = K.copy()
            ratio = focal / [K[t0, 0, 0], K[t0, 1, 1]]
            K_test[:, 0, 0] *= ratio[0]
            K_test[:, 1, 1] *= ratio[1]
            uv3_test = project(raw['tracks_3d'], E, K_test, m)
            focal_frames = []
            frames = []
            for t, frame_index in enumerate(m['source_frame_indices'][:count]):
                mask = np.asarray(Image.open(directory / 'mask' / f'{frame_index:05d}.png')) == 1
                frames.append({'model_frame': t, 'source_frame': frame_index,
                               'raw2d_mask': mask_hit(uv2[t], mask), 'raw3d_mask': mask_hit(uv3[t], mask),
                               'diagnostic_t0_focal_ratio_projection_mask': mask_hit(uv3_test[t], mask),
                               'self2d_error_px': pixel_error(uv_self2[t], q_original),
                               'self3d_error_px': pixel_error(uv_self3[t], q_original)})
                f, _ = focal_crossfit(raw['warp3d'][self_ids[t]], E[t], K[t], q, m)
                focal_frames.append(f)
            context = {'frames': count, 'input_integrity_passed': True,
                       'pair_selection_and_array_index_passed': True,
                       'independent_camera_decode_max_abs': float(np.max(abs(E_rebuilt[:, :3]-E))),
                       'independent_project_max_abs_px': float(np.nanmax(abs(uv3-raw['tracks_3d_projected_source_pixels']))),
                       't0_raw2d_vs_known_query_px': pixel_error(uv2[t0], q_original),
                       't0_raw3d_projection_vs_known_query_px': pixel_error(uv3[t0], q_original),
                       't0_camera_focal_px_processed': [float(K[t0, 0, 0]), float(K[t0, 1, 1])],
                       't0_camera_focal_px_source': [float(K[t0, 0, 0]*m['source_resolution_hw'][1]/w),
                                                   float(K[t0, 1, 1]*m['source_resolution_hw'][0]/h)],
                       't0_focal_only_crossfit': cf, 'per_frame_selfpair_focal_crossfit': focal_frames,
                       'per_frame_diagnostics': frames,
                       'mask_interpretation': 'mask hit cannot verify physical point identity or 3D depth'}
            audit['contexts'].append(context)
            if video == 'drift-turn':
                image = cv2.imread(str(directory / 'rgb' / f"{m['query_source_frame_index']:05d}.jpg"))
                panels = [draw(image, q_original, 'Approved query at t0'),
                          draw(image, uv2[t0], 'Original model 2D output at t0'),
                          draw(image, uv3[t0], 'Original 3D projected with model camera'),
                          draw(image, uv3_test[t0], 'DIAGNOSTIC fitted fx/fy only - NOT corrected ground truth')]
                cv2.imwrite(str(output / f'drift-turn_{count}frames_t0_diagnostic.png'),
                            np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:]))))
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest_before
        summary['videos'].append(audit)
    (output / 'cached-output-diagnostic.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'completed': True, 'gpu_used': False, 'output': str(output)}, indent=2))
    for video in summary['videos']:
        for context in video['contexts']:
            print(video['video'], context['frames'], 't0 2D', context['t0_raw2d_vs_known_query_px']['median'],
                  '3D', context['t0_raw3d_projection_vs_known_query_px']['median'],
                  'heldout focal only', context['t0_focal_only_crossfit']['heldout_pixel_error']['median'])


if __name__ == '__main__':
    main()
