"""CPU-only focal diagnostic; preserve model XYZ and use t0 queries for fitting.

This is effective focal fitting, not guaranteed physical camera calibration.
Future GT is used for evaluation only. No model imports or forward calls.
"""
import argparse
import hashlib
import html
import json
import subprocess
import tarfile
from pathlib import Path

import cv2
import numpy as np

from render_uniquery4r_movif_gt_review import panel


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(x):
    a = np.asarray(x).ravel()
    finite = a[np.isfinite(a)]
    return dict(count=len(finite), total=len(a), mean=float(finite.mean()) if len(finite) else None,
                median=float(np.median(finite)) if len(finite) else None,
                p90=float(np.quantile(finite, .9)) if len(finite) else None)


def rays(xyz, extrinsic):
    c = np.einsum('tij,tqj->tqi', extrinsic[:, :3, :3], xyz) + extrinsic[:, None, :3, 3]
    with np.errstate(divide='ignore', invalid='ignore'):
        r = c[..., :2] / c[..., 2:3]
    r[c[..., 2] <= 0] = np.nan
    return r


def project(r, k, m):
    processed = r * np.stack((k[:, 0, 0], k[:, 1, 1]), -1)[:, None] + k[:, None, :2, 2]
    h, w = m['processed_resolution_hw']
    oh, ow = m['source_resolution_hw']
    return (processed + .5) * [ow/w, oh/h] - .5


def fit(r, expected, center):
    valid = np.isfinite(r).all(-1) & np.isfinite(expected).all(-1)
    denom = np.sum(r[valid] ** 2, axis=0)
    if valid.sum() < 3 or np.any(denom <= 1e-12):
        raise ValueError('Insufficient/degenerate camera rays for focal fit')
    f = np.sum(r[valid] * (expected[valid]-center), axis=0) / denom
    if not np.isfinite(f).all() or np.any(f <= 0):
        raise ValueError('Non-positive effective focal; do not clamp into success')
    return f


def diagnostic(xyz, e, k, m, gt_uv, vis, gt_k):
    t0 = m['query_model_frame_index']
    h, w = m['processed_resolution_hw']
    oh, ow = m['source_resolution_hw']
    expected = (np.asarray(m['query_uv_source_pixels']) + .5) * [w/ow, h/oh] - .5
    r = rays(xyz, e)
    focal = fit(r[t0], expected, k[t0, :2, 2])
    original = np.array([k[t0, 0, 0], k[t0, 1, 1]])
    ratio = focal / original
    corrected_k = k.copy()
    corrected_k[:, 0, 0] *= ratio[0]
    corrected_k[:, 1, 1] *= ratio[1]
    original_uv, corrected_uv = project(r, k, m), project(r, corrected_k, m)
    heldout = np.full(expected.shape, np.nan)
    folds = []
    for fold in range(4):
        test = np.arange(len(expected)) % 4 == fold
        f = fit(r[t0, ~test], expected[~test], k[t0, :2, 2])
        heldout[test] = r[t0, test] * f + k[t0, :2, 2]
        folds.append(f.tolist())
    heldout_source = (heldout + .5) * [ow/w, oh/h] - .5
    # GT focal is read only after the fit. Keep predicted extrinsics/principal point.
    true_k = k.copy()
    true_focal_processed = np.stack((gt_k[:, 0, 0] * w/ow, gt_k[:, 1, 1] * h/oh), -1)
    true_k[:, 0, 0], true_k[:, 1, 1] = true_focal_processed.T
    true_uv = project(r, true_k, m)
    result = {
        't0': t0, 'fit_query_count': len(expected),
        'focal_t0_original_processed_px': original.tolist(),
        'focal_t0_refit_processed_px': focal.tolist(),
        'focal_t0_groundtruth_processed_px': true_focal_processed[t0].tolist(),
        'fixed_focal_ratio_xy': ratio.tolist(),
        'original_focal_relative_error_to_gt_t0': (abs(original/true_focal_processed[t0]-1)).tolist(),
        'refit_focal_relative_error_to_gt_t0': (abs(focal/true_focal_processed[t0]-1)).tolist(),
        'four_fold_focals_processed_px': folds,
        'heldout_t0_query_error_px': stats(np.linalg.norm(heldout_source-gt_uv[t0], axis=-1)),
        'fit_uses_future': False, 'fit_uses_gt_camera': False,
        'camera_extrinsics_and_principal_points_unchanged': True,
    }
    for label, uv in [('original', original_uv), ('refit', corrected_uv), ('true_focal_only', true_uv)]:
        error = np.linalg.norm(uv-gt_uv, axis=-1)
        result[label] = {'t0_error_px': stats(error[t0]),
                         'future_visible_error_px': stats(error[t0+1:][vis[t0+1:]]),
                         'future_all_error_px': stats(error[t0+1:])}
    return original_uv, corrected_uv, result


def gallery(base, rows):
    sections = []
    for r in rows:
        name = r['scene_id']
        root = f'outputs/{name}'
        before, after = (r[k]['future_visible_error_px']['mean'] for k in ('original', 'refit'))
        fx, fy = r['fixed_focal_ratio_xy']
        sections.append(f'<section id="{name}"><h2>{name}</h2><p>未来可见点平均投影误差：'
                        f'{before:.2f} → {after:.2f} px；t0确定的焦距比例 fx={fx:.3f}, fy={fy:.3f}。</p>'
                        f'<video controls loop preload="metadata" src="{root}/{name}_focal_review.mp4"></video>'
                        f'<details><summary>t0 对照与分帧图</summary>'
                        f'<a href="{root}/{name}_focal_review_t0.png"><img loading="lazy" src="{root}/{name}_focal_review_t0.png"></a>'
                        f'<a href="{root}/{name}_focal_review_contact.jpg"><img loading="lazy" src="{root}/{name}_focal_review_contact.jpg"></a>'
                        f'</details></section>')
    content = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MOVi-F 焦距重估对照</title><style>body{font:15px "Microsoft YaHei",sans-serif;margin:24px;background:#10151d;color:#edf2fa}
main{max-width:2050px;margin:auto}p{line-height:1.7;color:#cbd4df}video,img{width:100%;display:block}section{margin:32px 0;border-top:1px solid #405065;padding:16px 0}
h1{font-size:25px}h2{font-size:20px}a{color:#86baff}summary{cursor:pointer;margin:12px 0}nav{line-height:2.1}</style><main>
<h1>18个物体样本：重估焦距的四列对照</h1>
<p>从左到右：模型直接二维跟踪 / 原三维回投影 / 重估焦距后的三维回投影 / 仿真真值。相同颜色代表同一query，真值空心点表示近似不可见。</p>
<p>只在第6帧用48个已知query像素位置拟合fx、fy；将得到的两个比例固定用于全22帧。没有用未来真值拟合，没有改三维点、相机位置/朝向或画面中心。第三列仅为投影诊断，不能视为三维轨迹校正；左列不会改变。</p>
<p>拟合焦距可能吸收三维点/相机姿态的误差，不能直接等同真实焦距。真实相机焦距只用于独立数值检查。四折留出检查在记录中，视频展示全48点拟合结果。</p>
<p>源图256×256，显示放大2倍；22帧离线完整输入跟踪，12fps数据、8fps慢放；所有点保留。原三列结果仍保留在 <a href="visual-review.html">原始对照页</a>。</p><nav>'''
    content += ' · '.join(f'<a href="#{r["scene_id"]}">{html.escape(r["scene_id"])}</a>' for r in rows)
    content += '</nav>' + ''.join(sections) + '</main></html>'
    (base/'focal-review.html').write_text(content, encoding='utf-8')


def render(base):
    import imageio_ffmpeg
    status = json.loads((base/'logs/gt-preparation-status.json').read_text())
    rows, members = [], []
    for scene in status['scenes']:
        if scene['status'] != 'READY':
            continue
        name = scene['scene_id']
        directory, output = base/'inputs'/name, base/'outputs'/name
        raw_path, gt_path = output/f'{name}_22frames_raw.npz', directory/'gt_tracks.npz'
        hashes = {'raw': sha(raw_path), 'gt': sha(gt_path), 'queries': sha(directory/'queries.json')}
        m = json.loads((directory/'queries.json').read_text())
        with np.load(raw_path, allow_pickle=False) as raw, np.load(gt_path, allow_pickle=False) as gt:
            xyz = raw['tracks_3d'].astype(np.float64)
            gt_uv, vis = gt['tracks_2d'].copy(), gt['visibility'].copy()
            old, new, record = diagnostic(xyz, raw['extrinsic'].astype(np.float64),
                                         raw['intrinsic'].astype(np.float64), m, gt_uv, vis, gt['intrinsic_gt'])
            np.testing.assert_allclose(old, raw['tracks_3d_projected_source_pixels'], atol=.003, equal_nan=True)
            tracks = [raw['tracks_2d_source_pixels'].copy(), old, new, gt_uv]
            np.testing.assert_array_equal(xyz, raw['tracks_3d'])
        video = output/f'{name}_focal_review.mp4'
        if video.exists():
            raise FileExistsError(video)
        command = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-n',
                   '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '2048x560', '-r', '8', '-i', 'pipe:0',
                   '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-threads', '2',
                   '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)]
        proc = subprocess.Popen(command, stdin=subprocess.PIPE)
        samples = []
        try:
            for t in range(22):
                rgb = cv2.imread(str(directory/'rgb'/f'{t:03d}.png'))
                if rgb is None or rgb.shape[:2] != (256, 256):
                    raise ValueError('Bad RGB frame')
                titles = ['Model: direct 2D', '3D: original camera', '3D: t0 focal refit (XYZ same)', 'Simulator GT (hollow=hidden)']
                canvas = np.hstack([panel(rgb, uv, t, title, vis if j == 3 else None)
                                    for j, (uv, title) in enumerate(zip(tracks, titles))])
                proc.stdin.write(canvas.tobytes())
                if t == 6:
                    if not cv2.imwrite(str(output/f'{name}_focal_review_t0.png'), canvas):
                        raise RuntimeError('Image write failed')
                if t in (0, 6, 13, 21):
                    samples.append(cv2.resize(canvas, (1536, 420)))
        finally:
            proc.stdin.close()
        if proc.wait(timeout=90) != 0:
            raise RuntimeError('H264 encode failed')
        if not cv2.imwrite(str(output/f'{name}_focal_review_contact.jpg'), np.vstack(samples)):
            raise RuntimeError('Contact write failed')
        if hashes != {'raw': sha(raw_path), 'gt': sha(gt_path), 'queries': sha(directory/'queries.json')}:
            raise ValueError('Source arrays changed')
        record.update(scene_id=name, source_sha256=hashes, raw_xyz_unchanged=True)
        rows.append(record)
        members.extend(f'outputs/{name}/{name}_focal_review{suffix}' for suffix in ('.mp4', '_t0.png', '_contact.jpg'))
        print(json.dumps({'scene': name, 'future_px_before': record['original']['future_visible_error_px']['mean'],
                          'future_px_after': record['refit']['future_visible_error_px']['mean']}), flush=True)
    macro = {}
    for label in ('original', 'refit', 'true_focal_only'):
        macro[label] = float(np.mean([r[label]['future_visible_error_px']['mean'] for r in rows]))
    result = {'scene_count': len(rows), 'scenes': rows, 'future_visible_scene_macro_px': macro,
              'raw_xyz_unchanged': True, 'model_forward_called': False, 'new_3d_labels_created': False,
              'method': 't0_only_two_focal_fit_fixed_ratios_future; four_fold_heldout_diagnostic'}
    target = 'logs/focal-review-audit.json'
    (base/target).write_text(json.dumps(result, indent=2)+'\n')
    members.append(target)
    gallery(base, rows)
    members.append('focal-review.html')
    manifest = {'sha256': {f: sha(base/f) for f in members}}
    (base/'logs/focal-review-transfer.json').write_text(json.dumps(manifest, indent=2)+'\n')
    members.append('logs/focal-review-transfer.json')
    archive = base/'focal-review.tar.gz'
    if archive.exists():
        raise FileExistsError(archive)
    with tarfile.open(archive, 'w:gz', compresslevel=1) as tar:
        for member in members:
            tar.add(base/member, arcname=member, recursive=False)
    print(json.dumps({'archive_bytes': archive.stat().st_size, 'archive_sha256': sha(archive), 'macro_px': macro}), flush=True)


def verify(base):
    manifest_path = base/'logs/focal-review-compact-transfer.json'
    if not manifest_path.exists():
        manifest_path = base/'logs/focal-review-transfer.json'
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest['sha256'].items():
        if sha(base/name) != expected:
            raise ValueError('Checksum failed: '+name)
    audit = json.loads((base/'logs/focal-review-audit.json').read_text())
    for scene in audit['scenes']:
        name = scene['scene_id']
        cap = cv2.VideoCapture(str(base/f'outputs/{name}/{name}_focal_review.mp4'))
        assert (int(cap.get(3)), int(cap.get(4))) == (2048, 560)
        count = 0
        samples = []
        output = base/'outputs'/name
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if count == 6 and not (output/f'{name}_focal_review_t0.png').exists():
                assert cv2.imwrite(str(output/f'{name}_focal_review_t0.png'), image)
            if count in (0, 6, 13, 21):
                samples.append(cv2.resize(image, (1536, 420)))
            count += 1
        cap.release()
        if count != 22:
            raise ValueError('Incomplete decode: '+name)
        if not (output/f'{name}_focal_review_contact.jpg').exists():
            assert cv2.imwrite(str(output/f'{name}_focal_review_contact.jpg'), np.vstack(samples))
    (base/'focal-review-local-checks.json').write_text(json.dumps({
        'sha256_verified': True, 'videos_full_decode': True, 'scene_count': audit['scene_count'],
        'frames_per_video': 22, 'resolution': [2048, 560], 'raw_arrays_verified_on_remote': True,
        'raw_arrays_downloaded': False,
        'local_images_from_h264_decode': manifest_path.name == 'focal-review-compact-transfer.json'}, indent=2)+'\n')
    print(json.dumps({'verified_videos': audit['scene_count'], 'macro': audit['future_visible_scene_macro_px']}))


def compact(base):
    audit = json.loads((base/'logs/focal-review-audit.json').read_text())
    members = ['logs/focal-review-audit.json', 'focal-review.html']
    members += [f'outputs/{r["scene_id"]}/{r["scene_id"]}_focal_review.mp4' for r in audit['scenes']]
    manifest = 'logs/focal-review-compact-transfer.json'
    (base/manifest).write_text(json.dumps({'sha256': {p: sha(base/p) for p in members},
        'local_images_to_be_decoded_from_h264': True}, indent=2)+'\n')
    archive = base/'focal-review-compact.tar.gz'
    if archive.exists():
        raise FileExistsError(archive)
    with tarfile.open(archive, 'w:gz', compresslevel=1) as tar:
        for member in members+[manifest]:
            tar.add(base/member, arcname=member, recursive=False)
    print(json.dumps({'archive_bytes': archive.stat().st_size, 'archive_sha256': sha(archive)}), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--verify-only', action='store_true')
    ap.add_argument('--collect-compact', action='store_true')
    args = ap.parse_args()
    compact(args.base) if args.collect_compact else verify(args.base) if args.verify_only else render(args.base)
