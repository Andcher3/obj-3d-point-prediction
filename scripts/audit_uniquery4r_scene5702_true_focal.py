"""CPU scene_5702 true-focal video and direct-2D source-coordinate audit.

No model forward, no 3D coordinate changes, no pointwise anchoring.
"""
import argparse
import json
import subprocess
import tarfile
from pathlib import Path

import cv2
import numpy as np

from render_uniquery4r_movif_focal_review import rays, project, diagnostic, sha, stats
from render_uniquery4r_movif_gt_review import panel


def errors(a, b):
    return stats(np.linalg.norm(a-b, axis=-1))


def audit_2d(raw, m, gt):
    t0 = m['query_model_frame_index']
    pairs = raw['pair_idx'].tolist()
    slots = [pairs.index([t0, t]) for t in range(22)]
    source = raw['tracks_2d_source_pixels'].astype(np.float64)
    warp = raw['warp2d'][slots].astype(np.float64)
    h, w = m['processed_resolution_hw']
    oh, ow = m['source_resolution_hw']
    # Rebuild the paper's absolute [-1,1] coordinate convention independently.
    rebuilt = ((warp+1)/2*[w-1,h-1]+.5)*[ow/w,oh/h]-.5
    np.testing.assert_allclose(source, rebuilt, atol=3e-5)
    supplied = raw['query_uv_normalized_01'].astype(np.float64)
    restored_query = (supplied*[w-1,h-1]+.5)*[ow/w,oh/h]-.5
    expected = np.array(m['query_uv_source_pixels'])
    np.testing.assert_allclose(restored_query, expected, atol=2e-5)
    np.testing.assert_allclose(expected, gt['tracks_2d'][t0], atol=1e-8)
    np.testing.assert_array_equal(raw['source_frame_indices'], np.arange(22))
    np.testing.assert_array_equal(raw['tracks_3d'][t0], raw['warp3d'][slots[t0]])
    true = gt['tracks_2d'].astype(np.float64)
    visible = gt['visibility'].astype(bool)
    delta_error = (source-source[t0])-(true-true[t0])
    # 2D/3D both read from the same source-target pair, not swapped tracks.
    uv3 = raw['tracks_3d_projected_source_pixels'].astype(np.float64)
    # Per-axis spread around point-cloud center, independent of image center.
    spread = np.std(source[t0], axis=0)/np.std(expected, axis=0)
    bias = np.mean(source[t0]-expected, axis=0)
    source_motion = np.mean(source-source[t0], axis=1)
    true_motion = np.mean(true-true[t0], axis=1)
    return {
        'paper_warp2d_convention': 'absolute_target_coordinate_normalized_minus1_plus1',
        't0_pair': pairs[slots[t0]], 't0_pair_index': slots[t0],
        't0_query_to_gt_max_px': float(np.max(np.linalg.norm(expected-true[t0],axis=-1))),
        'input_uv_roundtrip_max_px': float(np.max(np.linalg.norm(restored_query-expected,axis=-1))),
        'raw_warp2d_to_cached_pixels_max_px': float(np.max(np.linalg.norm(source-rebuilt,axis=-1))),
        'direct_2d_t0_error_px': errors(source[t0], expected),
        'direct_2d_t0_centroid_bias_xy_px': bias.tolist(),
        'direct_2d_t0_centered_spread_ratio_xy': spread.tolist(),
        'direct_2d_future_visible_absolute_error_px': errors(source[7:][visible[7:]],true[7:][visible[7:]]),
        'direct_2d_future_visible_displacement_error_px': stats(np.linalg.norm(delta_error[7:][visible[7:]],axis=-1)),
        'centroid_motion_error_px': stats(np.linalg.norm(source_motion[7:]-true_motion[7:],axis=-1)),
        'direct2d_vs_original3d_projection_t0_error_px': errors(source[t0],uv3[t0]),
        'direct2d_vs_original3d_projection_future_error_px': errors(source[7:],uv3[7:]),
        'original_camera_intrinsics_used_in_direct_2d': False,
        'no_identity_override_at_t0': True,
    }


def run(base, name='scene_5702', package=True):
    import imageio_ffmpeg
    directory, output = base/'inputs'/name, base/'outputs'/name
    raw_path, gt_path = output/f'{name}_22frames_raw.npz', directory/'gt_tracks.npz'
    raw_sha, gt_sha = sha(raw_path), sha(gt_path)
    m = json.loads((directory/'queries.json').read_text())
    with np.load(raw_path,allow_pickle=False) as raw, np.load(gt_path,allow_pickle=False) as gt:
        xyz = raw['tracks_3d'].astype(np.float64)
        k = raw['intrinsic'].astype(np.float64)
        e = raw['extrinsic'].astype(np.float64)
        original, refit, result = diagnostic(xyz,e,k,m,gt['tracks_2d'],gt['visibility'],gt['intrinsic_gt'])
        oh, ow = m['source_resolution_hw']
        h, w = m['processed_resolution_hw']
        true_k = k.copy()
        true_k[:,0,0] = gt['intrinsic_gt'][:,0,0]*w/ow
        true_k[:,1,1] = gt['intrinsic_gt'][:,1,1]*h/oh
        true_projection = project(rays(xyz,e),true_k,m)
        result['direct_2d_audit'] = audit_2d(raw,m,gt)
        tracks = [raw['tracks_2d_source_pixels'],original,refit,true_projection,gt['tracks_2d']]
        visibility = gt['visibility'].copy()
        # Exact finite-source shape, full frame list, all 48 queries retained.
        assert all(x.shape == (22,48,2) for x in tracks)
    video = output/f'{name}_true_focal_review.mp4'
    if video.exists():
        raise FileExistsError(video)
    command = [imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-n',
               '-f','rawvideo','-pix_fmt','bgr24','-s','2560x560','-r','8','-i','pipe:0',
               '-an','-c:v','libx264','-preset','fast','-crf','20','-threads','2',
               '-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    proc = subprocess.Popen(command,stdin=subprocess.PIPE)
    samples = []
    try:
        for t in range(22):
            rgb = cv2.imread(str(directory/'rgb'/f'{t:03d}.png'))
            assert rgb is not None and rgb.shape[:2] == (256,256)
            titles = ['Model: direct 2D','3D: original camera','3D: t0 focal refit',
                      '3D: true fx/fy (pred pose)','Simulator GT (hollow=hidden)']
            canvas = np.hstack([panel(rgb,uv,t,title,visibility if j==4 else None)
                               for j,(uv,title) in enumerate(zip(tracks,titles))])
            proc.stdin.write(canvas.tobytes())
            if t==6:
                assert cv2.imwrite(str(output/f'{name}_true_focal_review_t0.png'),canvas)
            if t in (0,6,13,21):
                samples.append(cv2.resize(canvas,(1920,420)))
    finally:
        proc.stdin.close()
    if proc.wait(timeout=90) != 0:
        raise RuntimeError('Video encoder failed')
    assert cv2.imwrite(str(output/f'{name}_true_focal_review_contact.jpg'),np.vstack(samples))
    if sha(raw_path)!=raw_sha or sha(gt_path)!=gt_sha:
        raise ValueError('Source arrays changed')
    cap = cv2.VideoCapture(str(video))
    decoded = 0
    while cap.read()[0]:
        decoded+=1
    cap.release()
    assert decoded==22
    result.update(scene_id=name, raw_xyz_unchanged=True, raw_sha256=raw_sha,gt_sha256=gt_sha,
                  model_forward_called=False, true_focal_uses_predicted_extrinsics_and_principal_points=True,
                  true_camera_pose_not_substituted=True,frames=decoded)
    diagnostic_path=output/f'{name}_true_focal_audit.json'
    diagnostic_path.write_text(json.dumps(result,indent=2)+'\n')
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>scene_5702：真实焦距对照</title><style>body{font:16px "Microsoft YaHei",sans-serif;margin:24px;background:#10151d;color:#edf2fa}p{line-height:1.7}video,img{width:100%;display:block}a{color:#86baff}summary{cursor:pointer}</style>
<h1>scene_5702：真实焦距投影与二维输出检查</h1>
<p>从左到右：模型二维输出 / 原三维回投影 / t0拟合焦距回投影 / 仿真真实焦距回投影 / 仿真真值。</p>
<p>第四列只用真实fx/fy替换预测焦距。三维点、预测相机位置/朝向与主点保持不变。第五列使用完整真实几何和真实相机，因此第四列不保证等于第五列。</p>
<p>源256×256，显示2倍，12fps数据、8fps慢放，t0=6；没有GPU推理或训练，没有强制将输出移回query。</p>'''
    root=f'outputs/{name}'
    document+=f'<video controls loop preload="metadata" src="{root}/{video.name}"></video><details open><summary>t0 对照图</summary><img src="{root}/{name}_true_focal_review_t0.png"></details><details><summary>分帧总览</summary><img src="{root}/{name}_true_focal_review_contact.jpg"></details>'
    document+='<p><a href="focal-review.html#scene_5702">返回18样本焦距对照</a></p></html>'
    document=document.replace('scene_5702',name)
    page=base/f'{name}-true-focal.html'
    page.write_text(document,encoding='utf-8')
    if not package:
        print(json.dumps({'scene_id':name,'rendered_frames':decoded}),flush=True)
        return result
    members=[video,diagnostic_path,page,output/f'{name}_true_focal_review_t0.png',output/f'{name}_true_focal_review_contact.jpg']
    manifest=base/'logs/scene_5702-true-focal-transfer.json'
    manifest.write_text(json.dumps({'sha256':{p.relative_to(base).as_posix():sha(p) for p in members}},indent=2)+'\n')
    archive=base/'scene_5702-true-focal.tar.gz'
    if archive.exists():
        raise FileExistsError(archive)
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for p in members+[manifest]:
            tar.add(p,arcname=p.relative_to(base).as_posix(),recursive=False)
    print(json.dumps({'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),
                      'direct_2d':result['direct_2d_audit'],'true_focal_only':result['true_focal_only']}),flush=True)
    return result


def verify(base):
    manifest=json.loads((base/'logs/scene_5702-true-focal-transfer.json').read_text())
    for relative,expected in manifest['sha256'].items():
        if sha(base/relative)!=expected:
            raise ValueError('File hash mismatch: '+relative)
    video=base/'outputs/scene_5702/scene_5702_true_focal_review.mp4'
    cap=cv2.VideoCapture(str(video))
    if (int(cap.get(3)),int(cap.get(4)))!=(2560,560):
        raise ValueError('Bad video resolution')
    count=0
    while cap.read()[0]:
        count+=1
    cap.release()
    if count!=22:
        raise ValueError('Incomplete video')
    report={'all_file_sha256_passed':True,'video_decode_passed':True,'frames':count,
            'resolution':[2560,560],'raw_arrays_preserved_and_audited_on_remote':True,
            'model_forward_called':False}
    (base/'logs/scene_5702-true-focal-local-checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--base',type=Path,required=True)
    ap.add_argument('--verify-only',action='store_true')
    args=ap.parse_args()
    verify(args.base) if args.verify_only else run(args.base)
