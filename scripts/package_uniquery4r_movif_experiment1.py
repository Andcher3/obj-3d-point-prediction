"""Collect remote results, or verify and produce a local browser video gallery."""
import argparse
import hashlib
import html
import json
import subprocess
import tarfile
from pathlib import Path

import cv2


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def collect(base):
    audit=json.loads((base/'logs/gt-visual-render-audit.json').read_text())
    if not audit['render_completed'] or audit['scene_count']!=18:
        raise ValueError('All requested videos must complete before packaging')
    members=['logs/gt-visual-render-audit.json','logs/experiment1_gt_result_22frames.json',
             'logs/gt-preparation-status.json','logs/inference-input-audit.json',
             'logs/official-preprocessing-cpu.json','logs/source-audit.json',
             'logs/experiment1-gpu.log','logs/gpu-inference-exit-code.txt',
             'logs/gt-evaluation-exit-code.txt','query_contact_sheet.jpg','protocol.json']
    for r in audit['scenes']:
        name=r['scene_id']
        members.extend(f'outputs/{name}/{name}{suffix}' for suffix in
                       ['_gt_review.mp4','_gt_review_t0.png','_gt_review_contact.jpg',
                        '_22frames_raw.npz','_22frames_metadata.json'])
        members.append(f'outputs/{name}/gt_comparison_22frames.json')
        members.extend(f'inputs/{name}/{file}' for file in ['gt_tracks.npz','queries.json','sha256.json'])
    hashes={name:sha(base/name) for name in members}
    manifest='logs/visual-transfer-manifest.json'
    (base/manifest).write_text(json.dumps({'sha256':hashes,'source_rgb_and_shards_included':False},indent=2)+'\n')
    archive=base/'visual-review-transfer.tar.gz'
    if archive.exists():
        raise FileExistsError(archive)
    with tarfile.open(archive,'w:gz',compresslevel=3) as tar:
        for member in members+[manifest]:
            tar.add(base/member,arcname=member,recursive=False)
    result={'file':archive.name,'bytes':archive.stat().st_size,'sha256':sha(archive),
            'scene_count':18,'contains_source_rgb':False}
    (base/'logs/visual-transfer-archive.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


def compact(base):
    import imageio_ffmpeg
    audit=json.loads((base/'logs/gt-visual-render-audit.json').read_text())
    if not audit['render_completed'] or audit['scene_count']!=18:
        raise ValueError('All requested videos must complete')
    members=['logs/gt-visual-render-audit.json','logs/experiment1_gt_result_22frames.json',
             'logs/gt-preparation-status.json','logs/source-audit.json']
    for row in audit['scenes']:
        name=row['scene_id']
        source=base/f'outputs/{name}/{name}_gt_review.mp4'
        target=source.with_name(f'{name}_gt_review_browser.mp4')
        if not target.exists():
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error',
                            '-n','-i',str(source),'-c:v','libx264','-preset','fast','-crf','20',
                            '-threads','2','-pix_fmt','yuv420p','-movflags','+faststart','-an',str(target)],check=True)
        members.append(target.relative_to(base).as_posix())
        print(json.dumps({'compact_video':name,'bytes':target.stat().st_size}),flush=True)
    manifest='logs/visual-transfer-manifest.json'
    hashes={name:sha(base/name) for name in members}
    (base/manifest).write_text(json.dumps({'sha256':hashes,'source_rgb_and_shards_included':False,
        'raw_predictions_copied':False,'raw_predictions_preserved_on_remote':True},indent=2)+'\n')
    archive=base/'visual-review-compact.tar.gz'
    if archive.exists():
        raise FileExistsError(archive)
    with tarfile.open(archive,'w:gz',compresslevel=3) as tar:
        for member in members+[manifest]:
            tar.add(base/member,arcname=member,recursive=False)
    result={'file':archive.name,'bytes':archive.stat().st_size,'sha256':sha(archive),
            'scene_count':18,'raw_predictions_preserved_on_remote':True}
    (base/'logs/visual-transfer-compact.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


def local(base):
    manifest=json.loads((base/'logs/visual-transfer-manifest.json').read_text())
    for relative,expected in manifest['sha256'].items():
        if sha(base/relative)!=expected:
            raise ValueError('Transfer checksum failed: '+relative)
    status=json.loads((base/'logs/gt-preparation-status.json').read_text())
    records=json.loads((base/'logs/gt-visual-render-audit.json').read_text())['scenes']
    metrics=json.loads((base/'logs/experiment1_gt_result_22frames.json').read_text())
    details={r['scene_id']:r for r in metrics['per_scene']}
    assets={r['scene_id']:r['asset_id'] for r in status['scenes'] if r['status']=='READY'}
    sections=[]
    checks=[]
    for row in records:
        name=row['scene_id']
        directory=base/'outputs'/name
        source=directory/f'{name}_gt_review.mp4'
        target=directory/f'{name}_gt_review_browser.mp4'
        if not target.exists():
            subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-n','-i',str(source),
                            '-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p',
                            '-threads','2','-movflags','+faststart','-an',str(target)],check=True)
        capture=cv2.VideoCapture(str(target))
        shape=(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        decoded=0
        samples=[]
        while True:
            ok,image=capture.read()
            if not ok:
                break
            if decoded==6 and not (directory/f'{name}_gt_review_t0.png').exists():
                cv2.imwrite(str(directory/f'{name}_gt_review_t0.png'),image)
            if decoded in (0,6,13,21):
                samples.append(cv2.resize(image,(1152,420)))
            decoded+=1
        capture.release()
        if shape!=(1536,560) or frames!=22 or decoded!=22:
            raise ValueError('Browser video decode failed: '+name)
        contact=directory/f'{name}_gt_review_contact.jpg'
        if not contact.exists():
            import numpy as np
            cv2.imwrite(str(contact),np.vstack(samples))
        m=details[name]['scale_adjusted']['future']
        edge=100*m['relative_vector_error_bbox_fraction']['mean']
        length=100*m['relative_pair_length_error_fraction']['mean']
        root=f'outputs/{name}'
        sections.append(f'<section id="{name}"><h2>{name} · {html.escape(assets[name].replace("_"," "))}</h2>'
                        f'<p>后续点间向量平均误差：物体bbox对角线的 {edge:.2f}%；点间长度平均相对误差：{length:.2f}%。</p>'
                        f'<video controls loop preload="metadata" src="{root}/{target.name}"></video>'
                        f'<details><summary>查看 t0 对照图片与分帧总览</summary>'
                        f'<a href="{root}/{name}_gt_review_t0.png"><img loading="lazy" src="{root}/{name}_gt_review_t0.png"></a>'
                        f'<a href="{root}/{name}_gt_review_contact.jpg"><img loading="lazy" src="{root}/{name}_gt_review_contact.jpg"></a></details></section>')
        raw_path=directory/f'{name}_22frames_raw.npz'
        checks.append({'scene_id':name,'frames':decoded,'browser_video_sha256':sha(target),
                       'raw_prediction_sha256_unchanged':sha(raw_path)==row['raw_sha256'] if raw_path.exists() else None,
                       'raw_prediction_location':'local and remote' if raw_path.exists() else 'remote only'})
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>UniQuery4R / MOVi-F 真值对照</title><style>body{font:15px "Microsoft YaHei",sans-serif;margin:24px;background:#10151d;color:#edf2fa}
main{max-width:1550px;margin:auto}p{line-height:1.7;color:#cbd4df}video,img{width:100%;display:block}section{margin:32px 0;border-top:1px solid #405065;padding:16px 0}
h1{font-size:25px}h2{font-size:20px}a{color:#86baff}summary{cursor:pointer;margin:12px 0}nav{line-height:2.1}</style><main>
<h1>UniQuery4R：18 个物体样本与仿真真值对照</h1>
<p>每段从左至右：模型直接二维跟踪 / 模型三维点用估计相机回投影 / 仿真真值。相同颜色代表相同query；真值空心点表示近似可见性检查认为被遮挡或不可见，轨迹均未删点。</p>
<p>query 在第6帧（最后观测帧）采样；输入完整22帧，仅用于离线教师跟踪评价。原生12fps，视频以8fps慢放；原图256×256，仅为查看放大。完整输入跟踪不等同于未来预测。</p>
<p>模型轨迹与显示均为原始输出，未强制对齐、裁剪坐标或重新估计焦距。三维数值比较额外报告仅由t0确定且未来固定的整体尺度；图中回投影无需这种尺度调整。
向量误差除以物体真值bbox（三维包围盒）对角线；长度误差除以对应点对的真值长度。未训练关系模块，不能据此判断我们预测方法有效与否。</p>
<nav>'''+ ' · '.join(f'<a href="#{r["scene_id"]}">{r["scene_id"]}</a>' for r in records)+ '</nav>'+''.join(sections)+'</main></html>'
    (base/'visual-review.html').write_text(document,encoding='utf-8')
    for relative,expected in manifest['sha256'].items():
        if sha(base/relative)!=expected:
            raise ValueError('Packaging changed source artifact: '+relative)
    result={'local_transfer_sha256_passed':True,'all_browser_videos_decoded':True,
            'scene_count':len(checks),'scenes':checks,'model_forward_called':False}
    (base/'visual-packaging-checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'scene_count':len(checks),'gallery':str(base/'visual-review.html')}))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--base',type=Path,required=True)
    ap.add_argument('--collect',action='store_true')
    ap.add_argument('--compact',action='store_true')
    args=ap.parse_args()
    compact(args.base) if args.compact else collect(args.base) if args.collect else local(args.base)
