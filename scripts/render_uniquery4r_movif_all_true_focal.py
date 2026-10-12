"""Render actual true-focal comparisons, summarize focal accuracy, CPU only."""
import argparse
import csv
import io
import json
import tarfile
from pathlib import Path

import cv2
import numpy as np

from audit_uniquery4r_scene5702_true_focal import run
from render_uniquery4r_movif_focal_review import sha


def summarize(rows):
    original=np.array([r['original_focal_relative_error_to_gt_t0'] for r in rows])
    refit=np.array([r['refit_focal_relative_error_to_gt_t0'] for r in rows])
    mean_original=original.mean(1)
    mean_refit=refit.mean(1)
    pixel={k:float(np.mean([r[k]['future_visible_error_px']['mean'] for r in rows]))
           for k in ('original','refit','true_focal_only')}
    return {
        'scene_count':len(rows),
        'focal_comparison_time':'t0=6; both estimates compared against same true camera',
        'focal_error_definition':'abs(estimate-true)/true; joint error=mean(fx_error,fy_error)',
        't0_original_focal_relative_error_macro_xy':original.mean(0).tolist(),
        't0_refit_focal_relative_error_macro_xy':refit.mean(0).tolist(),
        't0_original_focal_relative_error_macro_joint':float(mean_original.mean()),
        't0_refit_focal_relative_error_macro_joint':float(mean_refit.mean()),
        't0_refit_closer_to_gt_count_xy':(refit<original).sum(0).tolist(),
        't0_refit_closer_to_gt_count_joint':int((mean_refit<mean_original).sum()),
        't0_original_closer_to_gt_count_joint':int((mean_original<mean_refit).sum()),
        'future_visible_projection_scene_macro_px':pixel,
        'true_focal_projection_better_than_original_count':sum(r['true_focal_only']['future_visible_error_px']['mean']<r['original']['future_visible_error_px']['mean'] for r in rows),
        'true_focal_projection_better_than_refit_count':sum(r['true_focal_only']['future_visible_error_px']['mean']<r['refit']['future_visible_error_px']['mean'] for r in rows),
        'raw_xyz_unchanged':True,'model_forward_called':False,'fit_uses_future':False,
        'all_true_focal_projections_keep_predicted_pose_and_principal_point':True,
        'scenes':rows,
    }


def make_gallery(base, report):
    headers=['场景','真实 fx/fy','预测 fx/fy','拟合 fx/fy','预测平均相对误差','拟合平均相对误差','更接近真实焦距']
    table=[]
    csv_rows=[]
    sections=[]
    for r in report['scenes']:
        name=r['scene_id']
        focal=[r[f'focal_t0_{k}_processed_px'] for k in ('groundtruth','original','refit')]
        original=sum(r['original_focal_relative_error_to_gt_t0'])/2
        refit=sum(r['refit_focal_relative_error_to_gt_t0'])/2
        closer='t0拟合' if refit<original else '模型预测' if original<refit else '相同'
        columns=[f'<a href="#{name}">{name}</a>']+[f'{f[0]:.2f} / {f[1]:.2f}' for f in focal]+[f'{100*original:.2f}%',f'{100*refit:.2f}%',closer]
        table.append('<tr>'+''.join(f'<td>{x}</td>' for x in columns)+'</tr>')
        csv_rows.append([name,*focal[0],*focal[1],*focal[2],original,refit,closer])
        root=f'outputs/{name}'
        old,fit,true=(r[k]['future_visible_error_px']['mean'] for k in ('original','refit','true_focal_only'))
        sections.append(f'<section id="{name}"><h2>{name}</h2><p>未来可见点平均投影误差：预测焦距 {old:.2f} px / t0拟合焦距 {fit:.2f} px / 真实焦距 {true:.2f} px。</p>'
                        f'<video controls loop preload="metadata" src="{root}/{name}_true_focal_review.mp4"></video>'
                        f'<details><summary>查看 t0 图片与分帧总览</summary>'
                        f'<img loading="lazy" src="{root}/{name}_true_focal_review_t0.png">'
                        f'<img loading="lazy" src="{root}/{name}_true_focal_review_contact.jpg"></details></section>')
    m=report['future_visible_projection_scene_macro_px']
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MOVi-F：真实焦距与拟合焦距对照</title><style>body{font:15px "Microsoft YaHei",sans-serif;background:#10151d;color:#edf2fa;margin:24px}main{max-width:2560px;margin:auto}p{line-height:1.7}video,img{width:100%;display:block}a{color:#86baff}table{border-collapse:collapse;width:100%}td,th{padding:9px;border:1px solid #405065;text-align:left}section{margin:32px 0;border-top:1px solid #405065;padding:16px 0}summary{cursor:pointer;margin:12px 0}</style><main>
<h1>18个物体样本：真实焦距的五列对照</h1>
<p>从左到右：模型直接二维输出 / 原三维投影 / t0拟合焦距投影 / 真实焦距投影 / 仿真真值。相同颜色对应同一query。</p>
<p>第四列只替换fx/fy，保留原三维点、预测相机位置/朝向与主点。第五列使用真实几何和真实相机。因此第四列仍可能偏移；此实验不能单独判定三维点与相机姿态各自的误差。</p>
<p>表中焦距均为t0、第6帧、252×252处理图的像素单位；真实fx/fy均为275.625。相对误差=|估计值−真实值|/真实值，整体误差取fx/fy两轴平均，表中保留各轴原始值。t0拟合只用输入query，不使用未来真值。</p>'''
    text+=f'<p>18场景未来可见点平均投影误差：原预测 {m["original"]:.2f} px，t0拟合 {m["refit"]:.2f} px，真实焦距 {m["true_focal_only"]:.2f} px。原始XYZ及三维评分不变。</p>'
    text+='<p>源图256×256，显示放大2倍；原生12fps、8fps慢放，完整22帧离线跟踪。GPU保持关闭，无重新推理或训练。图片由交付视频解码，scene_5702沿用原无损图片。</p>'
    text+='<table><thead><tr>'+''.join(f'<th>{h}</th>' for h in headers)+'</tr></thead><tbody>'+''.join(table)+'</tbody></table>'
    text+=''.join(sections)+'<p><a href="focal-review.html">原四列对照页</a></p></main></html>'
    (base/'true-focal-review.html').write_text(text,encoding='utf-8')
    stream=io.StringIO()
    writer=csv.writer(stream)
    writer.writerow(['scene_id','gt_fx','gt_fy','pred_fx','pred_fy','fit_fx','fit_fy','pred_relative_error_mean','fit_relative_error_mean','closer'])
    writer.writerows(csv_rows)
    (base/'logs/true-focal-comparison.csv').write_text(stream.getvalue(),encoding='utf-8-sig')


def render_all(base):
    status=json.loads((base/'logs/gt-preparation-status.json').read_text())
    rows=[]
    for scene in status['scenes']:
        if scene['status']!='READY':
            continue
        name=scene['scene_id']
        existing=base/f'outputs/{name}/{name}_true_focal_audit.json'
        if name=='scene_5702' and existing.exists():
            r=json.loads(existing.read_text())
            if sha(base/f'outputs/{name}/{name}_22frames_raw.npz')!=r['raw_sha256'] or sha(base/f'inputs/{name}/gt_tracks.npz')!=r['gt_sha256']:
                raise ValueError('Existing scene_5702 source changed')
            print(json.dumps({'reused_verified_scene':name}),flush=True)
        else:
            r=run(base,name,package=False)
        rows.append(r)
    report=summarize(rows)
    report_path=base/'logs/all-true-focal-audit.json'
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    make_gallery(base,report)
    members=['logs/all-true-focal-audit.json','logs/true-focal-comparison.csv','true-focal-review.html']
    # scene_5702 is already checked and delivered locally; do not duplicate it.
    members += [f'outputs/{r["scene_id"]}/{r["scene_id"]}_true_focal_review.mp4' for r in rows if r['scene_id']!='scene_5702']
    manifest='logs/all-true-focal-transfer.json'
    (base/manifest).write_text(json.dumps({'sha256':{p:sha(base/p) for p in members},
        'reuses_scene_5702_local_video':True,'local_images_to_decode':True},indent=2)+'\n')
    archive=base/'all-true-focal-compact.tar.gz'
    if archive.exists():
        raise FileExistsError(archive)
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for p in members+[manifest]:
            tar.add(base/p,arcname=p,recursive=False)
    print(json.dumps({'archive_bytes':archive.stat().st_size,'archive_sha256_parts':[sha(archive)[:32],sha(archive)[32:]],
                      'summary':{k:v for k,v in report.items() if k!='scenes'}}),flush=True)


def verify(base):
    manifest=json.loads((base/'logs/all-true-focal-transfer.json').read_text())
    for relative,expected in manifest['sha256'].items():
        if sha(base/relative)!=expected:
            raise ValueError('Transfer file checksum mismatch: '+relative)
    report=json.loads((base/'logs/all-true-focal-audit.json').read_text())
    old_manifest=json.loads((base/'logs/scene_5702-true-focal-transfer.json').read_text())
    name='outputs/scene_5702/scene_5702_true_focal_review.mp4'
    if sha(base/name)!=old_manifest['sha256'][name]:
        raise ValueError('Existing scene_5702 local video changed')
    for r in report['scenes']:
        name=r['scene_id']
        out=base/'outputs'/name
        video=out/f'{name}_true_focal_review.mp4'
        cap=cv2.VideoCapture(str(video))
        if (int(cap.get(3)),int(cap.get(4)))!=(2560,560):
            raise ValueError('Video dimensions wrong: '+name)
        count=0
        samples=[]
        while True:
            ok,image=cap.read()
            if not ok:
                break
            if count==6 and not (out/f'{name}_true_focal_review_t0.png').exists():
                assert cv2.imwrite(str(out/f'{name}_true_focal_review_t0.png'),image)
            if count in (0,6,13,21):
                samples.append(cv2.resize(image,(1920,420)))
            count+=1
        cap.release()
        if count!=22:
            raise ValueError('Incomplete video: '+name)
        if not (out/f'{name}_true_focal_review_contact.jpg').exists():
            assert cv2.imwrite(str(out/f'{name}_true_focal_review_contact.jpg'),np.vstack(samples))
    checks={'videos_full_decode':True,'files_sha256_passed':True,'scenes':len(report['scenes']),
            'scene_5702_original_verified_and_reused':True,'frames_each':22,'resolution':[2560,560],
            'new_local_images_from_h264':True,'model_forward_called':False}
    (base/'logs/all-true-focal-local-checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='scenes'},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--base',type=Path,required=True)
    ap.add_argument('--verify-only',action='store_true')
    args=ap.parse_args()
    verify(args.base) if args.verify_only else render_all(args.base)
