import json, numpy as np
from pathlib import Path
ROOT=Path('outputs/matrix')
IN=Path('inputs/tracks')
video_stems=['10790317943435558913','gear_v2_9-wrist_front_rgb_img-0']
seeds=[42,43,44]
all_rows=[]; summary={}
for stem in video_stems:
    preds=[]; rows=[]
    gt=None; gv=None
    for seed in seeds:
        d=ROOT/stem/f'seed_{seed}'/stem
        p=np.load(d/'pred_tracks.npy').astype(np.float32)
        pv=np.load(d/'pred_visibility.npy').astype(bool)
        if gt is None:
            gt=np.load(d/'gt_tracks.npy').astype(np.float32)
            gv=np.load(d/'gt_visibility.npy').astype(bool)
        valid=gv & np.isfinite(gt).all(-1)
        dist=np.linalg.norm(p-gt,axis=-1)
        ade=float(dist[valid].mean())
        fvalid=valid[-1]
        fde=float(dist[-1,fvalid].mean())
        pred_path=np.linalg.norm(np.diff(p,axis=0),axis=-1)
        gt_path=np.linalg.norm(np.diff(gt,axis=0),axis=-1)
        pairvalid=valid[1:]&valid[:-1]
        row={'video':stem,'seed':seed,'ade_px':ade,'fde_px':fde,
             'pred_path_px_mean':float(pred_path[pairvalid].mean()),
             'gt_path_px_mean':float(gt_path[pairvalid].mean()),
             'pred_visible_rate':float(pv.mean()),'gt_visible_rate':float(gv.mean())}
        rows.append(row); all_rows.append(row); preds.append(p)
    # static: last observed track frame 80
    z=np.load(IN/f'{stem}.npz'); tr=z['tracks'].astype(np.float32); vis=z['vis'].astype(bool)
    if tr.shape[0]==390: tr=np.transpose(tr,(1,0,2)); vis=vis.T
    static=np.repeat(tr[80:81],81,axis=0)
    valid=gv & np.isfinite(gt).all(-1)
    sd=np.linalg.norm(static-gt,axis=-1)
    static_ade=float(sd[valid].mean()); static_fde=float(sd[-1,valid[-1]].mean())
    pairs=[]
    for i in range(len(preds)):
      for j in range(i+1,len(preds)):
        dd=np.linalg.norm(preds[i]-preds[j],axis=-1)
        pairs.append({'seeds':[seeds[i],seeds[j]],'pairwise_ade_px':float(dd[valid].mean()),'pairwise_fde_px':float(dd[-1,valid[-1]].mean())})
    summary[stem]={'runs':rows,'best_of_3_ade_px':min(r['ade_px'] for r in rows),'best_of_3_fde_px':min(r['fde_px'] for r in rows),
                   'static_ade_px':static_ade,'static_fde_px':static_fde,
                   'mean_pairwise_ade_px':float(np.mean([x['pairwise_ade_px'] for x in pairs])),
                   'mean_pairwise_fde_px':float(np.mean([x['pairwise_fde_px'] for x in pairs])),
                   'pairs':pairs}
out={'protocol':{'videos':video_stems,'seeds':seeds,'steps':20,'history_frames':81,'future_frames':81,'resolution':[832,480],'text_conditioning':False,'video_conditioning':True},'videos':summary}
Path('analysis').mkdir(exist_ok=True)
Path('analysis/metrics.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
