"""Read-only audit of supplied round-two artifacts; no L1 inference."""
from pathlib import Path
import csv, hashlib, importlib.util, io, json, re, subprocess, sys
from collections import Counter, defaultdict
import numpy as np
from PIL import Image, ImageDraw

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'experiences_results/L1-perturbation/complete-search'
CODE = ROOT / 'experiences_results/l1_latent_perturbation_code'
OUT = Path(__file__).resolve().parent
FFMPEG = Path('D:/ffmpeg-7.0.2-essentials_build/bin/ffmpeg.exe')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name, obj): (OUT/name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
paths = sorted([p for root in (SRC,CODE) for p in root.rglob('*') if p.is_file()])
manifest = {str(p.relative_to(ROOT)): {'sha256':sha(p), 'bytes':p.stat().st_size} for p in paths}
save('source_manifest.json', manifest)
spec = importlib.util.spec_from_file_location('audited_complete_scheme', CODE/'complete_scheme.py')
scheme = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = scheme
spec.loader.exec_module(scheme)

def loadvec(p):
    with np.load(p, allow_pickle=False) as f: return f['latents'].copy()
def dist(a,b):
    n=min(len(a),len(b)); a=a[:n].astype(float); b=b[:n].astype(float)
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-15))
rows=[]; summary={}; all_groups={}; issues=[]
for clip in ('cut_0','cut_1','cut_2'):
    folder=SRC/clip
    report=json.loads((folder/'complete_search_report.json').read_text(encoding='utf-8'))
    rr=[]; cc=[]; groups=defaultdict(list)
    for file in sorted(folder.glob('*_raw.txt')):
        bid=file.name.removesuffix('_raw.txt')
        match=re.fullmatch(r'r(\d+)_(\d+)_sigma_([^_]+)_seed_(\d+)',bid)
        rnd,idx,sigma,seed=match.groups(); rnd=int(rnd); sigma=float(sigma)
        raw=file.read_text(encoding='utf-8')
        m=re.search(r'<answer>\s*(.*?)\s*</answer>',raw,re.S|re.I)
        answer=' '.join(m.group(1).split()) if m else ''
        vec=loadvec(folder/f'{bid}_latents.npz')
        valid=bool(answer and '<|latent_start|>' in raw and '<|latent_end|>' in raw and vec.size and np.isfinite(vec).all())
        prefix=raw.split('<|latent_start|>')[0]
        item={'clip':clip,'id':bid,'round':rnd,'sigma':sigma,'seed':int(seed),'complete_answer':bool(m),
              'valid':valid,'latent_shape':list(vec.shape),'latent_finite':bool(np.isfinite(vec).all()),
              'latent_blocks':raw.count('<|latent_start|>'),'im_end': '<|im_end|>' in raw,
              'selected':bid in report['retained'],'answer':answer,'prefix':prefix,
              'raw_sha256':sha(file),'latent_array_sha256':hashlib.sha256(vec.tobytes()).hexdigest()}
        if rnd==0 and int(idx)==0: basevec=vec; baseraw=raw; baseanswer=answer; baseprefix=prefix
        item.update(raw_changed_vs_initial=raw!=baseraw,answer_changed_vs_initial=answer!=baseanswer,
                    prefix_changed_vs_initial=prefix!=baseprefix,latent_relative_l2_vs_initial=dist(basevec,vec))
        rr.append(item)
        cc.append(scheme.BranchCandidate(bid,answer,sigma,int(seed),latent=vec,raw=raw,valid_format=valid))
        groups[answer].append(bid)
    comps=[c.__dict__ for c in scheme.compare_events(cc,target_object=report['target_object'])]
    if comps!=report['comparisons']: issues.append(f'{clip}: pairwise comparison mismatch')
    initial=[c for c in cc if c.branch_id.startswith('r0_')]
    final=[c.branch_id for c in scheme.retain_joint_branches(cc,branch_count=3,target_object=report['target_object'])]
    initial_selected=scheme.retain_joint_branches(initial,branch_count=3,target_object=report['target_object'])
    prompt=scheme.build_feedback_prompt(question='What will happen next in the video?',target_object=report['target_object'],retained=initial_selected,round_index=1)
    if final!=report['retained']: issues.append(f'{clip}: retention mismatch')
    if prompt!=report['rounds'][0]['prompt']: issues.append(f'{clip}: prompt mismatch')
    log=(folder/'run.log').read_text(encoding='utf-8')
    try: logreport=json.loads(log[log.index('{\n  "video_path"'):]); logmatches=logreport==report
    except Exception: logmatches=False
    a,b=rr[:2]
    summary[clip]={
       'target_object':report['target_object'],'runs':len(rr),'valid':sum(r['valid'] for r in rr),
       'baseline_raw_equal':a['raw_sha256']==b['raw_sha256'],
       'baseline_saved_latents_equal':a['latent_array_sha256']==b['latent_array_sha256'],
       'initial_noise_latent_changed':sum(r['latent_relative_l2_vs_initial']>0 for r in rr if r['round']==0 and r['sigma']>0),
       'initial_noise_answer_changed':sum(r['answer_changed_vs_initial'] for r in rr if r['round']==0 and r['sigma']>0),
       'initial_unique_answers':len({r['answer'] for r in rr if r['round']==0}),
       'feedback_unique_answers':len({r['answer'] for r in rr if r['round']==1}),
       'feedback_new_unique_answers':len({r['answer'] for r in rr if r['round']==1}-{r['answer'] for r in rr if r['round']==0}),
       'total_unique_answers':len(groups),'selected':final,'initial_selected':[c.branch_id for c in initial_selected],
       'initial_prefix_variants':len({r['prefix'] for r in rr if r['round']==0}),
       'feedback_prefix_variants':len({r['prefix'] for r in rr if r['round']==1}),
       'pairwise_count':len(comps),'pairwise_recomputed_match':comps==report['comparisons'],
       'prompt_recomputed_match':prompt==report['rounds'][0]['prompt'], 'retention_recomputed_match':final==report['retained'],
       'log_report_match':logmatches,'feedback_prompt':prompt,
       'selected_pairwise':[c for c in comps if c['left_id'] in final and c['right_id'] in final]}
    all_groups[clip]=[{'answer':a,'count':len(ids),'ids':ids,'selected_ids':[i for i in ids if i in final]} for a,ids in groups.items()]
    rows.extend(rr)
    video=SRC/f'{clip}.mp4'
    probe=json.loads(subprocess.check_output([str(FFMPEG.with_name('ffprobe.exe')),'-v','error','-show_format','-show_streams','-of','json',str(video)]))
    vs=next(s for s in probe['streams'] if s['codec_type']=='video'); dur=float(probe['format']['duration'])
    summary[clip]['video']={'duration':dur,'frames':vs.get('nb_frames'),'width':vs['width'],'height':vs['height'],'fps':vs['avg_frame_rate'], 'sha256':sha(video),
                           'public_copy_hash_matches':sha(video)==sha(SRC.parent/'public-action-samples'/video.name)}
    times=np.linspace(0,max(0,dur-.06),12)
    sheet=Image.new('RGB',(4*400,3*250),(240,240,240)); draw=ImageDraw.Draw(sheet)
    for i,t in enumerate(times):
        png=subprocess.check_output([str(FFMPEG),'-v','error','-ss',f'{t:.4f}','-i',str(video),'-frames:v','1','-f','image2pipe','-vcodec','png','-'])
        im=Image.open(io.BytesIO(png)).convert('RGB'); im.thumbnail((400,225))
        x=(i%4)*400; y=(i//4)*250; sheet.paste(im,(x+(400-im.width)//2,y+23))
        draw.text((x+6,y+4),f'{clip} review frame t={t:.3f}s',fill=(0,0,0))
    sheet.save(OUT/f'{clip}_observation.jpg',quality=90)
    summary[clip]['review_frame_times']=times.tolist()
save('runs.json',rows); save('answer_groups.json',all_groups)
with (OUT/'runs.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

tests=[
 ('paraphrase','The hand lifts the cup.','The person picks up the mug.'),
 ('order-reversal','The hand lifts the cup then places it on the table.','The hand places the cup on the table then lifts it.'),
 ('negation','The hand moves the cup.','The hand does not move the cup.'),
]
checks=[]
for name,a,b in tests:
    score=scheme.event_similarity(a,b,target_object='the cup')
    checks.append({'case':name,'a':a,'b':b,'similarity':score,'same_event':score>=.5})
duplicates=[scheme.BranchCandidate(f'b{i}','The hand lifts the cup.',.01,i) for i in range(4)]
checks.append({'case':'identical_answer_retention','selected':[c.branch_id for c in scheme.retain_joint_branches(duplicates,branch_count=3,target_object='the cup')]})
save('comparator_checks.json',checks)
after={str(p.relative_to(ROOT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in paths}
result={'samples':summary,'issues':issues,'source_file_count':len(manifest),'source_unchanged':manifest==after,
        'model_rerun':False,'review_frames_are_model_sampled_frames':False,
        'note':'Runtime model patch, actual sampled indices and per-run token counts are absent from this supplied code/results bundle.'}
save('audit_summary.json',result)
print(json.dumps({'samples':{k:{a:b for a,b in v.items() if a not in ('feedback_prompt','selected_pairwise','review_frame_times')} for k,v in summary.items()},'issues':issues,'source_unchanged':manifest==after,'comparator_checks':checks},indent=2,ensure_ascii=False))
