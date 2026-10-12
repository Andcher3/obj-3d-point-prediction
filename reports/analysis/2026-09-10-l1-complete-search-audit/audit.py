"""Read-only audit of supplied round-two outputs; no model inference or source edits."""
from pathlib import Path
import collections, csv, hashlib, importlib.util, json, re, sys
import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SRC = ROOT / 'experiences_results/L1-perturbation/complete-search'
CODE = ROOT / 'experiences_results/l1_latent_perturbation_code'
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
files = sorted([p for base in (SRC, CODE) for p in base.rglob('*') if p.is_file()])
before = {str(p.relative_to(ROOT)): digest(p) for p in files}
spec = importlib.util.spec_from_file_location('supplied_complete_scheme', CODE/'complete_scheme.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

rows, summaries = [], []
for directory in sorted(p for p in SRC.iterdir() if p.is_dir()):
    report = json.loads((directory/'complete_search_report.json').read_text(encoding='utf-8'))
    raw_map, latent_map, candidates, local_rows = {}, {}, [], []
    for f in sorted(directory.glob('*_raw.txt')):
        branch_id = f.name.removesuffix('_raw.txt')
        round_id, index, sigma, seed = re.fullmatch(r'r(\d+)_(\d+)_sigma_([\d.]+)_seed_(\d+)', branch_id).groups()
        raw = f.read_text(encoding='utf-8')
        match = re.search(r'<answer>(.*?)</answer>', raw, re.S)
        answer = match.group(1).strip() if match else ''
        latent = np.load(directory/(branch_id+'_latents.npz'))['latents']
        valid = bool(answer and '<|latent_start|>' in raw and '<|latent_end|>' in raw and np.isfinite(latent).all())
        raw_map[branch_id], latent_map[branch_id] = raw, latent
        initial_reason = raw.split('<|latent_start|>')[0]
        row = dict(video=directory.name, branch_id=branch_id, round=int(round_id), sigma=float(sigma), seed=int(seed),
                   answer=answer, valid_format=valid, eos='<|im_end|>' in raw, latent_nonempty=bool(latent.size),
                   latent_shape=list(latent.shape), latent_finite=bool(np.isfinite(latent).all()),
                   first_reason_sha256=hashlib.sha256(initial_reason.encode()).hexdigest(),
                   retained=branch_id in report['retained'])
        local_rows.append(row)
        candidates.append(module.BranchCandidate(branch_id,answer,float(sigma),int(seed),raw=raw,valid_format=valid))
    baseline_ids = [r['branch_id'] for r in local_rows if r['round']==0 and r['sigma']==0]
    b = latent_map[baseline_ids[0]].astype(np.float64)
    for row in local_rows:
        l = latent_map[row['branch_id']].astype(np.float64)
        n = min(len(b), len(l)); left, right = b[:n],l[:n]
        row['latent_equal_baseline'] = bool(np.array_equal(b,l))
        row['aligned_vectors'] = n
        row['latent_relative_l2'] = float(np.linalg.norm(right-left)/max(np.linalg.norm(left),1e-12))
        row['answer_changed_baseline'] = row['answer'] != local_rows[0]['answer']
    target=report['target_object']
    comparisons=[x.__dict__ for x in module.compare_events(candidates,target_object=target)]
    selection=module.retain_joint_branches(candidates,branch_count=3,target_object=target)
    initial_selection=module.retain_joint_branches(candidates[:12],branch_count=3,target_object=target)
    prompt=module.build_feedback_prompt(question='What will happen next in the video?',target_object=target,retained=initial_selection,round_index=1)
    retained_rows=[next(r for r in local_rows if r['branch_id']==bid) for bid in report['retained']]
    round_summaries=[]
    for rnd in (0,1):
        rr=[r for r in local_rows if r['round']==rnd]
        round_summaries.append(dict(round=rnd,count=len(rr),unique_answers=len(set(r['answer'] for r in rr)),
          first_reason_count=len(set(r['first_reason_sha256'] for r in rr)),
          nonzero_count=sum(r['sigma']>0 for r in rr),
          nonzero_answer_changed_baseline=sum(r['sigma']>0 and r['answer_changed_baseline'] for r in rr),
          groups=[dict(answer=a,branch_ids=[r['branch_id'] for r in rr if r['answer']==a]) for a in dict.fromkeys(r['answer'] for r in rr)]))
    log=(directory/'run.log').read_text(encoding='utf-8')
    logged=json.loads(log[log.index('{'):])
    summaries.append(dict(video=directory.name,target=target,video_sha256=digest(SRC/(directory.name+'.mp4')),
      candidate_count=len(local_rows),reported_count=report['candidate_count'],
      all_valid=all(r['valid_format'] for r in local_rows),all_eos=all(r['eos'] for r in local_rows),
      all_latents_nonempty_finite=all(r['latent_nonempty'] and r['latent_finite'] for r in local_rows),
      baseline_raw_equal=raw_map[baseline_ids[0]]==raw_map[baseline_ids[1]],
      baseline_array_equal=bool(np.array_equal(latent_map[baseline_ids[0]],latent_map[baseline_ids[1]])),
      round_summaries=round_summaries,retained=retained_rows,
      unique_retained_answers=len(set(r['answer'] for r in retained_rows)),
      all_comparisons_reproduced=comparisons==report['comparisons'],comparison_count=len(comparisons),
      selection_reproduced=[c.branch_id for c in selection]==report['retained'],
      feedback_prompt_reproduced=prompt==report['rounds'][0]['prompt'],
      report_matches_log=report==logged,trajectory_decoder=report['trajectory_decoder']))
    rows.extend(local_rows)

# Small deterministic probes of the supplied selection/comparison implementation.
duplicate_candidates=[module.BranchCandidate(f'b{i}','The hand lifts the cup.',0.1,i) for i in range(4)]
duplicate_selection=module.retain_joint_branches(duplicate_candidates,branch_count=3)
probes=dict(
  identical_answers_selected=[c.branch_id for c in duplicate_selection],
  identical_answers_unique=len(set(c.answer for c in duplicate_selection)),
  same_event_different_wording_similarity=module.event_similarity('The hand lifts the cup.','A person picks up a mug.'),
  opposite_order_similarity=module.event_similarity('The hand lifts the cup then places it on the table.','The hand places the cup on the table then lifts it.'),
  empty_latent_passes_finiteness=bool(np.isfinite(np.empty((0,0))).all()),
  notes='Diagnostic counterexamples, not a semantic benchmark. No source modification.')
after = {str(p.relative_to(ROOT)): digest(p) for p in files}
assert before==after
result=dict(verification_status='ANALYZED',model_inference_rerun=False,source_files=len(files),source_files_unchanged=before==after,
            candidate_total=len(rows),samples=summaries,code_probes=probes)
(OUT/'audit_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'source_manifest.json').write_text(json.dumps(before,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'run_audit.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
with (OUT/'run_audit.csv').open('w',encoding='utf-8-sig',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps({**{k:v for k,v in result.items() if k!='samples'},'samples':[{k:v for k,v in s.items() if k not in ('round_summaries','retained')} for s in summaries]},ensure_ascii=False,indent=2))
