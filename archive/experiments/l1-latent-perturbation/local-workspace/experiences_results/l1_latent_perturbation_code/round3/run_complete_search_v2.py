#!/usr/bin/env python
"""Controlled Future-L1 event-branch experiment with NLI deduplication."""
from __future__ import annotations

import argparse, hashlib, json, sys, time
from pathlib import Path
from typing import Any
import numpy as np
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from complete_scheme_v2 import (BranchCandidate, TransformersNLI, assess_candidate,
    build_feedback_prompt, compare_events, select_unique_branches, serialise_candidate)
from experiments.l1_latent_perturbation.run_l1_latent_perturbation import (
    ANSWER_RE, LATENT_TOKEN_RE, SYSTEM_PROMPT, extract_latents, tensor_sha256)
from src.model.future_l1 import FutureL1_Qwen3VL
from src.model.future_l1 import ActiveLatentBranchPool
from src.train.monkey_patch_forward import replace_qwen3_vl_generation_forward, replace_qwen3_with_mixed_modality_forward


def sha256(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True); p.add_argument('--nli-model',required=True)
    p.add_argument('--video',required=True); p.add_argument('--target-object',required=True)
    p.add_argument('--target-aliases',default=''); p.add_argument('--output-dir',required=True)
    p.add_argument('--question',default='What will happen next in the video?')
    p.add_argument('--sigmas',default='0.01,0.03,0.1,0.2'); p.add_argument('--seeds',default='11,22,33')
    p.add_argument('--branch-count',type=int,default=5); p.add_argument('--feedback-rounds',type=int,default=2)
    p.add_argument('--nli-threshold',type=float,default=.70); p.add_argument('--min-quality',type=float,default=.75)
    p.add_argument('--min-target-coverage',type=float,default=.75)
    p.add_argument('--max-new-tokens',type=int,default=320); p.add_argument('--nframes',type=int,default=8)
    p.add_argument('--max-pixels',type=int,default=401408); p.add_argument('--scale-mode',choices=['rms','std','absolute'],default='rms')
    return p.parse_args()


def sanitize(raw: str):
    visible=LATENT_TOKEN_RE.sub('',raw); visible=' '.join(visible.split())
    match=ANSWER_RE.search(visible)
    return visible, match.group(1).strip() if match else ''


def build_inputs(processor: Any, video_path: str, question: str, nframes: int, max_pixels: int):
    messages=[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':[
        {'type':'video','video':video_path,'nframes':nframes,'max_pixels':max_pixels,'min_pixels':256*28*28},
        {'type':'text','text':question}]}]
    rendered=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    image_inputs,video_inputs,video_kwargs=process_vision_info(messages,return_video_kwargs=True,image_patch_size=16,return_video_metadata=True)
    metadata=None
    if video_inputs is not None: video_inputs,metadata=map(list,zip(*video_inputs))
    inputs=processor(text=[rendered],images=image_inputs,videos=video_inputs,video_metadata=metadata,**video_kwargs,do_resize=False,return_tensors='pt')
    inputs.pop('mm_token_type_ids',None)
    return inputs.to('cuda'),rendered,video_kwargs,metadata


def main():
    args=parse_args(); video=Path(args.video); out=Path(args.output_dir)
    if not video.is_file(): raise FileNotFoundError(video)
    if out.exists() and any(out.iterdir()): raise FileExistsError(f'refusing to overwrite non-empty output: {out}')
    out.mkdir(parents=True,exist_ok=True)
    aliases=[x.strip() for x in args.target_aliases.split('|') if x.strip()] or [args.target_object]
    sigmas=[float(x) for x in args.sigmas.split(',') if x.strip() and float(x)>0]
    seeds=[int(x) for x in args.seeds.split(',') if x.strip()]

    replace_qwen3_with_mixed_modality_forward(); replace_qwen3_vl_generation_forward()
    started=time.perf_counter()
    model=FutureL1_Qwen3VL.from_pretrained(args.model,dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa',trust_remote_code=True).eval()
    processor=AutoProcessor.from_pretrained(args.model,max_pixels=args.max_pixels,min_pixels=256*28*28,trust_remote_code=True)
    model.config.infer_head_prefer='render'; model.config.fixed_latent_budget=False; model.config.infer_latent_multiplier=2.0
    nli=TransformersNLI(args.nli_model,device='cuda',batch_size=64)
    load_seconds=time.perf_counter()-started

    metadata={'experiment':'round3-event-aware-search','created_unix':time.time(),'model':args.model,'nli_model':args.nli_model,
      'video':str(video),'video_sha256':sha256(video),'target_object':args.target_object,'target_aliases':aliases,
      'question':args.question,'sigmas':sigmas,'seeds':seeds,'branch_count_max':args.branch_count,
      'feedback_rounds':args.feedback_rounds,'feedback_zero_control':True,'nli_threshold':args.nli_threshold,
      'min_quality':args.min_quality,'min_target_coverage':args.min_target_coverage,'max_new_tokens':args.max_new_tokens,
      'nframes':args.nframes,'max_pixels':args.max_pixels,'scale_mode':args.scale_mode,'do_sample':False,
      'active_latent_generation':True,'shared_prefill':True,'branch_cache_batching':True,
      'torch':torch.__version__,'cuda_build':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'load_seconds':load_seconds,
      'code_sha256':{p.name:sha256(p) for p in [Path(__file__),HERE/'complete_scheme_v2.py',ROOT/'src/model/future_l1.py']}}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    all_candidates=[]; round_records=[]; prompts=[]

    def generate_round(prompt: str, round_index: int):
        inputs,rendered,video_kwargs,video_metadata=build_inputs(processor,str(video),prompt,args.nframes,args.max_pixels)
        prompts.append({'round':round_index,'prompt':prompt,'rendered_prompt':rendered,'video_kwargs':video_kwargs,'video_metadata':video_metadata})
        specs=([(0.0,0),(0.0,0)] if round_index==0 else [(0.0,0)])+[(s,z) for s in sigmas for z in seeds]
        produced=[]
        for index,(sigma,seed) in enumerate(specs):
            torch.manual_seed(123456); torch.cuda.manual_seed_all(123456); torch.cuda.reset_peak_memory_stats()
            run_inputs={k:(v.clone() if isinstance(v,torch.Tensor) else v) for k,v in inputs.items()}
            tick=time.perf_counter()
            pool = ActiveLatentBranchPool(
                torch.zeros(1, device="cuda"),
                branch_count=args.branch_count,
                propose=lambda state, count: [state.clone() for _ in range(count)],
            )
            with torch.inference_mode():
                output=model.generate(**run_inputs,max_new_tokens=args.max_new_tokens,do_sample=False,use_cache=True,
                    return_dict_in_generate=True,output_scores=False,output_hidden_states=False,
                    latent_noise_std=sigma,latent_noise_seed=seed,latent_noise_scale_mode=args.scale_mode,
                    latent_branch_count=args.branch_count, latent_branch_pool=pool,
                    eos_token_id=processor.tokenizer.eos_token_id,pad_token_id=processor.tokenizer.pad_token_id)
            torch.cuda.synchronize(); elapsed=time.perf_counter()-tick
            for branch_index in range(output.sequences.shape[0]):
                trimmed=output.sequences[branch_index,inputs.input_ids.shape[1]:]
                raw=processor.tokenizer.decode(trimmed,skip_special_tokens=False,clean_up_tokenization_spaces=False)
                visible,answer=sanitize(raw)
                states, masks = getattr(output, "latent_states", None), getattr(output, "latent_mask", None)
                latent = np.empty((0, 0), dtype=np.float32)
                if states is not None and masks is not None:
                    selected = states[branch_index][masks[branch_index].to(dtype=torch.bool)]
                    latent = selected.detach().float().cpu().numpy() if selected.numel() else np.empty((0, int(states.shape[-1])), dtype=np.float32)
                branch_id=f'r{round_index}_{index:02d}_b{branch_index:02d}_sigma_{sigma:g}_seed_{seed}'
                latent_file=out/f'{branch_id}_latents.npz'; raw_file=out/f'{branch_id}_raw.txt'
                np.savez_compressed(latent_file,latents=latent.astype(np.float16)); raw_file.write_text(raw,encoding='utf-8')
                valid=bool(answer and '<|latent_start|>' in raw and '<|latent_end|>' in raw and '<|im_end|>' in raw and latent.size and np.isfinite(latent).all())
                item=assess_candidate(BranchCandidate(branch_id,answer,sigma,seed,round_index,latent,raw,valid,
                  metadata={'elapsed_seconds':elapsed,'generated_tokens':int(trimmed.numel()),'peak_gpu_memory_gib':torch.cuda.max_memory_allocated()/(1024**3),
                   'latent_shape':list(latent.shape),'latent_sha256_float32':tensor_sha256(latent),'latent_sha256_saved':sha256(latent_file),
                   'raw_sha256':sha256(raw_file),'noise_stats':getattr(output,'latent_noise_stats',None)}),target_aliases=aliases)
                produced.append(item); all_candidates.append(item)
                (out/f'{branch_id}.json').write_text(json.dumps(serialise_candidate(item),indent=2,default=str),encoding='utf-8')
            print(f'RUN={branch_id}; VALID={valid}; COVERAGE={item.target_coverage:.3f}; SUBJECT={item.subject_targeted}; QUALITY={item.quality:.3f}; TIME={elapsed:.3f}',flush=True)
        return produced

    answer_subject=args.target_object.strip()
    if answer_subject.lower().startswith('the '): answer_subject=answer_subject[4:]
    initial_prompt=(f'{args.question.rstrip()} Focus only on this target object: {args.target_object}. '
      'Predict exactly one immediate, observable next motion or state-changing interaction of that target. '
      f'The answer must be one sentence beginning exactly with: "The {answer_subject} will". '
      'Avoid alternatives joined by or, and do not describe any other object as the target.')
    generate_round(initial_prompt,0)
    comparisons=compare_events(all_candidates,entailment=nli,threshold=args.nli_threshold)
    retained=select_unique_branches(all_candidates,comparisons=comparisons,branch_count=args.branch_count,
      min_quality=args.min_quality,min_target_coverage=args.min_target_coverage)
    round_records.append({'round':0,'generated':[c.branch_id for c in all_candidates],'retained':[c.branch_id for c in retained]})
    for round_index in range(1,args.feedback_rounds+1):
        prompt=build_feedback_prompt(question=args.question,target_object=args.target_object,retained=retained,round_index=round_index)
        before={c.branch_id for c in retained}; generated=generate_round(prompt,round_index)
        comparisons=compare_events(all_candidates,entailment=nli,threshold=args.nli_threshold)
        retained=select_unique_branches(all_candidates,comparisons=comparisons,branch_count=args.branch_count,
          min_quality=args.min_quality,min_target_coverage=args.min_target_coverage)
        round_records.append({'round':round_index,'generated':[c.branch_id for c in generated],
          'retained':[c.branch_id for c in retained],'new_retained':[c.branch_id for c in retained if c.branch_id not in before]})
        partial={'metadata':metadata,'rounds':round_records,'retained':[serialise_candidate(c) for c in retained]}
        (out/'search.partial.json').write_text(json.dumps(partial,indent=2,default=str),encoding='utf-8')

    baseline=[c for c in all_candidates if c.round_index==0 and c.sigma==0]
    baseline_representatives = [
        next((c for c in baseline if c.branch_id.startswith(f"r0_{run_index:02d}_b00_")), None)
        for run_index in (0, 1)
    ]
    report={'metadata':metadata,'prompts':prompts,'candidate_count':len(all_candidates),
      'valid_count':sum(c.valid_format for c in all_candidates),'accepted_structure_count':sum(c.valid_format and c.quality>=args.min_quality and c.target_coverage>=args.min_target_coverage and c.subject_targeted and c.motion_specific for c in all_candidates),
      'baseline_raw_equal':all(baseline_representatives) and baseline_representatives[0].raw==baseline_representatives[1].raw,
      'baseline_latent_equal':all(baseline_representatives) and np.array_equal(baseline_representatives[0].latent,baseline_representatives[1].latent),
      'rounds':round_records,'candidates':[serialise_candidate(c) for c in all_candidates],
      'comparisons':[vars(c) for c in comparisons],'retained':[serialise_candidate(c) for c in retained],
      'retained_count':len(retained),'trajectory_decoder':None,
      'interpretation':'Retained entries are event candidates, not trajectory outputs or calibrated probabilities.'}
    (out/'complete_search_report.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
    print('FINAL='+json.dumps({'candidate_count':len(all_candidates),'retained_count':len(retained),'retained':[c.answer for c in retained]},ensure_ascii=False),flush=True)


if __name__=='__main__': main()
