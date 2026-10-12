#!/usr/bin/env python3
"""Probe MolmoMotion on a raw RGB clip with controlled placement instructions.

This is an interface test, not a metric benchmark: the clip has no calibrated
camera/depth track.  We supply a stable approximate metric point history and
compare prompt-conditioned 3D outputs, decoding validity, and seed variation.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

PROMPTS = {
    "mouse": "Move the drink bottle to the right and place it next to the computer mouse.",
    "black_cup": "Move the drink bottle to the right and place it next to the black water cup.",
    "laptop": "Move the drink bottle to the right and place it next to the laptop computer.",
    "stay": "Keep holding the drink bottle and do not put it down.",
}

def sha(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--future-horizon", type=int, default=30)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260920)
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(args.repo) / "src"))
    from molmo_motion import MolmoMotion, MolmoMotionProcessor
    from molmo_motion.nn.beam_search import TopKTopPSampler
    import cv2
    cap = cv2.VideoCapture(args.video)
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok: break
        frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    cap.release()
    if len(frames) < 5: raise RuntimeError("video needs at least 5 frames")
    # Eight points over the bottle in frame 4 (1280x720).  The raw clip is
    # intentionally short; this point set is only for conditioning, not GT.
    uv = np.array([[445,365],[505,355],[555,390],[435,455],[510,455],[565,470],[455,565],[530,585]], np.float32)
    fx = fy = 1100.0; cx, cy = frames[0].width/2, frames[0].height/2
    z = 1.0
    xyz = np.stack([(uv[:,0]-cx)*z/fx, (uv[:,1]-cy)*z/fy, np.full(len(uv),z)], -1)
    # H=3 history; same approximate points in camera-frame-at-t0.
    history_frames = [frames[i] for i in (0,2,4)]
    points_2d = torch.from_numpy(uv)
    points_3d = torch.from_numpy(np.stack([xyz, xyz, xyz], 0)).float()
    processor = MolmoMotionProcessor.from_pretrained(args.model)
    model = MolmoMotion.from_pretrained(args.model)
    model._internal = model._internal.to(torch.bfloat16).cuda()
    if processor.config.history_size != 3:
        raise RuntimeError(f"checkpoint history={processor.config.history_size}, expected 3")
    rows=[]; predictions={}
    for pname, prompt in PROMPTS.items():
        inp_cpu = processor(history_frames=history_frames, points_2d_at_t0=points_2d,
                            points_3d_history=points_3d, action=prompt,
                            future_horizon=args.future_horizon)
        inp = {k:(v.cuda() if torch.is_tensor(v) else v) for k,v in inp_cpu.items()}
        draws=[("greedy", None)] + [(f"sample{i+1}", i) for i in range(args.samples)]
        for dname, di in draws:
            sampler=None
            if di is not None:
                torch.manual_seed(args.seed + di); torch.cuda.manual_seed_all(args.seed + di)
                sampler=TopKTopPSampler(p=0.95,k=50,temperature=0.8)
            t=time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                if sampler is None:
                    res=model.predict_trajectory(**inp)
                    future=res.future_3d.detach().float().cpu(); text=res.future_text
                else:
                    batch={k:v for k,v in inp.items() if k not in ("target_tokens","loss_masks","future_horizon","history_size","anchor_3d")}
                    gen=model._internal.generate(batch=batch,max_steps=160*args.future_horizon,sampler=sampler,is_distributed=False)
                    tok=gen.token_ids[:,0].detach().cpu().numpy()[0]
                    text=model._internal.config.llm.build_tokenizer().decode(tok[tok>=0])
                    from molmo_motion.eval.egodex_3d_evaluator import parse_tracks_text, tracks_to_array
                    parsed=parse_tracks_text(text)
                    if parsed is None: raise RuntimeError(f"parse failed for {pname}/{dname}")
                    delta,_=tracks_to_array(parsed,num_points=8,num_frames=args.future_horizon,start_timestamp=3.0)
                    future=torch.from_numpy(np.asarray(delta,np.float32))+points_3d[-1].unsqueeze(1)
            pred_key=f"{pname}__{dname}"; predictions[pred_key]=future
            centroid=future.mean(0); step=torch.linalg.vector_norm(centroid[1:]-centroid[:-1],dim=-1)
            row={"prompt":pname,"prompt_text":prompt,"draw":dname,"seed":None if di is None else args.seed+di,
                 "shape":list(future.shape),"seconds":time.perf_counter()-t,"future_sha256":sha(future),
                 "future_text_sha256":hashlib.sha256(text.encode()).hexdigest(),
                 "path_length_m":float(step.sum()),"final_delta_m":(centroid[-1]-points_3d[-1].mean(0)).tolist(),
                 "finite_fraction":float(torch.isfinite(future).float().mean())}
            rows.append(row)
            np.savez_compressed(out/(pred_key+".npz"),future_3d=future.numpy(),points_2d=uv,points_3d_history=points_3d.numpy())
            (out/(pred_key+".txt")).write_text(text,encoding="utf-8")
            print("RESULT "+json.dumps(row,ensure_ascii=False),flush=True)
    pairwise={}
    keys=list(predictions)
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            pairwise[f"{a}__vs__{b}"]=float(torch.linalg.vector_norm(predictions[a]-predictions[b],dim=-1).mean())
    payload={"protocol":{"video":str(Path(args.video).resolve()),"history_frame_indices":[0,2,4],"approximate_3d_input":True,"points":uv.tolist(),"future_horizon":args.future_horizon,"samples_per_prompt":args.samples},"records":rows,"pairwise_mean_point_distance_m":pairwise}
    (out/"metrics.json").write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8")
    print(f"METRICS={out/'metrics.json'}")
if __name__ == "__main__": main()
