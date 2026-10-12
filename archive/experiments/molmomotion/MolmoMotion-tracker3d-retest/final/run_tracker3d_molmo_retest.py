#!/usr/bin/env python3
"""Aligned multi-video MolmoMotion text-control retest with offline TrackCraft pseudo-GT."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--spec-json", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--future-horizon", type=int, default=30)
    p.add_argument("--history-size", type=int, default=3)
    p.add_argument("--stochastic-samples", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--sample-seed", type=int, default=20260915)
    return p.parse_args()


def transform_points(matrix: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([xyz, np.ones((*xyz.shape[:-1], 1), np.float32)], axis=-1)
    return np.einsum("ij,...j->...i", matrix.astype(np.float32), homogeneous)[..., :3]


def project(xyz: np.ndarray, k: np.ndarray) -> np.ndarray:
    z = np.clip(xyz[..., 2], 1e-6, None)
    u = k[0, 0] * xyz[..., 0] / z + k[0, 2]
    v = k[1, 1] * xyz[..., 1] / z + k[1, 2]
    return np.stack([u, v], axis=-1).astype(np.float32)


def sha_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    delta = torch.linalg.vector_norm(a.float() - b.float(), dim=-1)
    va = a[:, -1].mean(0) - a[:, 0].mean(0)
    vb = b[:, -1].mean(0) - b[:, 0].mean(0)
    denom = torch.linalg.vector_norm(va) * torch.linalg.vector_norm(vb)
    cosine = float(torch.dot(va, vb) / denom) if float(denom) > 1e-12 else math.nan
    return {"ade_m": float(delta.mean()), "fde_m": float(delta[:, -1].mean()), "direction_cosine": cosine}


def motion_metrics(future: torch.Tensor, anchor: torch.Tensor) -> dict:
    future = future.detach().float().cpu()
    anchor = anchor.detach().float().cpu()
    full = torch.cat([anchor[:, None], future], 1)
    step = torch.linalg.vector_norm(full[:, 1:] - full[:, :-1], dim=-1)
    delta = future[:, -1].mean(0) - anchor.mean(0)
    return {
        "mean_path_length_m": float(step.sum(1).mean()),
        "centroid_final_delta_m": delta.tolist(),
        "finite_fraction": float(torch.isfinite(future).float().mean()),
    }


def save_overlay(path: Path, image: Image.Image, k: np.ndarray, anchor_uv: np.ndarray,
                 pseudo_gt: torch.Tensor, predictions: dict[str, torch.Tensor]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    ax.imshow(image)
    gt_px = project(pseudo_gt.numpy(), k).mean(0)
    gt_full = np.concatenate([anchor_uv.mean(0, keepdims=True), gt_px], 0)
    ax.plot(gt_full[:, 0], gt_full[:, 1], "w--", linewidth=2.5, label="offline TrackCraft pseudo-GT")
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(predictions))))
    for color, (name, pred) in zip(colors, predictions.items()):
        px = project(pred.numpy(), k).mean(0)
        full = np.concatenate([anchor_uv.mean(0, keepdims=True), px], 0)
        ax.plot(full[:, 0], full[:, 1], color=color, linewidth=1.8, label=name)
    ax.scatter(anchor_uv[:, 0], anchor_uv[:, 1], s=20, c="yellow", edgecolors="black")
    ax.set_xlim(0, image.width)
    ax.set_ylim(image.height, 0)
    ax.axis("off")
    ax.legend(loc="best", fontsize=6)
    fig.tight_layout(pad=0)
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def decode_generation(model, inputs: dict, future_horizon: int, sampler=None):
    from molmo_motion.eval.egodex_3d_evaluator import parse_tracks_text, tracks_to_array

    batch = {"input_ids": inputs["input_ids"], "attention_mask": inputs.get("attention_mask")}
    for key in ("images", "image_masks"):
        if inputs.get(key) is not None:
            batch[key] = inputs[key]
    for key, value in inputs.items():
        if key in batch or key in ("target_tokens", "loss_masks", "future_horizon", "history_size", "anchor_3d"):
            continue
        batch[key] = value
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model._internal.generate(
            batch=batch,
            max_steps=160 * future_horizon,
            sampler=sampler,
            is_distributed=False,
        )
    token_ids = out.token_ids[:, 0].detach().cpu().numpy()[0]
    tokenizer = model._internal.config.llm.build_tokenizer()
    text = tokenizer.decode(token_ids[token_ids >= 0])
    parsed = parse_tracks_text(text)
    parse_success = parsed is not None
    if parsed is None:
        future = torch.zeros((model.config.num_points, future_horizon, 3), dtype=torch.float32)
    else:
        history_size = inputs.get("history_size", model.config.history_size)
        delta, _ = tracks_to_array(parsed, num_points=model.config.num_points,
                                   num_frames=future_horizon, start_timestamp=float(history_size))
        future = torch.from_numpy(np.asarray(delta, dtype=np.float32))
        anchor = inputs.get("anchor_3d")
        if anchor is not None:
            future = future + anchor.detach().cpu().squeeze(0)
    return future, text, parse_success, float(out.scores[0, 0].detach().float().cpu())


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo / "src"))
    from molmo_motion import MolmoMotion, MolmoMotionProcessor
    from molmo_motion.nn.beam_search import TopKTopPSampler

    specs = json.loads(Path(args.spec_json).read_text(encoding="utf-8"))["samples"]
    processor = MolmoMotionProcessor.from_pretrained(args.model)
    model = MolmoMotion.from_pretrained(args.model)
    model._internal = model._internal.to(torch.bfloat16).cuda()
    if processor.config.history_size != args.history_size:
        raise ValueError(f"checkpoint history={processor.config.history_size}, requested={args.history_size}")

    all_records = []
    sample_summaries = []
    for sample_i, spec in enumerate(specs):
        sample_id = spec["id"]
        sample_out = output / sample_id
        (sample_out / "arrays").mkdir(parents=True, exist_ok=True)
        (sample_out / "future_text").mkdir(exist_ok=True)
        camera = np.load(spec["camera_npz"], allow_pickle=True)
        dense = np.load(spec["dense_npz"])
        track_map = np.asarray(dense["track_map"], np.float32)
        rgb = np.asarray(dense["rgb"], np.uint8)
        extrinsics = np.asarray(camera["extrinsics_w2c"], np.float32)
        raw_frame_indices = np.asarray(camera["frame_indices"], np.int32)
        fx, fy, cx, cy = np.asarray(camera["fx_fy_cx_cy"], np.float32)
        k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)
        query_uv0 = np.asarray(json.loads(Path(spec["query_json"]).read_text())["frame0_query_pixels"], np.int64)
        t0 = args.history_size - 1
        if track_map.shape[0] <= t0:
            raise ValueError(f"{sample_id}: tracker output shorter than history")
        u0, v0 = query_uv0[:, 0], query_uv0[:, 1]
        xyz_frame0 = np.stack([track_map[t, v0, u0] for t in range(track_map.shape[0])], 0)
        xyz_t0_all = transform_points(extrinsics[t0], xyz_frame0)
        uv_t0 = project(xyz_t0_all[t0], k)
        keep = (np.isfinite(xyz_t0_all).all(axis=(0, 2)) & np.isfinite(uv_t0).all(1)
                & (xyz_t0_all[:args.history_size, :, 2] > 1e-4).all(0)
                & (uv_t0[:, 0] >= 0) & (uv_t0[:, 0] < rgb.shape[2])
                & (uv_t0[:, 1] >= 0) & (uv_t0[:, 1] < rgb.shape[1]))
        if keep.sum() < 4:
            raise RuntimeError(f"{sample_id}: only {int(keep.sum())} valid tracks")
        query_uv0, uv_t0, xyz_t0_all = query_uv0[keep], uv_t0[keep], xyz_t0_all[:, keep]
        history_xyz = xyz_t0_all[:args.history_size]
        pseudo_gt = torch.from_numpy(xyz_t0_all[args.history_size:]).float().permute(1, 0, 2)
        history_frames = [Image.fromarray(rgb[t]) for t in range(args.history_size)]
        points_2d = torch.from_numpy(uv_t0).float()
        points_3d = torch.from_numpy(history_xyz).float()
        pseudo_metrics = motion_metrics(pseudo_gt, points_3d[-1])
        predictions = {}
        sample_records = []
        for prompt_i, prompt in enumerate(spec["prompts"]):
            inputs_cpu = processor(history_frames=history_frames, points_2d_at_t0=points_2d,
                                   points_3d_history=points_3d, action=prompt["text"],
                                   future_horizon=args.future_horizon)
            input_sha = sha_tensor(inputs_cpu["input_ids"])
            inputs = {k0: (v.cuda() if torch.is_tensor(v) else v) for k0, v in inputs_cpu.items()}
            modes = [("greedy", None, None)]
            for draw in range(args.stochastic_samples):
                seed = args.sample_seed + sample_i * 1000 + prompt_i * 100 + draw
                sampler = TopKTopPSampler(p=args.top_p, k=args.top_k, temperature=args.temperature)
                modes.append((f"sample{draw + 1}", sampler, seed))
            for draw_name, sampler, seed in modes:
                if seed is not None:
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                future, future_text, parse_success, sequence_score = decode_generation(
                    model, inputs, args.future_horizon, sampler=sampler)
                elapsed = time.perf_counter() - started
                key = f"{prompt['name']}__{draw_name}"
                predictions[key] = future
                eval_h = min(pseudo_gt.shape[1], future.shape[1])
                gt_cmp = compare(future[:, :eval_h], pseudo_gt[:, :eval_h])
                row = {
                    "sample_id": sample_id,
                    "prompt": prompt["name"],
                    "prompt_text": prompt["text"],
                    "draw": draw_name,
                    "seed": seed,
                    "parse_success": parse_success,
                    "seconds": elapsed,
                    "sequence_score": sequence_score,
                    "input_ids_sha256": input_sha,
                    "future_array_sha256": sha_tensor(future),
                    "future_text_sha256": hashlib.sha256(future_text.encode()).hexdigest(),
                    "eval_horizon_frames": eval_h,
                    "pseudo_gt_ade_m": gt_cmp["ade_m"],
                    "pseudo_gt_fde_m": gt_cmp["fde_m"],
                    **motion_metrics(future, points_3d[-1]),
                }
                sample_records.append(row)
                all_records.append(row)
                np.savez_compressed(sample_out / "arrays" / f"{key}.npz", future_3d=future.numpy(),
                                    pseudo_gt=pseudo_gt.numpy(), points_3d_history=history_xyz,
                                    points_2d_at_t0=uv_t0, intrinsics=k, raw_frame_indices=raw_frame_indices,
                                    action=np.asarray(prompt["text"]))
                (sample_out / "future_text" / f"{key}.txt").write_text(future_text, encoding="utf-8")
                print("RESULT " + json.dumps(row, ensure_ascii=False), flush=True)
        pairwise = {}
        keys = list(predictions)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                pairwise[f"{a}__vs__{b}"] = compare(predictions[a], predictions[b])
        save_overlay(sample_out / "prediction_vs_pseudogt.png", history_frames[-1], k, uv_t0,
                     pseudo_gt, {k0: v for k0, v in predictions.items() if k0.endswith("__greedy")})
        summary = {
            "sample_id": sample_id,
            "source_video": spec["source_video"],
            "source_fps": spec["source_fps"],
            "raw_frame_indices": raw_frame_indices.tolist(),
            "history_raw_frames": raw_frame_indices[:args.history_size].tolist(),
            "offline_pseudo_gt_raw_frames": raw_frame_indices[args.history_size:].tolist(),
            "valid_query_count": int(keep.sum()),
            "pseudo_gt_motion": pseudo_metrics,
            "records": sample_records,
            "pairwise": pairwise,
        }
        (sample_out / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        sample_summaries.append(summary)
    aggregate = {"protocol": {
        "history_size": args.history_size,
        "future_horizon": args.future_horizon,
        "source_sampling_fps": 15,
        "offline_pseudo_gt_note": "TrackCraft sees future frames only for offline evaluation; MolmoMotion receives frames 0..2 only.",
        "stochastic_samples_per_prompt": args.stochastic_samples,
        "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
    }, "samples": sample_summaries, "records": all_records}
    (output / "aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"AGGREGATE={output / 'aggregate_metrics.json'}")


if __name__ == "__main__":
    main()
