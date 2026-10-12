#!/usr/bin/env python3
"""Observed-only MotionForesight interface smoke for E3 preparation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def sample_map(values: torch.Tensor, uv: np.ndarray, height: int, width: int) -> torch.Tensor:
    """B,C,T,H,W -> T,N,C at model-pixel coordinates."""
    grid = torch.as_tensor(uv, dtype=torch.float32, device=values.device).clone()
    grid[:, 0] = 2.0 * grid[:, 0] / max(width - 1, 1) - 1.0
    grid[:, 1] = 2.0 * grid[:, 1] / max(height - 1, 1) - 1.0
    grid = grid.view(1, 1, -1, 2)
    per_time = values[0].permute(1, 0, 2, 3)
    return F.grid_sample(per_time, grid.expand(per_time.shape[0], -1, -1, -1),
                         mode="bilinear", align_corners=True)[:, :, 0].permute(0, 2, 1)


def choose_queries(depth: np.ndarray, count: int, height: int, width: int) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    resized = cv2.resize(valid.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
    yy, xx = np.where(resized)
    order = np.lexsort((xx, yy))
    picked = order[np.linspace(0, len(order) - 1, count, dtype=np.int64)]
    return np.stack([xx[picked], yy[picked]], axis=-1).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--user-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--array-output", type=Path, default=None)
    parser.add_argument(
        "--query-json",
        type=Path,
        default=None,
        help="Optional query JSON in original input-pixel coordinates.",
    )
    parser.add_argument("--query-count", type=int, default=48)
    parser.add_argument(
        "--attn-q-chunk",
        type=int,
        default=128,
        help="Query-axis SDPA chunk size for V100/math-attention fallback; <=0 keeps the original attention.",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo))
    from scripts.inference_model import build_model_from_ckpt
    from models_pretrained.future_scene_flow.sparse_dataset import (
        _apply_rigid, _compute_pj_norm, _unproject_world,
    )

    if args.attn_q_chunk > 0:
        from einops import rearrange

        def chunked_flash_attention(q, k, v, num_heads, compatibility_mode=False):
            """Exact attention factorization with bounded query-axis memory."""
            q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
            k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
            v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
            step = min(args.attn_q_chunk, q.shape[2])
            chunks = [
                F.scaled_dot_product_attention(q[:, :, start:start + step], k, v)
                for start in range(0, q.shape[2], step)
            ]
            return rearrange(torch.cat(chunks, dim=2), "b n s d -> b s (n d)", n=num_heads)

        # ``models_pretrained`` installs the vendored packages under a
        # top-level ``diffsynth`` name; patch both aliases after imports settle.
        import importlib
        for module_name in (
            "diffsynth.models.wan_video_dit",
            "models_pretrained.diffsynth.models.wan_video_dit",
        ):
            try:
                importlib.import_module(module_name).flash_attention = chunked_flash_attention
            except ModuleNotFoundError:
                pass

    model, state = build_model_from_ckpt(args.checkpoint, args.base_checkpoint)
    cfg = model.config
    data = np.load(args.user_npz, allow_pickle=True)
    rgb = np.asarray(data["rgb"][:cfg.obs_frames])
    depth = np.asarray(data["depth_map"][:cfg.obs_frames])
    w2c = np.asarray(data["extrinsics_w2c"][:cfg.obs_frames], dtype=np.float64)
    fx, fy, cx, cy = np.asarray(data["fx_fy_cx_cy"], dtype=np.float64)
    source_h, source_w = rgb.shape[1:3]

    rgb_obs = np.stack([cv2.resize(frame, (cfg.width, cfg.height)) for frame in rgb])
    rgb_obs = (rgb_obs.astype(np.float32) / 127.5 - 1.0).transpose(0, 3, 1, 2)
    intr = np.asarray([
        fx * cfg.width / source_w, fy * cfg.height / source_h,
        cx * cfg.width / source_w, cy * cfg.height / source_h,
    ])
    pj_world = np.stack([
        _unproject_world(cv2.resize(frame, (cfg.width, cfg.height)).astype(np.float64), intr, pose)
        for frame, pose in zip(depth, w2c)
    ])
    pj = _apply_rigid(w2c[-1], pj_world)
    norm = _compute_pj_norm(pj, 80.0, 2.0, 98.0)
    pj_n = ((pj - norm.mean) / norm.scale).astype(np.float32)
    p0 = np.transpose(pj_n[0], (2, 0, 1))
    if args.query_json is None:
        query_uv_original = choose_queries(depth[0], args.query_count, source_h, source_w)
    else:
        query_payload = json.loads(args.query_json.read_text(encoding="utf-8"))
        query_sets = query_payload["query_sets"]
        key = str(args.query_count)
        if key not in query_sets:
            raise ValueError(f"query JSON lacks query_sets[{key!r}]")
        query_uv_original = np.asarray(query_sets[key], dtype=np.float32)
        if query_uv_original.shape != (args.query_count, 2):
            raise ValueError(f"unexpected query shape: {query_uv_original.shape}")
    query_uv = query_uv_original * np.asarray(
        [cfg.width / source_w, cfg.height / source_h], dtype=np.float32
    )

    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        captured["last_block"] = output.detach().to("cpu")

    handle = model.dit.blocks[-1].register_forward_hook(hook)
    rgb_t = torch.from_numpy(rgb_obs).unsqueeze(0).to(model.device)
    pj_t = torch.from_numpy(np.transpose(pj_n, (0, 3, 1, 2))).unsqueeze(0).to(model.device)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        query_latent = model(rgb_t, pj_t)
        xyz_latent, _ = model.split_latent(query_latent)
        delta = model.decode_xyz(xyz_latent).float()
    handle.remove()
    prediction = model.denormalize(
        model.reconstruct(delta, torch.from_numpy(p0).unsqueeze(0).to(model.device).float()),
        torch.from_numpy(norm.mean).unsqueeze(0).to(model.device).float(),
        torch.tensor([norm.scale], device=model.device),
    )

    p0_query = sample_map(prediction, query_uv, cfg.height, cfg.width).cpu().numpy()
    output_h = sample_map(xyz_latent.float(), query_uv, cfg.height, cfg.width).cpu().numpy()
    hidden = captured["last_block"]
    token_count = hidden.shape[1] // 2
    time_count = cfg.total_frames
    spatial_tokens = token_count // time_count
    token_h = cfg.height // 16
    token_w = spatial_tokens // token_h
    row_hidden = hidden[:, token_count:].reshape(1, time_count, token_h, token_w, -1)
    row_hidden = row_hidden.permute(0, 4, 1, 2, 3).float()
    dit_h = sample_map(row_hidden, query_uv, cfg.height, cfg.width).cpu().numpy()

    future = slice(cfg.obs_frames, cfg.total_frames)
    perm = np.arange(args.query_count)[::-1]
    p0_permuted = sample_map(prediction, query_uv[perm], cfg.height, cfg.width).cpu().numpy()
    permutation_error = float(np.max(np.abs(p0_permuted[:, np.argsort(perm)] - p0_query)))
    observed_hash = hashlib.sha256(rgb_t.cpu().numpy().tobytes() + pj_t.cpu().numpy().tobytes()).hexdigest()
    payload = {
        "status": "PASS",
        "scope": "official-example observed-only interface smoke; not an E3 result",
        "checkpoint_config": state["config"],
        "input": {
            "observed_rgb_shape": list(rgb_t.shape),
            "observed_pointmap_shape": list(pj_t.shape),
            "observed_tensor_sha256": observed_hash,
            "future_frame_tensor_supplied": False,
        },
        "outputs": {
            "dense_prediction_shape": list(prediction.shape),
            "query_latent_shape": list(query_latent.shape),
            "p0_future_query_shape": list(p0_query[future].shape),
            "h_output_latent_future_shape": list(output_h[future].shape),
            "h_last_dit_future_shape": list(dit_h[future].shape),
            "finite": bool(np.isfinite(p0_query).all() and np.isfinite(output_h).all() and np.isfinite(dit_h).all()),
        },
        "query_sampling": {
            "count": args.query_count,
            "anchor": "first_observed_frame",
            "permutation_sampling_max_abs_error": permutation_error,
        },
        "coordinate_frame": "last_observed_camera",
        "temporal_boundary": {
            "model_observed_frames": cfg.obs_frames,
            "model_future_frames": cfg.future_frames,
            "formal_e3_future_mapping_verified": False,
        },
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "attention": {
            "implementation": "query_axis_chunked_sdpa" if args.attn_q_chunk > 0 else "model_default",
            "query_chunk": int(args.attn_q_chunk),
        },
    }
    if args.array_output is not None:
        args.array_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.array_output,
            p0_query=p0_query.astype(np.float32),
            h_output_latent=output_h.astype(np.float32),
            h_last_dit=dit_h.astype(np.float32),
            query_uv_model=query_uv.astype(np.float32),
            query_uv_original=query_uv_original.astype(np.float32),
            observed_frame_count=np.int32(cfg.obs_frames),
            total_frame_count=np.int32(cfg.total_frames),
            coordinate_frame=np.asarray("last_observed_camera"),
            future_frames_supplied_to_model=np.asarray(False),
        )
        payload["array_output"] = str(args.array_output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
