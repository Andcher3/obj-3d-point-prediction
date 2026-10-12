#!/usr/bin/env python3
"""Official two-run 12-frame interleaving for a 23-frame offline teacher."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def reconstruct_pj0(depth0, intrinsics, query_model, model_h, model_w, orig_h, orig_w):
    depth_model = np.asarray(
        Image.fromarray(np.asarray(depth0, np.float32), mode="F").resize(
            (model_w, model_h), Image.Resampling.NEAREST
        ), dtype=np.float64,
    )
    scale = np.asarray([model_w / orig_w, model_h / orig_h], dtype=np.float64)
    k_model = np.asarray(intrinsics, dtype=np.float64) * np.asarray(
        [scale[0], scale[1], scale[0], scale[1]], dtype=np.float64
    )
    uv = np.asarray(query_model, dtype=np.float64).astype(np.int64)
    z = depth_model[uv[:, 1], uv[:, 0]]
    fx, fy, cx, cy = k_model
    return np.stack(((uv[:, 0] - cx) / fx * z, (uv[:, 1] - cy) / fy * z, z), axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--query-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=416)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo))
    import trackcraft_release_adapter as adapter
    import evaluation.dust3r_eval_utils as dust3r_utils
    import evaluation.wan_scene_flow_predictor as predictor_module
    adapter.install_release_loader()
    predictor_module.WanSceneFlowPredictor = adapter.ReleaseCheckpointPredictor

    source = np.load(args.input_npz, allow_pickle=True)
    frame_indices = np.asarray(source["frame_indices"], dtype=np.int32)
    if len(source["images_jpeg_bytes"]) != 23 or frame_indices.shape != (23,):
        raise ValueError("official 22-frame evaluation requires exactly 23 source frames")
    if not np.array_equal(np.diff(frame_indices), np.ones(22, dtype=np.int32)):
        raise ValueError(f"source frames must be contiguous, got {frame_indices.tolist()}")
    data = {key: source[key] for key in source.files}
    data["tracks_XYZ"] = np.zeros((23, 1, 3), dtype=np.float32)
    data["visibility"] = np.ones((23, 1), dtype=bool)
    fd, temporary = tempfile.mkstemp(suffix=".npz"); os.close(fd)
    np.savez_compressed(temporary, **data)
    try:
        video, _, _, intrinsics, _, _, _, w2c = dust3r_utils.load_npz_data(temporary, num_frames=23)
    finally:
        os.unlink(temporary)
    depth = np.asarray(source["depth_map"], dtype=np.float32)
    orig_h, orig_w = depth.shape[1:]
    query_payload = json.loads(args.query_json.read_text(encoding="utf-8"))
    query = np.asarray(query_payload["query_sets"]["48"], dtype=np.float32)
    torch.cuda.reset_peak_memory_stats()
    predictor = predictor_module.WanSceneFlowPredictor(
        checkpoint_path=str(args.checkpoint), height=args.height, width=args.width,
        device="cuda", track_latent_length=12, resize_mode="stretch",
        predict_vis=True, parallel_vae_decode=False,
    )
    run_indices = np.asarray([list(range(0, 23, 2)), [0] + list(range(1, 23, 2))], dtype=np.int32)
    if run_indices.shape != (2, 12):
        raise AssertionError(run_indices.shape)
    raw_sum = np.zeros((23, 48, 3), dtype=np.float64)
    anchored_sum = np.zeros_like(raw_sum)
    count = np.zeros((23, 48), dtype=np.float64)
    delta_runs, pj0_runs, query_model_runs = [], [], []
    raw_runs, anchored_runs, pj0_sources = [], [], []
    for indices in run_indices:
        with torch.no_grad():
            raw = np.asarray(predictor.predict(
                [video[i] for i in indices], query, np.ones((12, 48), dtype=bool), intrinsics,
                depth_map=depth[indices], extrinsics_w2c=w2c[indices],
            ), dtype=np.float64)
        q_model = np.asarray(predictor._last_query_uv_model, dtype=np.float64)
        oob = np.asarray(predictor._last_oob_mask, dtype=bool)
        if raw.shape != (12, 48, 3) or not np.all(oob):
            raise ValueError(f"unexpected interleaved output {raw.shape}, oob={int((~oob).sum())}")
        uv_int = q_model.astype(np.int64)
        cached_pj = getattr(predictor, "_last_pj_input", None)
        if cached_pj is not None and np.asarray(cached_pj).shape[:3] == (12, args.height, args.width):
            pj0 = np.asarray(cached_pj[0, uv_int[:, 1], uv_int[:, 0]], dtype=np.float64).copy()
            pj0_source = "predictor._last_pj_input"
        else:
            pj0 = reconstruct_pj0(
                depth[indices[0]], intrinsics, q_model, args.height, args.width, orig_h, orig_w
            )
            pj0_source = "synchronized_nearest_depth_Kmodel_integer_query_fallback"
        delta = pj0 - raw[0]
        anchored = raw + delta[None]
        for local, global_index in enumerate(indices):
            raw_sum[global_index] += raw[local]
            anchored_sum[global_index] += anchored[local]
            count[global_index] += 1
        delta_runs.append(delta); pj0_runs.append(pj0); query_model_runs.append(q_model.copy())
        raw_runs.append(raw.copy()); anchored_runs.append(anchored.copy())
        pj0_sources.append(pj0_source)
    if not np.all(count > 0):
        raise ValueError("interleaved merge left an uncovered frame/query")
    raw_final = raw_sum / count[..., None]
    anchored_final = anchored_sum / count[..., None]
    raw_eval, anchored_eval = raw_final[:22], anchored_final[:22]
    delta_runs = np.asarray(delta_runs); pj0_runs = np.asarray(pj0_runs)
    raw_runs = np.asarray(raw_runs); anchored_runs = np.asarray(anchored_runs)
    per_run_displacement_change = np.max(np.abs(
        (anchored_runs - anchored_runs[:, 0:1]) - (raw_runs - raw_runs[:, 0:1])
    ), axis=(1, 2, 3))
    displacement_change = float(np.max(np.abs(
        (anchored_eval - anchored_eval[0:1]) - (raw_eval - raw_eval[0:1])
    )))
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz, trackcraft3r_tracks=anchored_eval.astype(np.float32),
        trackcraft3r_tracks_raw_interleaved=raw_eval.astype(np.float32),
        query_uv_original=query, query_uv_model=np.asarray(query_model_runs[0], dtype=np.float32),
        oob_mask=np.ones(48, dtype=bool), model_resolution_hw=np.asarray([args.height, args.width], dtype=np.int32),
        input_frame_count=np.int32(23), output_frame_count=np.int32(22),
        source_frame_indices_input=frame_indices,
        source_frame_indices_output=frame_indices[:22],
        interleaved_run_indices=run_indices, frame0_pj_anchor_xyz_runs=pj0_runs.astype(np.float32),
        frame0_anchor_delta_xyz_runs=delta_runs.astype(np.float32),
        trackcraft3r_tracks_raw_runs=raw_runs.astype(np.float32),
        trackcraft3r_tracks_anchored_runs=anchored_runs.astype(np.float32),
        per_run_temporal_displacement_max_abs_change=per_run_displacement_change.astype(np.float64),
        frame0_anchor_method=np.asarray("per-run same-run integer-query Pj0 constant xyz translation before interleaved merge"),
        coordinate_frame=np.asarray("frame0_camera_common_coordinates"),
        official_long_video_protocol=np.asarray(True),
        extra_context_source_frame=np.int32(frame_indices[-1]),
    )
    summary = {
        "status": "OFFICIAL_INTERLEAVED_DERIVED_NOT_QUALITY_APPROVED",
        "input_frames": 23, "output_frames": 22, "run_indices": run_indices.tolist(),
        "output_shape": list(anchored_eval.shape),
        "finite_fraction": float(np.isfinite(anchored_eval).mean()),
        "gpu_device": torch.cuda.get_device_name(),
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "source_frame_indices_input": frame_indices.tolist(),
        "source_frame_indices_output": frame_indices[:22].tolist(),
        "pj0_sources": pj0_sources,
        "frame0_pj_between_runs_max_abs": float(np.max(np.abs(pj0_runs[0] - pj0_runs[1]))),
        "anchored_frame0_between_runs_max_abs": float(np.max(np.abs(pj0_runs[0] - pj0_runs[1]))),
        "temporal_displacement_max_abs_change_after_merged_anchor": displacement_change,
        "per_run_temporal_displacement_max_abs_change": per_run_displacement_change.tolist(),
        "anchor_delta_norm_per_run": [
            {"p50": float(np.quantile(np.linalg.norm(d, axis=-1), .5)),
             "p90": float(np.quantile(np.linalg.norm(d, axis=-1), .9)),
             "max": float(np.max(np.linalg.norm(d, axis=-1)))} for d in delta_runs
        ],
    }
    (args.output_npz.parent / "official_interleaved_derivation.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
