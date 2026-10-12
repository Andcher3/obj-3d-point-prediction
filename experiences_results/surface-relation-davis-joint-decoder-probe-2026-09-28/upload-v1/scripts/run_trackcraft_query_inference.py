#!/usr/bin/env python3
"""Run TrackCraft3R on the formal 12-frame input and sample real queries only."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--query-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=416)
    parser.add_argument("--num-frames", type=int, default=12)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--query-count", type=int, default=48)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo))
    import trackcraft_release_adapter as adapter
    import evaluation.dust3r_eval_utils as dust3r_utils
    import evaluation.wan_scene_flow_predictor as predictor_module

    adapter.install_release_loader()
    predictor_module.WanSceneFlowPredictor = adapter.ReleaseCheckpointPredictor

    source = np.load(args.input_npz, allow_pickle=True)
    total = int(source["images_jpeg_bytes"].shape[0])
    span = args.num_frames * args.frame_stride
    if span > total:
        raise ValueError(f"input has {total} frames, need {span}")

    # load_npz_data expects track fields; add a temporary one-track placeholder.
    temp_path = None
    if "tracks_XYZ" not in source.files:
        data = {key: source[key] for key in source.files}
        data["tracks_XYZ"] = np.zeros((total, 1, 3), dtype=np.float32)
        data["visibility"] = np.ones((total, 1), dtype=bool)
        fd, temp_path = tempfile.mkstemp(suffix=".npz")
        os.close(fd)
        np.savez_compressed(temp_path, **data)
        load_path = temp_path
    else:
        load_path = str(args.input_npz)
    try:
        video_list, _, _, intrinsics, _, _, _, extrinsics_w2c = dust3r_utils.load_npz_data(
            load_path, num_frames=args.num_frames, frame_stride=args.frame_stride
        )
    finally:
        if temp_path:
            os.unlink(temp_path)

    depth_map = source["depth_map"][:span:args.frame_stride]
    query_payload = json.loads(args.query_json.read_text(encoding="utf-8"))
    key = str(args.query_count)
    if key not in query_payload["query_sets"]:
        raise ValueError(f"query JSON lacks query_sets[{key!r}]")
    query_uv = np.asarray(query_payload["query_sets"][key], dtype=np.float32)
    if query_uv.shape != (args.query_count, 2):
        raise ValueError(f"unexpected query shape {query_uv.shape}")
    visibility = np.ones((args.num_frames, len(query_uv)), dtype=bool)

    predictor = predictor_module.WanSceneFlowPredictor(
        checkpoint_path=str(args.checkpoint),
        height=args.height,
        width=args.width,
        device="cuda",
        track_latent_length=args.num_frames,
        resize_mode="stretch",
        predict_vis=True,
        parallel_vae_decode=False,
    )
    with torch.no_grad():
        tracks = predictor.predict(
            video_list, query_uv, visibility, intrinsics,
            depth_map=depth_map, extrinsics_w2c=extrinsics_w2c,
        )
    output = {
        "trackcraft3r_tracks": np.asarray(tracks, dtype=np.float32),
        "query_uv_original": query_uv,
        "query_uv_model": np.asarray(getattr(predictor, "_last_query_uv_model", query_uv), dtype=np.float32),
        "oob_mask": np.asarray(getattr(predictor, "_last_oob_mask", np.ones(len(query_uv), dtype=bool))),
        "model_resolution_hw": np.asarray([args.height, args.width], dtype=np.int32),
        "input_frame_count": np.asarray(args.num_frames, dtype=np.int32),
    }
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **output)
    print(json.dumps({
        "output": str(args.output_npz),
        "track_shape": list(output["trackcraft3r_tracks"].shape),
        "finite_fraction": float(np.isfinite(output["trackcraft3r_tracks"]).mean()),
        "nonzero_motion_fraction": float((np.linalg.norm(np.diff(output["trackcraft3r_tracks"], axis=0), axis=-1) > 1e-6).mean()),
        "oob_count": int((~output["oob_mask"]).sum()),
        "model_resolution_hw": [args.height, args.width],
        "trajectory_source": "trackcraft3r_pseudo_gt",
    }, indent=2))


if __name__ == "__main__":
    main()
