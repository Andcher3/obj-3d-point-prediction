#!/usr/bin/env python3
"""Project directly sampled frame-0 DAVIS query tracks to source RGB pixels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_davis_visualization_projection_bundle import project


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    d = args.video_dir
    m = json.loads((d / "query_frame0_manifest.json").read_text(encoding="utf-8"))
    q = np.asarray(m["query_sets"]["48"], dtype=np.float32)
    with (np.load(d / "observed7_query_tracks.npz") as o,
          np.load(d / "full22_query_tracks.npz") as t,
          np.load(d / "observed7_user.npz", allow_pickle=True) as ou,
          np.load(d / "full22_user.npz", allow_pickle=True) as tu):
        if not np.allclose(o["query_uv_original"], q) or not np.allclose(t["query_uv_original"], q):
            raise ValueError("query order mismatch")
        op = np.asarray(o["trackcraft3r_tracks"], dtype=np.float64)
        tp = np.asarray(t["trackcraft3r_tracks"], dtype=np.float64)
        if op.shape != (7, 48, 3) or tp.shape != (22, 48, 3):
            raise ValueError(f"unexpected shapes {op.shape} {tp.shape}")
        ouv, ov = project(op, np.asarray(ou["extrinsics_w2c"]), np.asarray(ou["fx_fy_cx_cy"]))
        tuv, tv = project(tp, np.asarray(tu["extrinsics_w2c"]), np.asarray(tu["fx_fy_cx_cy"]))
        processed = np.asarray(tu["depth_map"].shape[1:3], dtype=np.int32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output,
        video_id=np.asarray(m["video_id"]),
        source_frame_indices=np.asarray(m["source_frame_indices"], dtype=np.int32),
        observed_source_frame_indices=np.asarray(m["observed_source_frame_indices"], dtype=np.int32),
        query_uv_frame0_processed=q,
        query_uv_frame0_source=np.asarray(m["query_uv_frame0_source_image"], dtype=np.float32),
        teacher_uv_processed=tuv.astype(np.float32),
        observed_uv_processed=ouv.astype(np.float32),
        teacher_valid=tv, observed_valid=ov,
        processed_resolution_hw=processed,
        query_reference_source_frame_index=np.int32(14),
        output_camera_source_frame_index=np.int32(20),
        time_basis=np.asarray("consecutive_davis_source_frame_index"),
        preview_mp4_fps_used=np.asarray(False),
    )
    print(json.dumps({"video_id":m["video_id"],"observed_shape":list(op.shape),"teacher_shape":list(tp.shape),"output":str(args.output)},indent=2))


if __name__ == "__main__":
    main()
