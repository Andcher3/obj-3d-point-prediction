#!/usr/bin/env python3
"""Rebuild a MotionForesight-compatible sparse NPZ from dense TrackCraft output.

This is a reconstructed adapter, not the unpublished official training-label
generator. It preserves the official SparseTrackDataset schema and marks the
source as reconstructed dense TrackCraft output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def sample_nearest(track_map: np.ndarray, uv: np.ndarray, source_hw: tuple[int, int]) -> np.ndarray:
    t, h, w, c = track_map.shape
    sh, sw = source_hw
    x = np.clip(np.rint(uv[:, 0] * w / sw), 0, w - 1).astype(np.int64)
    y = np.clip(np.rint(uv[:, 1] * h / sh), 0, h - 1).astype(np.int64)
    return track_map[:, y, x, :].astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", type=Path, required=True)
    ap.add_argument("--user", type=Path, required=True)
    ap.add_argument("--query-json", type=Path, required=True,
                    help="JSON containing query_uv_t0_source or query_sets.48")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--text", default="")
    args = ap.parse_args()

    dense = np.load(args.dense, allow_pickle=True)
    user = np.load(args.user, allow_pickle=True)
    if "track_map" not in dense.files:
        raise KeyError("dense NPZ must contain track_map")
    required_user = {"rgb", "depth_map", "extrinsics_w2c", "fx_fy_cx_cy"}
    missing = sorted(required_user.difference(user.files))
    if missing:
        raise KeyError(f"user NPZ missing {missing}")

    payload = json.loads(args.query_json.read_text(encoding="utf-8"))
    if "query_uv_t0_source" in payload:
        uv = np.asarray(payload["query_uv_t0_source"], dtype=np.float32)
    else:
        uv = np.asarray(payload["query_sets"]["48"], dtype=np.float32)
    rgb = np.asarray(user["rgb"])
    depth = np.asarray(user["depth_map"])
    extr = np.asarray(user["extrinsics_w2c"], dtype=np.float32)
    K = np.asarray(user["fx_fy_cx_cy"], dtype=np.float64)
    track_map = np.asarray(dense["track_map"], dtype=np.float32)
    t = min(len(rgb), len(depth), len(extr), track_map.shape[0])
    rgb, depth, extr, track_map = rgb[:t], depth[:t], extr[:t], track_map[:t]
    tracks = sample_nearest(track_map, uv, (rgb.shape[1], rgb.shape[2]))
    visibility = np.isfinite(tracks).all(axis=-1) & (np.linalg.norm(tracks, axis=-1) > 1e-8)
    out = {
        "rgb": rgb,
        "tracks_xyz": tracks,
        "query_uv": uv,
        "visibility": visibility,
        "extrinsics_w2c": extr,
        "image_size_hw": np.asarray(rgb.shape[1:3], dtype=np.int32),
        "fx_fy_cx_cy": K,
        "user_npz": np.asarray(str(args.user.resolve())),
        "text": np.asarray(args.text),
        "source_kind": np.asarray("reconstructed_from_dense_trackcraft_output"),
        "source_dense_npz": np.asarray(str(args.dense.resolve())),
        "source_note": np.asarray("Not the unpublished official MotionForesight sparse-label generator"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **out)
    print(json.dumps({
        "status": "PASS",
        "output": str(args.output),
        "tracks_xyz": list(tracks.shape),
        "query_count": int(len(uv)),
        "validity_fraction": float(visibility.mean()),
        "coordinate_claim": "track_map frame-0 camera coordinates; extrinsics frame-0 normalized",
        "official_generator": False,
    }, indent=2))


if __name__ == "__main__":
    main()
