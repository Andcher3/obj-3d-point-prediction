#!/usr/bin/env python3
"""Derive Pj0-anchored query tracks without modifying raw TrackCraft3R outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def derive(user_path: Path, raw_path: Path, output_path: Path) -> dict:
    with np.load(user_path, allow_pickle=True) as user, np.load(raw_path) as raw_npz:
        raw = np.asarray(raw_npz["trackcraft3r_tracks"], dtype=np.float64)
        query_original = np.asarray(raw_npz["query_uv_original"], dtype=np.float64)
        query_model = np.asarray(raw_npz["query_uv_model"], dtype=np.float64)
        oob = np.asarray(raw_npz["oob_mask"], dtype=bool)
        model_h, model_w = (int(v) for v in raw_npz["model_resolution_hw"])
        depth = np.asarray(user["depth_map"][0], dtype=np.float64)
        k = np.asarray(user["fx_fy_cx_cy"], dtype=np.float64)
        w2c0 = np.asarray(user["extrinsics_w2c"][0], dtype=np.float64)
        if raw.shape[1] != int(oob.sum()) or not np.all(oob):
            raise ValueError("this fixed 48-query derivation requires every query to be in bounds")
        if np.max(np.abs(w2c0 - np.eye(4))) > 1e-5:
            raise ValueError("input frame-0 w2c is not normalized to identity")
        h, w = depth.shape
        scale = np.asarray([model_w / w, model_h / h], dtype=np.float64)
        expected_model = query_original * scale
        if not np.allclose(expected_model, query_model, atol=5e-4):
            raise ValueError("stored model query coordinates differ from stretch-resize semantics")
        k_model = k * np.asarray([scale[0], scale[1], scale[0], scale[1]])
        depth_model = np.asarray(
            Image.fromarray(depth.astype(np.float32), mode="F").resize(
                (model_w, model_h), Image.Resampling.NEAREST
            ),
            dtype=np.float64,
        )
        uv_int = query_model.astype(np.int64)
        z = depth_model[uv_int[:, 1], uv_int[:, 0]]
        if not np.isfinite(z).all() or not np.all(z > 0):
            raise ValueError("invalid Pj0 depth at an integer model query")
        fx, fy, cx, cy = k_model
        pj0 = np.stack(
            ((uv_int[:, 0] - cx) / fx * z, (uv_int[:, 1] - cy) / fy * z, z), axis=-1
        )
        delta = pj0 - raw[0]
        anchored = raw + delta[None]
        raw_displacement = raw - raw[0:1]
        anchored_displacement = anchored - anchored[0:1]
        displacement_difference = float(np.max(np.abs(raw_displacement - anchored_displacement)))
        frame0_difference = float(np.max(np.abs(anchored[0] - pj0)))
        payload = {key: raw_npz[key] for key in raw_npz.files if key != "trackcraft3r_tracks"}
        payload.update(
            trackcraft3r_tracks=anchored.astype(np.float32),
            trackcraft3r_tracks_raw=raw.astype(np.float32),
            frame0_pj_anchor_xyz=pj0.astype(np.float32),
            frame0_anchor_delta_xyz=delta.astype(np.float32),
            frame0_anchor_method=np.asarray("per-query constant xyz translation to same-run integer-pixel Pj0"),
            coordinate_frame=np.asarray("frame0_camera_common_coordinates"),
            raw_source_path=np.asarray(str(raw_path)),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return {
        "raw_source": str(raw_path),
        "output": str(output_path),
        "shape": list(raw.shape),
        "frame0_anchor_max_abs_xyz": frame0_difference,
        "temporal_displacement_max_abs_change": displacement_difference,
        "anchor_delta_norm": {
            "p50": float(np.quantile(np.linalg.norm(delta, axis=-1), 0.5)),
            "p90": float(np.quantile(np.linalg.norm(delta, axis=-1), 0.9)),
            "max": float(np.max(np.linalg.norm(delta, axis=-1))),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--contexts", nargs="+", choices=("observed7", "full22"),
        default=("observed7", "full22"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "DERIVED_NOT_QUALITY_APPROVED",
        "video_id": args.source_dir.name,
        "source_raw_directory": str(args.source_dir),
        "method": "same-run integer-query Pj0 per-query constant 3D translation",
        "contexts_requested": list(args.contexts),
        "contexts": {},
    }
    configurations = {context: (user, tracks) for context, user, tracks in (
        ("observed7", "observed7_user.npz", "observed7_query_tracks.npz"),
        ("full22", "full22_user.npz", "full22_query_tracks.npz"),
    )}
    for context in args.contexts:
        user, tracks = configurations[context]
        summary["contexts"][context] = derive(
            args.source_dir / user, args.source_dir / tracks, args.output_dir / tracks
        )
    (args.output_dir / "pj0_anchor_derivation.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
