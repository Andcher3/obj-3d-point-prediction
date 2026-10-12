#!/usr/bin/env python3
"""Compare raw and Pj0-anchored DAVIS tracks, including per-frame mask hits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from build_davis_visualization_projection_bundle import project


def stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return {
        "p50": float(np.quantile(x, 0.5)), "p90": float(np.quantile(x, 0.9)),
        "mean": float(np.mean(x)), "max": float(np.max(x)),
    }


def source_mask_hits(uv_processed: np.ndarray, source_indices: list[int], processed_hw: tuple[int, int], mask_dir: Path) -> dict:
    ph, pw = processed_hw
    counts, fractions = [], []
    for uv, frame in zip(uv_processed, source_indices):
        mask = np.asarray(Image.open(mask_dir / f"{frame:05d}.png"), dtype=np.uint8) == 1
        h, w = mask.shape
        xy = np.rint(uv * np.asarray([w / pw, h / ph])).astype(np.int64)
        inside = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
        hits = int(np.sum(mask[xy[inside, 1], xy[inside, 0]]))
        counts.append(hits); fractions.append(hits / len(uv))
    return {
        "source_frame_indices": source_indices, "counts_of_48": counts,
        "fraction_stats": stats(np.asarray(fractions)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--anchored-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.source_dir / "query_frame0_manifest.json").read_text(encoding="utf-8"))
    output = {
        "status": "DIAGNOSTIC_ONLY_NOT_TRAINING_APPROVAL", "video_id": manifest["video_id"],
        "source_raw_directory": str(args.source_dir), "anchored_directory": str(args.anchored_dir),
        "contexts": {},
    }
    projected = {}
    contexts = (
        ("observed7", manifest["observed_source_frame_indices"], args.source_dir / "observed7_user.npz", args.source_dir / "observed7_query_tracks.npz", "trackcraft3r_tracks"),
        ("official_teacher22", list(range(14, 36)), args.anchored_dir / "full23_user.npz", args.anchored_dir / "full22_query_tracks.npz", "trackcraft3r_tracks_raw_interleaved"),
    )
    for context, frames, user_path, raw_path, raw_key in contexts:
        with np.load(user_path, allow_pickle=True) as user, \
             np.load(raw_path) as raw_npz, \
             np.load(args.anchored_dir / ("observed7_query_tracks.npz" if context == "observed7" else "full22_query_tracks.npz")) as anchor_npz:
            raw = np.asarray(raw_npz[raw_key], dtype=np.float64)
            anchored = np.asarray(anchor_npz["trackcraft3r_tracks"], dtype=np.float64)
            stored_w2c = np.asarray(user["extrinsics_w2c"], dtype=np.float64)[: len(raw)]
            w2c = np.einsum("tij,jk->tik", stored_w2c, np.linalg.inv(stored_w2c[0]))
            k = np.asarray(user["fx_fy_cx_cy"], dtype=np.float64)
            raw_uv, raw_valid = project(raw, w2c, k)
            anchored_uv, anchored_valid = project(anchored, w2c, k)
            if not raw_valid.all() or not anchored_valid.all():
                raise ValueError("nonpositive projection depth")
            processed_hw = tuple(int(x) for x in user["depth_map"].shape[1:3])
            projected[context] = anchored_uv
            delta_key = "frame0_anchor_delta_xyz" if context == "observed7" else "frame0_anchor_delta_xyz_runs"
            delta = np.asarray(anchor_npz[delta_key], dtype=np.float64)
            output["contexts"][context] = {
                "raw_frame0_query_error_px": stats(np.linalg.norm(raw_uv[0] - anchor_npz["query_uv_original"], axis=-1)),
                "anchored_frame0_query_error_px": stats(np.linalg.norm(anchored_uv[0] - anchor_npz["query_uv_original"], axis=-1)),
                "raw_mask_hits": source_mask_hits(raw_uv, frames, processed_hw, args.mask_dir),
                "anchored_mask_hits": source_mask_hits(anchored_uv, frames, processed_hw, args.mask_dir),
                "temporal_displacement_max_abs_change": float(np.max(np.abs(
                    (anchored - anchored[0:1]) - (raw - raw[0:1])
                ))),
                "anchor_delta_norm": stats(np.linalg.norm(delta, axis=-1).reshape(-1)),
            }
            if context == "official_teacher22":
                output["contexts"][context]["anchor_delta_norm_per_run"] = [
                    stats(np.linalg.norm(run_delta, axis=-1)) for run_delta in delta
                ]
                output["contexts"][context]["per_run_temporal_displacement_max_abs_change"] = np.asarray(
                    anchor_npz["per_run_temporal_displacement_max_abs_change"], dtype=float
                ).tolist()
    raw_audit = json.loads((args.source_dir / "quality_audit.json").read_text(encoding="utf-8"))
    anchored_audit = json.loads((args.anchored_dir / "quality_audit.json").read_text(encoding="utf-8"))
    output["quality_before_legacy_direct22"] = raw_audit
    output["quality_after"] = anchored_audit
    output["cross_context_t0_uv_after_px"] = stats(np.linalg.norm(projected["observed7"][6] - projected["official_teacher22"][6], axis=-1))
    (args.anchored_dir / "pj0_anchor_comparison.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "video_id": output["video_id"], "quality_before_legacy_direct22": raw_audit["status"],
        "quality_after": anchored_audit["status"],
        "teacher_frame0_mask_before_after": [output["contexts"]["official_teacher22"]["raw_mask_hits"]["counts_of_48"][0], output["contexts"]["official_teacher22"]["anchored_mask_hits"]["counts_of_48"][0]],
        "future21_29_mask_counts_after": output["contexts"]["official_teacher22"]["anchored_mask_hits"]["counts_of_48"][7:16],
    }, indent=2))


if __name__ == "__main__":
    main()
