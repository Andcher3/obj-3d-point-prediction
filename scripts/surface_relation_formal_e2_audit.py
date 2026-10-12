#!/usr/bin/env python3
"""Audit whether a trajectory directory satisfies the formal E2 data protocol."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from surface_relation_e2_lite import infer_video_group


def audit(data_root: Path, min_videos: int, min_queries: int, max_queries: int) -> dict:
    files = sorted(data_root.rglob("*.npz"))
    eligible_files = []
    for path in files:
        relative_path = path.relative_to(data_root).as_posix()
        lower_path = relative_path.lower()
        if "surface-relation" in lower_path or "formal-e2" in lower_path or "vipe-alltracker" in lower_path:
            continue
        eligible_files.append(path)
    records = []
    for path in eligible_files:
        relative_path = path.relative_to(data_root).as_posix()
        try:
            with np.load(path) as loaded:
                required = {"future_3d", "pseudo_gt", "points_3d_history", "points_2d_at_t0", "intrinsics"}
                missing = sorted(required - set(loaded.files))
                if missing:
                    records.append({"path": relative_path, "status": "missing_keys", "missing": missing})
                    continue
                future = loaded["future_3d"]
                pseudo_gt = loaded["pseudo_gt"]
                history = loaded["points_3d_history"]
                points_2d = loaded["points_2d_at_t0"]
                intrinsics = loaded["intrinsics"]
                source_video = str(loaded["source_video"].item()) if "source_video" in loaded.files else infer_video_group(path)
                trajectory_source = str(loaded["trajectory_source"].item()) if "trajectory_source" in loaded.files else None
                prediction_source = str(loaded["prediction_source"].item()) if "prediction_source" in loaded.files else None
                validity = loaded["validity"] if "validity" in loaded.files else None
                projection = history[-1, :, :2] / np.clip(history[-1, :, 2:3], 1e-8, None)
                projection[:, 0] = projection[:, 0] * intrinsics[0, 0] + intrinsics[0, 2]
                projection[:, 1] = projection[:, 1] * intrinsics[1, 1] + intrinsics[1, 2]
                projection_error = float(np.max(np.abs(projection - points_2d)))
                required_time_keys = {"t0_seconds", "future_timestamps_s", "pseudo_gt_timestamps_s"}
                missing_time_keys = sorted(required_time_keys - set(loaded.files))
                records.append(
                    {
                        "path": relative_path,
                        "source_video_group": source_video,
                        "status": "ok",
                        "query_count": int(history.shape[1]),
                        "history_frames": int(history.shape[0]),
                        "prediction_shape": list(future.shape),
                        "pseudo_gt_shape": list(pseudo_gt.shape),
                        "projection_max_abs_error": projection_error,
                        "has_time_metadata": not missing_time_keys,
                        "missing_time_keys": missing_time_keys,
                        "trajectory_source": trajectory_source,
                        "prediction_source": prediction_source,
                        "valid_fraction": float(np.asarray(validity).mean()) if validity is not None else None,
                    }
                )
        except Exception as exc:
            records.append({"path": relative_path, "status": "error", "error": repr(exc)})

    valid = [record for record in records if record.get("status") == "ok"]
    by_video: dict[str, list[dict]] = defaultdict(list)
    for record in valid:
        by_video[record["source_video_group"]].append(record)

    video_summaries = {}
    for video, items in sorted(by_video.items()):
        counts = [item["query_count"] for item in items]
        video_summaries[video] = {
            "file_count": len(items),
            "query_count_min": min(counts),
            "query_count_max": max(counts),
            "query_count_values": sorted(set(counts)),
            "projection_error_max": max(item["projection_max_abs_error"] for item in items),
            "time_metadata_file_count": sum(item["has_time_metadata"] for item in items),
            "paths": [item["path"] for item in items],
        }

    reasons = []
    if len(video_summaries) < min_videos:
        reasons.append(f"independent_source_videos={len(video_summaries)} < required_min={min_videos}")
    for video, summary in video_summaries.items():
        if summary["query_count_min"] < min_queries or summary["query_count_max"] > max_queries:
            reasons.append(
                f"{video}: query_count_range={summary['query_count_min']}..{summary['query_count_max']} "
                f"outside {min_queries}..{max_queries}"
            )
        if summary["time_metadata_file_count"] < summary["file_count"]:
            reasons.append(f"{video}: missing t0_seconds/future_timestamps_s/pseudo_gt_timestamps_s in some files")
    for item in valid:
        if item["history_frames"] != 3:
            reasons.append(f"{item['path']}: history_frames={item['history_frames']} != 3")
        if item["prediction_shape"] != [9, item["query_count"], 3] or item["pseudo_gt_shape"] != [9, item["query_count"], 3]:
            reasons.append(f"{item['path']}: future/pseudo_gt must both be [9,query_count,3]")
        if item["trajectory_source"] != "trackcraft3r_pseudo_gt":
            reasons.append(f"{item['path']}: trajectory_source must be trackcraft3r_pseudo_gt")
        if item["prediction_source"] != "controlled_perturbation_baseline":
            reasons.append(f"{item['path']}: prediction_source must be controlled_perturbation_baseline")
        if item["valid_fraction"] is None or item["valid_fraction"] < 1.0:
            reasons.append(f"{item['path']}: all selected history/future tracks must be valid")
    if valid and max(item["projection_max_abs_error"] for item in valid) > 1e-3:
        reasons.append("points_2d_at_t0 does not match projection of observed 3D points within 1e-3")

    return {
        "formal_e2_ready": not reasons,
        "protocol": {
            "minimum_independent_videos": min_videos,
            "query_count_range": [min_queries, max_queries],
            "required_observed_history": True,
            "required_time_metadata": True,
            "required_projection_consistency_tolerance": 1e-3,
            "main_graph": "2d_symmetric_knn_k3",
            "minimum_comparison": "observed_3d_symmetric_knn_k3",
            "hybrid": "deferred_until_real_mask_depth_boundary_data",
        },
        "data_root": str(data_root),
        "npz_file_count": len(eligible_files),
        "valid_npz_file_count": len(valid),
        "source_video_count": len(video_summaries),
        "source_videos": video_summaries,
        "reasons_not_ready": reasons,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("experiences_results"))
    parser.add_argument("--output", type=Path, default=Path("experiences_results/surface-relation-formal-e2-audit-2026-09-22.json"))
    parser.add_argument("--min-videos", type=int, default=5)
    parser.add_argument("--min-queries", type=int, default=48)
    parser.add_argument("--max-queries", type=int, default=64)
    args = parser.parse_args()
    payload = audit(args.data_root, args.min_videos, args.min_queries, args.max_queries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("formal_e2_ready", "npz_file_count", "source_video_count", "reasons_not_ready")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
