#!/usr/bin/env python3
"""Summarize formal E3 P0/H interface and coordinate-registration audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for report_path in sorted(args.root.glob("*/formal_interface_aligned.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        cache_path = report_path.with_name("p0_h_cache_aligned.npz")
        cache = np.load(cache_path, allow_pickle=True)
        residual = np.linalg.norm(
            cache["pseudo_gt_t0_aligned"] - cache["observed_only_trackcraft_t0"], axis=-1
        )
        target = cache["observed_only_trackcraft_t0"]
        diagonal = float(np.linalg.norm(target.max(axis=0) - target.min(axis=0)))
        normalized = residual / max(diagonal, 1e-12)
        row = {
            "video_id": report["video_id"],
            "status": report["status"],
            "no_future_input": not report["input"]["future_frame_tensor_supplied_to_motionforesight"],
            "p0_shape": report["outputs"]["p0_future_shape"],
            "h_output_shape": report["outputs"]["h_output_latent_shape"],
            "h_dit_shape": report["outputs"]["h_last_dit_shape"],
            "finite": report["outputs"]["finite"],
            "permutation_error": report["query_alignment"]["permutation_sampling_max_abs_error"],
            "query_anchor_error_px_max": report["query_alignment"]["match_error_px_max"],
            "time_mapping_valid": report["time_mapping"]["valid"],
            "time_mapping_unique": report["time_mapping"]["unique_mapping"],
            "time_match_error_s_max": max(report["time_mapping"]["formal_match_errors_s"]),
            "alignment_scale": report["pseudo_gt_alignment"]["scale"],
            "alignment_rotation_deg": report["pseudo_gt_alignment"]["rotation_angle_deg"],
            "alignment_translation_norm": report["pseudo_gt_alignment"]["translation_norm"],
            "alignment_error_median_m": float(np.median(residual)),
            "alignment_error_p90_m": float(np.quantile(residual, 0.9)),
            "alignment_error_max_m": float(np.max(residual)),
            "alignment_error_median_bbox_diag": float(np.median(normalized)),
            "alignment_error_p90_bbox_diag": float(np.quantile(normalized, 0.9)),
            "alignment_error_max_bbox_diag": float(np.max(normalized)),
            "alignment_count_over_10pct_bbox_diag": int(np.sum(normalized > 0.10)),
            "cache_sha256": sha256(cache_path),
        }
        rows.append(row)

    interface_gate = bool(rows) and all(
        row["status"] == "PASS"
        and row["no_future_input"]
        and row["p0_shape"] == [15, 48, 3]
        and row["h_output_shape"] == [15, 48, 16]
        and row["h_dit_shape"] == [15, 48, 1536]
        and row["finite"]
        and row["permutation_error"] == 0.0
        and row["time_mapping_valid"]
        and row["time_mapping_unique"]
        for row in rows
    )
    payload = {
        "date": "2026-09-23",
        "scope": "MotionForesight observed-only formal E3 interface preflight",
        "video_count": len(rows),
        "interface_gate_pass": interface_gate,
        "supervision_registration_gate_status": "BLOCKED",
        "supervision_registration_blocker": (
            "The existing E2 pseudo-GT was estimated in a different DA3 sequence context. "
            "Shared-t0 similarity registration leaves large video-dependent residuals."
        ),
        "relation_head_training_started": False,
        "rows": rows,
        "limits": [
            "The released checkpoint's physical time-rate calibration on Ego4D is not established.",
            "Pseudo-GT registration is estimated from shared t0 queries only; per-query residuals remain a supervision-validity issue.",
            "The current five videos cover rigid and articulated objects only.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
