#!/usr/bin/env python3
"""F1: audit baseline structural drift against same-input TrackCraft3R tracks.

This is a descriptive pilot. It does not train a Relation Head and does not
claim that TrackCraft3R is external ground truth. The comparison is performed
in the last-observed camera frame and uses the exact 48 queries from F0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def symmetric_knn(uv: np.ndarray, k: int = 3) -> list[tuple[int, int]]:
    d = np.linalg.norm(uv[:, None] - uv[None, :], axis=-1)
    d = d + np.eye(len(uv), dtype=np.float64) * 1e9
    directed = [set(np.argsort(d[i])[:k].tolist()) for i in range(len(uv))]
    edges = {
        tuple(sorted((i, int(j))))
        for i, js in enumerate(directed)
        for j in js
        if i != int(j)
    }
    return sorted(edges)


def transform_frame0_to_camera(
    tracks_frame0: np.ndarray, extrinsics_w2c: np.ndarray, camera_index: int
) -> np.ndarray:
    """Transform T,N,3 frame-0-camera points into camera_index coordinates."""
    t, n, _ = tracks_frame0.shape
    frame0_to_world = np.linalg.inv(np.asarray(extrinsics_w2c[0], dtype=np.float64))
    world_to_camera = np.asarray(extrinsics_w2c[camera_index], dtype=np.float64)
    xyz1 = np.concatenate(
        [tracks_frame0.astype(np.float64), np.ones((t, n, 1), dtype=np.float64)],
        axis=-1,
    )
    world = xyz1 @ frame0_to_world.T
    camera = world @ world_to_camera.T
    return camera[..., :3]


def quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def linear_slope(values: np.ndarray) -> float:
    x = np.arange(1, len(values) + 1, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or not np.isfinite(y).all():
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def pearson_horizon(values: np.ndarray) -> float:
    x = np.arange(1, len(values) + 1, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trackcraft-npz", type=Path, required=True)
    ap.add_argument("--motionforesight-npz", type=Path, required=True)
    ap.add_argument("--user-npz", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    tc = np.load(args.trackcraft_npz, allow_pickle=True)
    mf = np.load(args.motionforesight_npz, allow_pickle=True)
    user = np.load(args.user_npz, allow_pickle=True)

    tracks = np.asarray(tc["trackcraft3r_tracks"], dtype=np.float64)
    pred_all = np.asarray(mf["p0_query"], dtype=np.float64)
    uv = np.asarray(tc["query_uv_original"], dtype=np.float64)
    extrinsics = np.asarray(user["extrinsics_w2c"], dtype=np.float64)
    obs_count = int(np.asarray(user["observed_frame_count"]).item())

    checks = {
        "track_shape": list(tracks.shape),
        "prediction_shape": list(pred_all.shape),
        "query_shape": list(uv.shape),
        "observed_frame_count": obs_count,
        "finite": bool(
            np.isfinite(tracks).all()
            and np.isfinite(pred_all).all()
            and np.isfinite(extrinsics).all()
        ),
        "query_match": bool(
            tracks.shape == pred_all.shape
            and tracks.shape[1] == len(uv)
            and np.array_equal(
                uv, np.asarray(mf["query_uv_original"], dtype=np.float64)
            )
        ),
    }
    if not checks["finite"] or not checks["query_match"]:
        raise RuntimeError(f"F1 input audit failed: {checks}")
    if tracks.shape[0] != pred_all.shape[0] or obs_count >= tracks.shape[0]:
        raise RuntimeError(f"unexpected temporal shape: {checks}")

    gt = transform_frame0_to_camera(tracks, extrinsics, obs_count - 1)
    pred = pred_all
    future = np.arange(obs_count, tracks.shape[0])
    t0 = obs_count - 1
    pred_t0 = pred[t0]
    gt_t0 = gt[t0]
    edges = symmetric_knn(uv, args.k)
    edge_i = np.asarray([i for i, _ in edges], dtype=np.int64)
    edge_j = np.asarray([j for _, j in edges], dtype=np.int64)

    bbox_diag = float(np.linalg.norm(gt_t0.max(axis=0) - gt_t0.min(axis=0)))
    radius = float(np.median(np.linalg.norm(gt_t0 - gt_t0.mean(axis=0), axis=1)))
    scale = max(bbox_diag, 1e-8)

    rows: list[dict] = []
    for h, frame in enumerate(future, start=1):
        p = pred[frame]
        g = gt[frame]
        p_centered = p - p.mean(axis=0, keepdims=True)
        g_centered = g - g.mean(axis=0, keepdims=True)
        point_error = np.linalg.norm(p - g, axis=1)
        centered_error = np.linalg.norm(p_centered - g_centered, axis=1)
        translation_error = float(np.linalg.norm(p.mean(axis=0) - g.mean(axis=0)))

        pv = p[edge_j] - p[edge_i]
        gv = g[edge_j] - g[edge_i]
        edge_vector_error = np.linalg.norm(pv - gv, axis=1)
        edge_gt_length = np.linalg.norm(gv, axis=1)
        edge_relative = edge_vector_error / np.maximum(edge_gt_length, 1e-8)
        p_dist = np.linalg.norm(pv, axis=1)
        g_dist = edge_gt_length
        shape_relative = np.abs(p_dist - g_dist) / np.maximum(g_dist, 1e-8)

        rows.append(
            {
                "horizon": h,
                "source_frame_index": int(frame),
                "point_error_m": quantiles(point_error),
                "component_centered_point_error_m": quantiles(centered_error),
                "component_centered_point_error_normalized": quantiles(
                    centered_error / scale
                ),
                "centroid_translation_error_m": translation_error,
                "edge_vector_error_m": quantiles(edge_vector_error),
                "edge_relative_error": quantiles(edge_relative),
                "edge_length_shape_relative_error": quantiles(shape_relative),
                "predicted_centroid_m": p.mean(axis=0).tolist(),
                "target_centroid_m": g.mean(axis=0).tolist(),
            }
        )

    def series(key: str, subkey: str | None = None) -> np.ndarray:
        values = []
        for row in rows:
            value = row[key] if subkey is None else row[key][subkey]
            values.append(value)
        return np.asarray(values, dtype=np.float64)

    centered_p50 = series(
        "component_centered_point_error_normalized", "p50"
    )
    edge_rel_p50 = series("edge_relative_error", "p50")
    shape_rel_p50 = series("edge_length_shape_relative_error", "p50")
    center_translation = series("centroid_translation_error_m")

    # Pilot gate: a structural-drift signal must be nontrivial at late
    # horizons and show a positive horizon trend in at least two independent
    # structure measures. This is not a claim of statistical significance.
    late = slice(max(0, len(rows) - 5), len(rows))
    early = slice(0, min(5, len(rows)))
    centered_late = float(np.median(centered_p50[late]))
    edge_late = float(np.median(edge_rel_p50[late]))
    shape_late = float(np.median(shape_rel_p50[late]))
    centered_growth = centered_late - float(np.median(centered_p50[early]))
    edge_growth = edge_late - float(np.median(edge_rel_p50[early]))
    positive_trends = sum(
        [
            linear_slope(centered_p50) > 0.0,
            linear_slope(edge_rel_p50) > 0.0,
            linear_slope(shape_rel_p50) > 0.0,
        ]
    )
    drift_signal = bool(
        positive_trends >= 2
        and centered_late >= 0.05
        and edge_late >= 0.05
        and (centered_growth > 0.01 or edge_growth > 0.01)
    )

    payload = {
        "status": "PASS",
        "scope": "F1 baseline structural-drift pilot; descriptive and pseudo-GT based",
        "video_id": "9069bd26-0139-4e88-9957-9f9489b35a91",
        "protocol": {
            "observed_frame_count": obs_count,
            "future_frame_count": int(len(future)),
            "t0_frame_index": t0,
            "prediction_source": "MotionForesight observed-only p0_query",
            "comparison_source": "TrackCraft3R query-only tracks transformed to last-observed camera",
            "graph": f"2D symmetric kNN k={args.k}",
        },
        "input_checks": checks,
        "reference_scales_m": {
            "t0_bbox_diagonal": bbox_diag,
            "t0_median_center_radius": radius,
            "normalization_scale": scale,
        },
        "horizon_rows": rows,
        "horizon_series": {
            "centered_point_error_normalized_p50": centered_p50.tolist(),
            "edge_relative_error_p50": edge_rel_p50.tolist(),
            "edge_length_shape_relative_error_p50": shape_rel_p50.tolist(),
            "centroid_translation_error_m": center_translation.tolist(),
        },
        "trend_summary": {
            "centered_point_error_normalized": {
                "slope_per_horizon": linear_slope(centered_p50),
                "pearson_horizon_correlation": pearson_horizon(centered_p50),
            },
            "edge_relative_error": {
                "slope_per_horizon": linear_slope(edge_rel_p50),
                "pearson_horizon_correlation": pearson_horizon(edge_rel_p50),
            },
            "edge_length_shape_relative_error": {
                "slope_per_horizon": linear_slope(shape_rel_p50),
                "pearson_horizon_correlation": pearson_horizon(shape_rel_p50),
            },
            "late_median": {
                "centered_point_error_normalized": centered_late,
                "edge_relative_error": edge_late,
                "edge_length_shape_relative_error": shape_late,
            },
            "early_to_late_growth": {
                "centered_point_error_normalized": centered_growth,
                "edge_relative_error": edge_growth,
            },
            "positive_trend_measures": int(positive_trends),
            "pilot_structural_drift_signal": drift_signal,
            "interpretation": (
                "Proceed to F2 probe on this pilot if true; "
                "do not generalize to all videos from one sample."
            ),
        },
        "limitations": [
            "TrackCraft3R is a pseudo-GT source, not external ground truth.",
            "Only one complete F0 video is available in this pilot.",
            "The gate is descriptive and not a statistical significance test.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
