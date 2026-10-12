#!/usr/bin/env python3
"""Audit and package one DAVIS sequence for the frozen joint-decoder probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def transform(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    xyz1 = np.concatenate([points, np.ones((*points.shape[:-1], 1), dtype=points.dtype)], axis=-1)
    return np.einsum("ij,...j->...i", matrix, xyz1)[..., :3]


def fit_similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    source_mean = source.mean(0)
    target_mean = target.mean(0)
    x = source - source_mean
    y = target - target_mean
    u, singular, vt = np.linalg.svd(x.T @ y)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    scale = float(singular.sum() / max(float(np.square(x).sum()), 1e-12))
    translation = target_mean - scale * (source_mean @ rotation)
    return scale, rotation, translation


def apply_similarity(points: np.ndarray, fit: tuple[float, np.ndarray, np.ndarray]) -> np.ndarray:
    scale, rotation, translation = fit
    return scale * (points @ rotation) + translation


def symmetric_knn(uv: np.ndarray, k: int = 3) -> list[tuple[int, int]]:
    distance = np.linalg.norm(uv[:, None] - uv[None], axis=-1)
    distance += np.eye(len(uv)) * 1e12
    return sorted({tuple(sorted((i, int(j)))) for i in range(len(uv)) for j in np.argsort(distance[i])[:k]})


def graph_audit(n: int, edges: list[tuple[int, int]], valid: np.ndarray) -> dict:
    keep = set(np.flatnonzero(valid).tolist())
    adjacency = {i: [] for i in keep}
    kept_edges = [(i, j) for i, j in edges if i in keep and j in keep]
    for i, j in kept_edges:
        adjacency[i].append(j)
        adjacency[j].append(i)
    components = []
    unseen = set(keep)
    while unseen:
        root = min(unseen)
        stack = [root]
        component = []
        unseen.remove(root)
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbour in adjacency[node]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)
        components.append(sorted(component))
    timer = 0
    discovery: dict[int, int] = {}
    low: dict[int, int] = {}
    bridges: list[tuple[int, int]] = []

    def visit(node: int, parent: int | None) -> None:
        nonlocal timer
        discovery[node] = low[node] = timer
        timer += 1
        for neighbour in adjacency[node]:
            if neighbour == parent:
                continue
            if neighbour not in discovery:
                visit(neighbour, node)
                low[node] = min(low[node], low[neighbour])
                if low[neighbour] > discovery[node]:
                    bridges.append(tuple(sorted((node, neighbour))))
            else:
                low[node] = min(low[node], discovery[neighbour])

    for node in sorted(keep):
        if node not in discovery:
            visit(node, None)
    return {
        "edge_count": len(kept_edges),
        "component_count": len(components),
        "component_sizes": sorted([len(x) for x in components], reverse=True),
        "bridge_edges": sorted(set(bridges)),
        "connected": len(components) == 1,
    }


def project(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = intrinsics
    z = np.maximum(points[..., 2], 1e-12)
    return np.stack([fx * points[..., 0] / z + cx, fy * points[..., 1] / z + cy], axis=-1)


def quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.quantile(values, 0.5)),
        "p90": float(np.quantile(values, 0.9)),
        "mean": float(values.mean()),
        "max": float(values.max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--observed-user", type=Path, required=True)
    ap.add_argument("--full-user", type=Path, required=True)
    ap.add_argument("--observed-dense", type=Path, required=True)
    ap.add_argument("--full-query-track", type=Path, required=True)
    ap.add_argument("--motionforesight-arrays", type=Path, required=True)
    ap.add_argument("--query-t0-json", type=Path, required=True)
    ap.add_argument("--mapping-json", type=Path, required=True)
    ap.add_argument("--cache-output", type=Path, required=True)
    ap.add_argument("--audit-output", type=Path, required=True)
    args = ap.parse_args()

    obs_user = np.load(args.observed_user, allow_pickle=True)
    full_user = np.load(args.full_user, allow_pickle=True)
    obs_dense = np.load(args.observed_dense)
    full_track = np.load(args.full_query_track, allow_pickle=True)
    mf = np.load(args.motionforesight_arrays, allow_pickle=True)
    t0_manifest = json.loads(args.query_t0_json.read_text(encoding="utf-8"))
    mapping = json.loads(args.mapping_json.read_text(encoding="utf-8"))

    query_t0 = np.asarray(t0_manifest["query_sets"]["48"], dtype=np.float64)
    query_frame0 = np.asarray(mapping["query_sets"]["48"], dtype=np.float64)
    if query_t0.shape != (48, 2) or query_frame0.shape != (48, 2):
        raise ValueError("expected exactly 48 ordered queries")
    if not np.allclose(np.asarray(full_track["query_uv_original"]), query_frame0, atol=1e-4):
        raise ValueError("full TrackCraft query order differs from verified frame-0 mapping")
    if not np.allclose(np.asarray(mf["query_uv_original"]), query_frame0, atol=1e-4):
        raise ValueError("MotionForesight query order differs from verified frame-0 mapping")

    dense_map = np.asarray(obs_dense["track_map"], dtype=np.float64)
    dense_uv = np.asarray(mapping["query_uv_frame0_dense"], dtype=np.int64)
    observed_frame0 = dense_map[:, dense_uv[:, 1], dense_uv[:, 0]]
    observed_w2c = np.asarray(obs_user["extrinsics_w2c"], dtype=np.float64)
    observed_history = transform(observed_w2c[6], observed_frame0)

    teacher_frame0 = np.asarray(full_track["trackcraft3r_tracks"], dtype=np.float64)
    full_w2c = np.asarray(full_user["extrinsics_w2c"], dtype=np.float64)
    teacher_t0_camera = transform(full_w2c[6], teacher_frame0)
    teacher_t0 = teacher_t0_camera[6]
    observed_t0 = observed_history[6]

    fold_error = np.full(48, np.nan, dtype=np.float64)
    fold_rows = []
    for fold in range(4):
        held = np.arange(48) % 4 == fold
        train = ~held
        fit = fit_similarity(teacher_t0[train], observed_t0[train])
        predicted = apply_similarity(teacher_t0[held], fit)
        fold_error[held] = np.linalg.norm(predicted - observed_t0[held], axis=-1)
        fold_rows.append({"fold": fold, "train_queries": int(train.sum()), "held_queries": int(held.sum()), "held_error_m": quantiles(fold_error[held])})
    bbox_3d = max(float(np.linalg.norm(np.ptp(observed_t0, axis=0))), 1e-8)
    registration_normalized = fold_error / bbox_3d

    full_fit = fit_similarity(teacher_t0, observed_t0)
    teacher_aligned = apply_similarity(teacher_t0_camera, full_fit)
    teacher_t0_aligned = teacher_aligned[6]
    future_teacher = teacher_aligned[7:16]

    full_intrinsics = np.asarray(full_user["fx_fy_cx_cy"], dtype=np.float64)
    teacher_anchor_uv = project(teacher_t0, full_intrinsics)
    bbox_2d = max(float(t0_manifest["bbox_diagonal_px"]), 1e-8)
    teacher_anchor_error = np.linalg.norm(teacher_anchor_uv - query_t0, axis=-1)
    teacher_anchor_normalized = teacher_anchor_error / bbox_2d
    observed_mapping_normalized = np.asarray(mapping["match_error_normalized_bbox"], dtype=np.float64)

    teacher_valid = np.isfinite(future_teacher).all(-1) & (future_teacher[..., 2] > 0)
    observed_valid = np.isfinite(observed_history).all(-1).all(0) & (observed_history[..., 2] > 0).all(0)
    oob_valid = np.asarray(full_track["oob_mask"], dtype=bool)
    per_query_valid = observed_valid & teacher_valid.all(0) & oob_valid
    validity = teacher_valid & per_query_valid[None]
    graph = graph_audit(48, symmetric_knn(query_t0, 3), per_query_valid)

    p0_all = np.asarray(mf["p0_query"], dtype=np.float32)
    h_all = np.asarray(mf["h_output_latent"], dtype=np.float32)
    if p0_all.shape != (22, 48, 3) or h_all.shape[:2] != (22, 48):
        raise ValueError(f"unexpected MotionForesight arrays: {p0_all.shape}, {h_all.shape}")
    p0 = p0_all[7:16]
    h = h_all[7:16]

    gates = {
        "query_count_48": True,
        "valid_queries_at_least_44": int(per_query_valid.sum()) >= 44,
        "observed_mapping_p90_le_5pct": float(np.quantile(observed_mapping_normalized, 0.9)) <= 0.05,
        "observed_mapping_max_le_10pct": float(observed_mapping_normalized.max()) <= 0.10,
        "teacher_anchor_p90_le_5pct": float(np.quantile(teacher_anchor_normalized, 0.9)) <= 0.05,
        "teacher_anchor_max_le_10pct": float(teacher_anchor_normalized.max()) <= 0.10,
        "registration_p90_le_10pct": float(np.quantile(registration_normalized, 0.9)) <= 0.10,
        "registration_90pct_below_10pct": float((registration_normalized <= 0.10).mean()) >= 0.90,
        "registration_max_le_20pct": float(registration_normalized.max()) <= 0.20,
        "valid_graph_connected": bool(graph["connected"]),
        "finite_baseline_and_teacher": bool(np.isfinite(p0).all() and np.isfinite(h).all() and np.isfinite(future_teacher).all()),
    }
    status = "PASS" if all(gates.values()) else "FAIL"

    args.cache_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.cache_output,
        p0_formal9=p0,
        h_output_latent_formal9=h,
        pseudo_gt_aligned=future_teacher.astype(np.float32),
        pseudo_gt_t0_aligned=teacher_t0_aligned.astype(np.float32),
        pseudo_gt_validity=validity,
        query_anchor_match_error_px=teacher_anchor_error.astype(np.float32),
        query_uv_frame0_dense=dense_uv.astype(np.float32),
        observed_only_trackcraft_t0=observed_t0.astype(np.float32),
        query_uv_t0_source=query_t0.astype(np.float32),
        query_uv_frame0_input=query_frame0.astype(np.float32),
        registration_crossfit_error_normalized=registration_normalized.astype(np.float32),
        coordinate_frame=np.asarray("observed_only_last_camera"),
        prediction_source=np.asarray("motionforesight_observed_only"),
        pseudo_gt_source=np.asarray("trackcraft3r_offline_full22_query_only"),
        time_basis=np.asarray("consecutive_davis_source_frame_index"),
    )
    payload = {
        "status": status,
        "video_id": t0_manifest["video_id"],
        "gates": gates,
        "valid_query_count": int(per_query_valid.sum()),
        "observed_mapping_error_normalized_bbox": quantiles(observed_mapping_normalized),
        "teacher_t0_anchor_error_px": quantiles(teacher_anchor_error),
        "teacher_t0_anchor_error_normalized_bbox": quantiles(teacher_anchor_normalized),
        "registration_crossfit_error_m": quantiles(fold_error),
        "registration_crossfit_error_normalized_3d_bbox": quantiles(registration_normalized),
        "registration_crossfit_fraction_below_10pct": float((registration_normalized <= 0.10).mean()),
        "registration_folds": fold_rows,
        "graph": graph,
        "shapes": {"p0": list(p0.shape), "h_output": list(h.shape), "teacher_future": list(future_teacher.shape)},
        "time": {
            "source_frame_indices": t0_manifest["source_frame_indices"],
            "observed_source_frame_indices": t0_manifest["observed_source_frame_indices"],
            "evaluated_future_source_frame_indices": t0_manifest["offline_future_source_frame_indices"][:9],
            "basis": t0_manifest["time_basis"],
            "preview_mp4_fps_used": False,
        },
        "sources": {
            "baseline": "MotionForesight prediction from seven observed frames only",
            "pseudo_gt": "offline full-22-frame TrackCraft3R query-only trajectory",
        },
    }
    args.audit_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
