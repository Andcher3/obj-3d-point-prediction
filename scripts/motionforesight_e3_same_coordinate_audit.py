#!/usr/bin/env python3
"""Audit MotionForesight P0/H cache alignment and derive relation targets.

This is an interface/label audit only. It does not train a Relation Head.
The cache must contain decoded-coordinate predictions and an explicitly aligned
pseudo-GT for the formal nine future slots.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def symmetric_knn(uv: np.ndarray, k: int = 3) -> list[tuple[int, int]]:
    d = np.linalg.norm(uv[:, None] - uv[None, :], axis=-1)
    n = len(uv)
    directed = {i: set(np.argsort(d[i] + np.eye(n)[i] * 1e9)[:k].tolist()) for i in range(n)}
    edges = set()
    for i, js in directed.items():
        for j in js:
            if i != j:
                edges.add(tuple(sorted((i, int(j)))))
    return sorted(edges)


def edge_vectors(x: np.ndarray, edges: list[tuple[int, int]]) -> np.ndarray:
    return np.stack([x[:, j] - x[:, i] for i, j in edges], axis=1)


def summarize(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    required = {
        "p0_formal9", "h_output_latent_formal9", "h_last_dit_formal9",
        "pseudo_gt_aligned", "pseudo_gt_validity", "query_uv_t0_source",
        "formal_future_slot_indices", "formal_future_target_timestamps_s",
        "observed_timestamps_s", "prediction_timestamps_s",
    }
    missing = sorted(required.difference(z.files))
    if missing:
        return {"file": str(path), "status": "FAIL", "missing_keys": missing}

    pred = np.asarray(z["p0_formal9"], dtype=np.float64)
    gt = np.asarray(z["pseudo_gt_aligned"], dtype=np.float64)
    valid = np.asarray(z["pseudo_gt_validity"], dtype=bool)
    uv = np.asarray(z["query_uv_t0_source"], dtype=np.float64)
    slots = np.asarray(z["formal_future_slot_indices"], dtype=np.int64)
    target_ts = np.asarray(z["formal_future_target_timestamps_s"], dtype=np.float64)
    pred_ts = np.asarray(z["prediction_timestamps_s"], dtype=np.float64)
    obs_ts = np.asarray(z["observed_timestamps_s"], dtype=np.float64)

    shape_ok = pred.shape == gt.shape == (9, 48, 3) and valid.shape == (9, 48)
    finite_ok = bool(np.isfinite(pred).all() and np.isfinite(gt).all())
    slot_ok = bool(np.array_equal(slots, np.arange(9)) and len(pred_ts) >= 15)
    timestamp_error = np.abs(pred_ts[slots] - target_ts) if len(pred_ts) > int(slots.max()) else np.full(9, np.inf)
    timestamp_ok = bool(np.isfinite(timestamp_error).all() and np.max(timestamp_error) <= 0.5 / 15.0 + 1e-9)

    edges = symmetric_knn(uv, 3)
    edge_gt = edge_vectors(gt, edges)
    edge_pred = edge_vectors(pred, edges)
    edge_valid = valid[:, [i for i, _ in edges]] & valid[:, [j for _, j in edges]]
    edge_err = np.linalg.norm(edge_pred - edge_gt, axis=-1)
    edge_scale = np.linalg.norm(edge_gt, axis=-1)
    edge_mask = edge_valid & np.isfinite(edge_err)
    node_err = np.linalg.norm(pred - gt, axis=-1)
    node_mask = valid & np.isfinite(node_err)
    gt_scale = float(np.median(np.linalg.norm(gt - gt[0:1], axis=-1)))
    pred_scale = float(np.median(np.linalg.norm(pred - pred[0:1], axis=-1)))

    def stats(values, mask):
        v = values[mask]
        if v.size == 0:
            return {"count": 0}
        return {"count": int(v.size), "p50": float(np.quantile(v, .5)),
                "p90": float(np.quantile(v, .9)), "max": float(np.max(v)),
                "mean": float(np.mean(v))}

    status = "PASS" if (shape_ok and finite_ok and slot_ok and timestamp_ok and np.all(valid.sum(1) >= 44)) else "FAIL"
    return {
        "file": str(path), "status": status,
        "coordinate_frame": str(np.asarray(z["coordinate_frame"]).item()),
        "prediction_source": str(np.asarray(z["prediction_source"]).item()),
        "pseudo_gt_source": str(np.asarray(z["pseudo_gt_source"]).item()),
        "shape_check": {"ok": bool(shape_ok), "p0_formal9": list(pred.shape), "pseudo_gt_aligned": list(gt.shape), "validity": list(valid.shape)},
        "finite_check": finite_ok,
        "time_check": {"slot_indices": slots.tolist(), "slot_ok": slot_ok, "max_match_error_s": float(np.max(timestamp_error)), "timestamp_ok": timestamp_ok, "observed_frame_count": int(len(obs_ts))},
        "query_check": {"count": int(len(uv)), "edge_count_2d_knn3": len(edges), "valid_per_horizon": valid.sum(1).astype(int).tolist()},
        "decoded_coordinate_relation": {
            "node_error_m": stats(node_err, node_mask),
            "edge_vector_error_m": stats(edge_err, edge_mask),
            "edge_vector_gt_norm_m": stats(edge_scale, edge_mask),
            "edge_relative_error": stats(edge_err / np.maximum(edge_scale, 1e-8), edge_mask),
            "gt_temporal_displacement_scale_m": gt_scale,
            "pred_temporal_displacement_scale_m": pred_scale,
        },
        "supervision_ready": False,
        "supervision_note": "Cache-level decoded-coordinate audit only; no training release.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rows = [summarize(p) for p in sorted(args.cache_dir.glob("*.npz"))]
    payload = {
        "scope": "MotionForesight same-coordinate P0/H and decoded-coordinate relation audit",
        "source_chain": "SparseTrackDataset uses last-observed-camera transform and shared observed-Pj normalization; cache files are derived audit artifacts",
        "relation_target": "2D symmetric kNN k=3 edge vectors in decoded aligned coordinates",
        "relation_loss_ready": False,
        "videos": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
