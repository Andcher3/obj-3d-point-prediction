#!/usr/bin/env python3
"""Formal E2 runner for timestamp-aligned Ego4D trajectory bundles."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from surface_relation_e2_lite import (
    graph_diagnostics,
    incidence_matrix,
    noisy_relation,
    project_points,
    quantile_summary,
    solve_component_gauge,
    symmetric_knn_edges,
    trajectory_metrics,
)


TARGET_FPS = 15.0
QUERY_COUNTS = (32, 48, 64)
CONDITIONS = ("oracle", "gaussian_0.01", "gaussian_0.05", "gaussian_0.10", "gaussian_0.20", "outlier_0.05", "outlier_0.10", "outlier_0.20")


def nearest_indices(source_times: np.ndarray, target_times: np.ndarray) -> tuple[list[int | None], list[float]]:
    indices: list[int | None] = []
    errors: list[float] = []
    tolerance = 0.5 / TARGET_FPS
    for target in target_times:
        index = int(np.argmin(np.abs(source_times - target)))
        error = float(abs(source_times[index] - target))
        indices.append(index if error <= tolerance else None)
        errors.append(error)
    return indices, errors


def align_future(bundle: dict, t0_seconds: float) -> tuple[np.ndarray | None, dict]:
    target_times = np.asarray([t0_seconds + j / TARGET_FPS for j in range(1, 10)], dtype=np.float64)
    future_times = np.asarray(bundle["future_timestamps_s"], dtype=np.float64)
    gt_times = np.asarray(bundle["pseudo_gt_timestamps_s"], dtype=np.float64)
    future_indices, future_errors = nearest_indices(future_times, target_times)
    gt_indices, gt_errors = nearest_indices(gt_times, target_times)
    valid = all(index is not None for index in future_indices + gt_indices)
    alignment = {
        "target_times_s": target_times.tolist(),
        "future_indices": future_indices,
        "pseudo_gt_indices": gt_indices,
        "future_match_errors_s": future_errors,
        "pseudo_gt_match_errors_s": gt_errors,
        "valid": valid,
    }
    if not valid:
        return None, alignment
    future = np.asarray(bundle["future_3d"])[future_indices]
    pseudo_gt = np.asarray(bundle["pseudo_gt"])[gt_indices]
    return np.stack([future, pseudo_gt], axis=0), alignment


def run_bundle(bundle_path: Path, base_seed: int, output_root: Path, gate: str, trajectory_source: str) -> dict:
    with np.load(bundle_path, allow_pickle=False) as loaded:
        bundle = {key: loaded[key] for key in loaded.files}
    required = {"future_3d", "future_timestamps_s", "pseudo_gt", "pseudo_gt_timestamps_s", "points_3d_history", "points_2d_at_t0", "intrinsics", "t0_seconds"}
    missing = sorted(required - set(bundle))
    if missing:
        return {"bundle": str(bundle_path), "ready": False, "reason": f"missing_keys={missing}"}
    aligned, alignment = align_future(bundle, float(bundle["t0_seconds"]))
    if aligned is None:
        return {"bundle": str(bundle_path), "ready": False, "alignment": alignment, "reason": "timestamp_match_failed"}
    future_all, gt_all = aligned
    history_all = np.asarray(bundle["points_3d_history"], dtype=np.float64)
    points_2d_all = np.asarray(bundle["points_2d_at_t0"], dtype=np.float64)
    history_2d_all = project_points(history_all.copy(), np.asarray(bundle["intrinsics"], dtype=np.float64))
    max_queries = min(future_all.shape[1], gt_all.shape[1], history_all.shape[1], points_2d_all.shape[0])
    if max_queries < 48:
        return {"bundle": str(bundle_path), "ready": False, "alignment": alignment, "reason": f"valid_query_count={max_queries}<48"}

    source_video = str(bundle["source_video"].item()) if "source_video" in bundle else bundle_path.stem
    motion_category = str(bundle["motion_category"].item()) if "motion_category" in bundle else "unknown"
    records = []
    density_outputs: dict[tuple[str, int, str, int], torch.Tensor] = {}
    for query_count in QUERY_COUNTS:
        future = torch.tensor(future_all[:, :query_count].tolist(), dtype=torch.float64)
        pseudo_gt = torch.tensor(gt_all[:, :query_count].tolist(), dtype=torch.float64)
        history = history_all[:, :query_count]
        history_2d = history_2d_all[:, :query_count]
        points_2d = points_2d_all[:query_count]
        graph_edges = {
            "knn_2d": symmetric_knn_edges(points_2d, 3),
            "knn_3d": symmetric_knn_edges(history[-1], 3),
        }
        evaluation_edges = sorted(set(graph_edges["knn_2d"]) | set(graph_edges["knn_3d"]))
        for graph_type, graph_points in (("knn_2d", points_2d), ("knn_3d", history[-1])):
            edges = graph_edges[graph_type]
            if len(edges) == 0:
                continue
            incidence = incidence_matrix(query_count, edges)
            graph_stats = graph_diagnostics(edges, history, history_2d, graph_type, 3)
            oracle_delta = torch.einsum("en,tnd->ted", incidence, pseudo_gt - future)
            target_relations = torch.einsum("en,tnd->ted", incidence, pseudo_gt)
            edge_scale = float(torch.median(torch.linalg.vector_norm(target_relations, dim=-1)))
            for condition_index, condition in enumerate(CONDITIONS):
                seed_count = 1 if condition == "oracle" else 20
                for seed_index in range(seed_count):
                    rng = np.random.default_rng(np.random.SeedSequence([base_seed, query_count, condition_index, seed_index]))
                    proposed = noisy_relation(oracle_delta, condition, edge_scale, rng)
                    correction, diagnostics = solve_component_gauge(proposed, edges, query_count)
                    prediction = future + correction
                    metrics = trajectory_metrics(prediction, pseudo_gt, future, evaluation_edges)
                    solver_edge_metrics = trajectory_metrics(prediction, pseudo_gt, future, edges)
                    density_outputs[(graph_type, query_count, condition, seed_index)] = prediction
                    records.append({
                        "gate": gate,
                        "trajectory_source": trajectory_source,
                        "bundle": str(bundle_path),
                        "source_video": source_video,
                        "motion_category": motion_category,
                        "query_count": query_count,
                        "graph_type": graph_type,
                        "condition": condition,
                        "seed_index": seed_index,
                        "num_edges": len(edges),
                        "evaluation_graph": "union_2d_3d_knn_k3",
                        "evaluation_edge_count": len(evaluation_edges),
                        "num_components": diagnostics["components"],
                        "kkt_condition_number": diagnostics["condition_number"],
                        **graph_stats,
                        **metrics,
                        **{f"solver_edge_{key}": value for key, value in solver_edge_metrics.items()},
                    })

    subset_records = []
    for graph_type in ("knn_2d", "knn_3d"):
        for condition in CONDITIONS:
            seed_count = 1 if condition == "oracle" else 20
            for seed_index in range(seed_count):
                base = density_outputs.get((graph_type, 32, condition, seed_index))
                if base is None:
                    continue
                for query_count in (48, 64):
                    other = density_outputs.get((graph_type, query_count, condition, seed_index))
                    if other is None:
                        continue
                    common_error = torch.linalg.vector_norm(base - other[:, :32], dim=-1).mean().item()
                    common_delta = torch.linalg.vector_norm(base - other[:, :32], dim=-1)
                    subset_records.append({
                        "gate": gate,
                        "trajectory_source": trajectory_source,
                        "bundle": str(bundle_path),
                        "source_video": source_video,
                        "motion_category": motion_category,
                        "graph_type": graph_type,
                        "condition": condition,
                        "seed_index": seed_index,
                        "comparison": f"32_vs_{query_count}",
                        "common_query_prediction_mean_distance": common_error,
                        "common_query_prediction_max_distance": float(common_delta.max()),
                        "common_query_prediction_distance_variance": float(common_delta.var(unbiased=False)),
                    })
    return {"bundle": str(bundle_path), "ready": True, "gate": gate, "trajectory_source": trajectory_source, "alignment": alignment, "records": records, "query_subset_records": subset_records}


def summarize_records(records: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for item in records:
        key = (item["source_video"], item["motion_category"], item["graph_type"], item["query_count"], item["condition"])
        grouped[key].append(item)
    summaries = []
    for key, items in sorted(grouped.items()):
        source_video, motion_category, graph_type, query_count, condition = key
        summaries.append({
            "source_video": source_video,
            "motion_category": motion_category,
            "graph_type": graph_type,
            "query_count": query_count,
            "condition": condition,
            "seed_count": len(items),
            "temporal_rve_auc": quantile_summary([item["temporal_rve_auc"] for item in items]),
            "auc_recovery_fraction": quantile_summary([item["auc_recovery_fraction"] for item in items]),
            "ade": quantile_summary([item["ade"] for item in items]),
            "fde": quantile_summary([item["fde"] for item in items]),
            "failure_fraction": float(np.mean([item["failed"] for item in items])),
            "worst_temporal_rve_auc": float(max(item["temporal_rve_auc"] for item in items)),
            "worst_auc_recovery_fraction": float(min(item["auc_recovery_fraction"] for item in items)),
            "num_components": int(max(item["num_components"] for item in items)),
            "bridge_edge_ratio": float(max(item["bridge_edge_ratio"] for item in items)),
            "global_lambda2": float(min(item["global_lambda2"] for item in items)),
            "kkt_condition_number": float(max(item["kkt_condition_number"] for item in items)),
            "history_graph_jaccard_min": float(min(item["history_graph_jaccard_min"] for item in items)),
        })
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-seed", type=int, default=20260922)
    parser.add_argument("--gate", choices=("e2-g", "e2-m"), default="e2-g")
    parser.add_argument("--trajectory-source", choices=("controlled_perturbation_baseline", "trackcraft3r_pseudo_gt", "molmomotion_prediction"), default="controlled_perturbation_baseline")
    args = parser.parse_args()
    bundles = sorted(args.bundle_dir.glob("*.npz"))
    results = [run_bundle(path, args.base_seed, args.output.parent, args.gate, args.trajectory_source) for path in bundles]
    ready = [result for result in results if result.get("ready")]
    all_records = [record for result in ready for record in result["records"]]
    payload = {
        "protocol": {
            "gate": args.gate,
            "trajectory_source": args.trajectory_source,
            "interpretation_boundary": (
                "graph/integrator auxiliary experiment; does not validate MolmoMotion or Relation Head dense prediction"
                if args.gate == "e2-g" else
                "model dense-interface experiment; requires validated variable-query checkpoint or adapter"
            ),
            "target_fps": TARGET_FPS,
            "query_counts": QUERY_COUNTS,
            "main_graph": "knn_2d_k3",
            "comparison_graph": "knn_3d_k3",
            "timestamp_tolerance_s": 0.5 / TARGET_FPS,
            "conditions": CONDITIONS,
            "non_oracle_seeds": 20,
        },
        "bundle_count": len(bundles),
        "ready_bundle_count": len(ready),
        "formal_data_gate_passed": len(ready) >= 5,
        "per_video_summaries": summarize_records(all_records),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("bundle_count", "ready_bundle_count", "formal_data_gate_passed")}, indent=2))


if __name__ == "__main__":
    main()
