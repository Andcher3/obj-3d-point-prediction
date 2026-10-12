#!/usr/bin/env python3
"""Restricted E2-G.1: deterministically repair observed-3D kNN connectivity."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from surface_relation_e2_lite import (
    bridge_edges,
    connected_components,
    graph_diagnostics,
    incidence_matrix,
    noisy_relation,
    project_points,
    quantile_summary,
    solve_component_gauge,
    symmetric_knn_edges,
    trajectory_metrics,
)


QUERY_COUNTS = (32, 48, 64)
CONDITIONS = (
    "oracle", "gaussian_0.01", "gaussian_0.05", "gaussian_0.10", "gaussian_0.20",
    "outlier_0.05", "outlier_0.10", "outlier_0.20",
)


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[rb] = ra
        return True


def repair_edges(points_2d: np.ndarray, points_3d: np.ndarray, k: int = 3) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    edges_2d = symmetric_knn_edges(points_2d, k)
    edges_3d = symmetric_knn_edges(points_3d, k)
    components = connected_components(len(points_3d), edges_3d)
    component_id = {}
    for index, component in enumerate(components):
        for node in component:
            component_id[node] = index
    uf = UnionFind(len(components))
    # Kruskal over only 2D edges crossing distinct observed-3D components.
    candidates = []
    for source, target in edges_2d:
        cs, ct = component_id[source], component_id[target]
        if cs != ct:
            length = float(np.linalg.norm(points_2d[source] - points_2d[target]))
            candidates.append((length, source, target, cs, ct))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    added = []
    for length, source, target, cs, ct in candidates:
        if uf.union(cs, ct):
            added.append((source, target))
        if len(added) == len(components) - 1:
            break
    repaired = sorted(set(edges_3d) | set(added))
    return repaired, added


def component_centered_error(prediction: torch.Tensor, target: torch.Tensor, edges: list[tuple[int, int]]) -> float:
    components = connected_components(prediction.shape[1], edges)
    values = []
    for component in components:
        index = torch.tensor(component, dtype=torch.long)
        pred_centered = prediction[:, index] - prediction[:, index].mean(dim=1, keepdim=True)
        target_centered = target[:, index] - target[:, index].mean(dim=1, keepdim=True)
        values.append(torch.linalg.vector_norm(pred_centered - target_centered, dim=-1))
    return float(torch.cat(values, dim=1).mean()) if values else 0.0


def topology_jaccard(reference: set[tuple[int, int]], current: set[tuple[int, int]]) -> float:
    union = reference | current
    return len(reference & current) / len(union) if union else 1.0


def graph_info(graph_type: str, edges: list[tuple[int, int]], history_3d: np.ndarray, history_2d: np.ndarray, query_count: int) -> dict:
    if graph_type == "knn_2d":
        return graph_diagnostics(edges, history_3d, history_2d, "knn_2d", 3)
    if graph_type == "knn_3d":
        return graph_diagnostics(edges, history_3d, history_2d, "knn_3d", 3)
    components = connected_components(query_count, edges)
    bridges = bridge_edges(query_count, edges)
    eigenvalues = np.linalg.eigvalsh(
        np.diag(np.sum([[1.0 if (i, j) in edges or (j, i) in edges else 0.0 for j in range(query_count)] for i in range(query_count)], axis=1))
        - np.asarray([[1.0 if (i, j) in edges or (j, i) in edges else 0.0 for j in range(query_count)] for i in range(query_count)])
    )
    positive = eigenvalues[eigenvalues > 1e-9]
    lengths = np.asarray([np.linalg.norm(history_3d[:, j] - history_3d[:, i], axis=-1) for i, j in edges]) if edges else np.empty((0, 3))
    edge_mean = np.maximum(lengths.mean(axis=1), 1e-8) if len(lengths) else np.asarray([1.0])
    return {
        "num_nodes": query_count,
        "num_edges": len(edges),
        "num_components": len(components),
        "largest_component_coverage": max(map(len, components)) / query_count,
        "bridge_edge_count": len(bridges),
        "bridge_edge_ratio": len(bridges) / len(edges) if edges else 1.0,
        "global_lambda2": float(eigenvalues[1]) if query_count > 1 else 0.0,
        "spectral_condition_number": float(positive[-1] / positive[0]) if len(positive) else 1.0,
        "kkt_condition_number": float(np.linalg.cond(np.eye(query_count))),
        "history_graph_jaccard_mean": 0.0,
        "history_graph_jaccard_min": 0.0,
        "history_edge_length_cv_median": float(np.median(lengths.std(axis=1) / edge_mean)) if len(lengths) else float("inf"),
        "history_edge_length_range_median": float(np.median((lengths.max(axis=1) - lengths.min(axis=1)) / edge_mean)) if len(lengths) else float("inf"),
    }


def load_bundle(path: Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str]:
    with np.load(path, allow_pickle=False) as loaded:
        bundle = {key: loaded[key] for key in loaded.files}
    future = np.asarray(bundle["future_3d"], dtype=np.float64)
    target = np.asarray(bundle["pseudo_gt"], dtype=np.float64)
    history = np.asarray(bundle["points_3d_history"], dtype=np.float64)
    points_2d = np.asarray(bundle["points_2d_at_t0"], dtype=np.float64)
    history_2d = project_points(history.copy(), np.asarray(bundle["intrinsics"], dtype=np.float64))
    source_video = str(bundle["source_video"].item())
    motion_category = str(bundle["motion_category"].item())
    return bundle, future, target, history, points_2d, history_2d, source_video, motion_category


def run_bundle(path: Path, base_seed: int) -> dict:
    bundle, future_all, target_all, history_all, points_2d_all, history_2d_all, source_video, motion_category = load_bundle(path)
    records = []
    density_records = []
    for query_count in QUERY_COUNTS:
        future = torch.tensor(future_all[:, :query_count].tolist(), dtype=torch.float64)
        target = torch.tensor(target_all[:, :query_count].tolist(), dtype=torch.float64)
        history = history_all[:, :query_count]
        history_2d = history_2d_all[:, :query_count]
        points_2d = points_2d_all[:query_count]
        edges_2d = symmetric_knn_edges(points_2d, 3)
        edges_3d = symmetric_knn_edges(history[-1], 3)
        repaired, added = repair_edges(points_2d, history[-1], 3)
        edge_map = {"knn_2d": edges_2d, "knn_3d": edges_3d, "knn_3d_repaired": repaired}
        shared_edges = sorted(set(edges_2d) | set(edges_3d))
        history_repairs = [repair_edges(history_2d[t], history[t], 3)[0] for t in range(3)]
        repaired_jaccard = [topology_jaccard(set(repaired), set(current)) for current in history_repairs]
        added_lengths = [float(np.linalg.norm(points_2d[j] - points_2d[i])) for i, j in added]
        target_rel = {}
        for graph_type, edges in edge_map.items():
            if not edges:
                continue
            incidence = incidence_matrix(query_count, edges)
            target_rel[graph_type] = float(torch.median(torch.linalg.vector_norm(torch.einsum("en,tnd->ted", incidence, target), dim=-1)))
            for condition_index, condition in enumerate(CONDITIONS):
                seed_count = 1 if condition == "oracle" else 20
                for seed_index in range(seed_count):
                    rng = np.random.default_rng(np.random.SeedSequence([base_seed, query_count, condition_index, seed_index]))
                    oracle_delta = torch.einsum("en,tnd->ted", incidence, target - future)
                    proposed = noisy_relation(oracle_delta, condition, target_rel[graph_type], rng)
                    correction, diagnostics = solve_component_gauge(proposed, edges, query_count)
                    prediction = future + correction
                    native = trajectory_metrics(prediction, target, future, edges)
                    shared = trajectory_metrics(prediction, target, future, shared_edges)
                    records.append({
                        "source_video": source_video,
                        "motion_category": motion_category,
                        "query_count": query_count,
                        "graph_type": graph_type,
                        "condition": condition,
                        "seed_index": seed_index,
                        "num_edges": len(edges),
                        "num_components": len(connected_components(query_count, edges)),
                        "added_connection_edge_count": len(added) if graph_type == "knn_3d_repaired" else 0,
                        "added_connection_edge_length_sum_px": float(sum(added_lengths)) if graph_type == "knn_3d_repaired" else 0.0,
                        "added_connection_edge_length_max_px": float(max(added_lengths)) if added_lengths and graph_type == "knn_3d_repaired" else 0.0,
                        "added_connection_bridge_count": len(bridge_edges(query_count, edges)) if graph_type == "knn_3d_repaired" else None,
                        "repaired_history_graph_jaccard_mean": float(np.mean(repaired_jaccard)) if graph_type == "knn_3d_repaired" else None,
                        "repaired_history_graph_jaccard_min": float(np.min(repaired_jaccard)) if graph_type == "knn_3d_repaired" else None,
                        "component_centered_point_error": component_centered_error(prediction, target, edges),
                        "shared_union_component_centered_point_error": component_centered_error(prediction, target, shared_edges),
                        "kkt_condition_number": diagnostics["condition_number"],
                        "graph_native": native,
                        "shared_union": shared,
                    })
        # 48 -> 64 is measured after the same seed-matched correction.
        if query_count == 64:
            density_records.append({"source_video": source_video, "motion_category": motion_category, "query_count": query_count, "added_connection_edge_count": len(added), "added_connection_edge_length_sum_px": float(sum(added_lengths))})
    return {"bundle": str(path), "source_video": source_video, "motion_category": motion_category, "records": records, "density_records": density_records}


def summarize(records: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["source_video"], record["motion_category"], record["graph_type"], record["query_count"], record["condition"])].append(record)
    out = []
    for key, items in sorted(grouped.items()):
        video, category, graph, query, condition = key
        out.append({
            "source_video": video, "motion_category": category, "graph_type": graph, "query_count": query, "condition": condition,
            "seed_count": len(items),
            "graph_native_rve_auc": quantile_summary([x["graph_native"]["temporal_rve_auc"] for x in items]),
            "shared_union_rve_auc": quantile_summary([x["shared_union"]["temporal_rve_auc"] for x in items]),
            "component_centered_point_error": quantile_summary([x["component_centered_point_error"] for x in items]),
            "failure_fraction": float(np.mean([x["graph_native"]["failed"] for x in items])),
            "num_components": max(x["num_components"] for x in items),
            "added_connection_edge_count": max(x["added_connection_edge_count"] for x in items),
            "added_connection_edge_length_sum_px": max(x["added_connection_edge_length_sum_px"] for x in items),
            "added_connection_edge_length_max_px": max(x["added_connection_edge_length_max_px"] for x in items),
            "added_connection_bridge_count": max(x["added_connection_bridge_count"] or 0 for x in items),
            "repaired_history_graph_jaccard_mean": next((x["repaired_history_graph_jaccard_mean"] for x in items if x["repaired_history_graph_jaccard_mean"] is not None), None),
            "repaired_history_graph_jaccard_min": next((x["repaired_history_graph_jaccard_min"] for x in items if x["repaired_history_graph_jaccard_min"] is not None), None),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-seed", type=int, default=20260922)
    args = parser.parse_args()
    results = [run_bundle(path, args.base_seed) for path in sorted(args.bundle_dir.glob("*.npz"))]
    records = [record for result in results for record in result["records"]]
    payload = {
        "experiment_gate": "E2-G.1",
        "protocol": {
            "query_counts": QUERY_COUNTS,
            "conditions": CONDITIONS,
            "non_oracle_seeds": 20,
            "repair_rule": "observed_3d_symmetric_knn_k3 components connected by shortest observed_2d_symmetric_knn_k3 crossing edges under deterministic Kruskal tie-break",
            "interpretation_boundary": "graph/integrator auxiliary experiment; no MolmoMotion or Relation Head claim",
        },
        "bundle_count": len(results),
        "source_video_count": len({result["source_video"] for result in results}),
        "results": results,
        "summaries": summarize(records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"bundle_count": payload["bundle_count"], "source_video_count": payload["source_video_count"], "record_count": len(records)}, indent=2))


if __name__ == "__main__":
    main()
