#!/usr/bin/env python3
"""Lightweight diagnostics for E2-G.1 repaired 3D candidate graphs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from surface_relation_e2_lite import connected_components, incidence_matrix, solve_component_gauge, symmetric_knn_edges


def components(num_nodes: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    return connected_components(num_nodes, edges)


def repair_edges(points_2d: np.ndarray, points_3d: np.ndarray, k: int = 3) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    edges_2d = symmetric_knn_edges(points_2d, k)
    edges_3d = symmetric_knn_edges(points_3d, k)
    comp_id = {}
    for index, comp in enumerate(components(len(points_3d), edges_3d)):
        for node in comp:
            comp_id[node] = index
    parent = list(range(len(set(comp_id.values()))))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        a, b = find(a), find(b)
        if a == b:
            return False
        parent[b] = a
        return True

    crossing = []
    for source, target in edges_2d:
        source_comp, target_comp = comp_id[source], comp_id[target]
        if source_comp != target_comp:
            length = float(np.linalg.norm(points_2d[source] - points_2d[target]))
            crossing.append((length, source, target, source_comp, target_comp))
    crossing.sort(key=lambda item: (item[0], item[1], item[2]))
    added = []
    for _, source, target, source_comp, target_comp in crossing:
        if union(source_comp, target_comp):
            added.append((source, target))
        if len(added) >= len(parent) - 1:
            break
    return sorted(set(edges_3d) | set(added)), added


def centered_error(prediction: torch.Tensor, target: torch.Tensor, edges: list[tuple[int, int]]) -> float:
    values = []
    for comp in components(prediction.shape[1], edges):
        index = torch.tensor(comp, dtype=torch.long)
        values.append(torch.linalg.vector_norm(
            (prediction[:, index] - prediction[:, index].mean(dim=1, keepdim=True))
            - (target[:, index] - target[:, index].mean(dim=1, keepdim=True)), dim=-1
        ))
    return float(torch.cat(values, dim=1).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        required=True,
        help="Directory containing <source_video>_query_manifest.json files.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-seed", type=int, default=20260922)
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()

    records = []
    for bundle_path in sorted(args.bundle_dir.glob("*.npz")):
        with np.load(bundle_path, allow_pickle=False) as loaded:
            history = np.asarray(loaded["points_3d_history"], dtype=np.float64)[:, :48]
            points_2d = np.asarray(loaded["points_2d_at_t0"], dtype=np.float64)[:48]
            target = np.asarray(loaded["pseudo_gt"], dtype=np.float64)[:, :48]
            future = np.asarray(loaded["future_3d"], dtype=np.float64)[:, :48]
            source_video = str(loaded["source_video"].item())
            motion_category = str(loaded["motion_category"].item())
        manifest_path = args.manifest_dir / f"{source_video}_query_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing query manifest for {source_video}: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "bbox_xyxy" not in manifest:
            raise KeyError(f"Manifest has no bbox_xyxy: {manifest_path}")
        bbox = np.asarray(manifest["bbox_xyxy"], dtype=np.float64)
        if bbox.shape != (4,) or not np.all(np.isfinite(bbox)):
            raise ValueError(f"Invalid bbox_xyxy in {manifest_path}: {manifest['bbox_xyxy']!r}")
        bbox_diag = float(np.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]))

        repaired, added = repair_edges(points_2d, history[-1], 3)
        local_3d_lengths = np.asarray(
            [np.linalg.norm(history[-1, j] - history[-1, i]) for i, j in symmetric_knn_edges(history[-1], 3)],
            dtype=np.float64,
        )
        connection_rows = []
        for source, target_node in added:
            length_2d = float(np.linalg.norm(points_2d[target_node] - points_2d[source]))
            length_3d = float(np.linalg.norm(history[-1, target_node] - history[-1, source]))
            connection_rows.append({
                "source": source,
                "target": target_node,
                "length_2d_px": length_2d,
                "length_2d_over_bbox_diag": length_2d / max(bbox_diag, 1e-8),
                "length_3d": length_3d,
                "length_3d_over_local_3d_median": length_3d / max(float(np.median(local_3d_lengths)), 1e-8),
            })

        incidence = incidence_matrix(48, repaired)
        oracle_delta = torch.einsum(
            "en,tnd->ted",
            incidence,
            torch.tensor(target.tolist(), dtype=torch.float64) - torch.tensor(future.tolist(), dtype=torch.float64),
        )
        edge_scale = float(torch.median(torch.linalg.vector_norm(
            torch.einsum("en,tnd->ted", incidence, torch.tensor(target.tolist(), dtype=torch.float64)), dim=-1
        )))
        selected_edge_indices = [repaired.index(edge) for edge in added]
        anomaly_errors = []
        for seed_index in range(args.samples):
            seed = int.from_bytes(hashlib.sha256(f"{source_video}:{seed_index}".encode()).digest()[:8], "little")
            rng = np.random.default_rng(seed + args.base_seed)
            proposed = oracle_delta.clone()
            if selected_edge_indices:
                anomaly = rng.normal(0.0, 0.5 * edge_scale, size=(9, len(selected_edge_indices), 3))
                proposed[:, selected_edge_indices, :] += torch.tensor(anomaly.tolist(), dtype=torch.float64)
            correction, diagnostics = solve_component_gauge(proposed, repaired, 48)
            prediction = torch.tensor(future.tolist(), dtype=torch.float64) + correction
            anomaly_errors.append({
                "seed_index": seed_index,
                "component_centered_point_error": centered_error(prediction, torch.tensor(target.tolist(), dtype=torch.float64), repaired),
                "kkt_condition_number": diagnostics["condition_number"],
            })

        records.append({
            "source_video": source_video,
            "motion_category": motion_category,
            "query_count": 48,
            "bbox_xyxy": bbox.tolist(),
            "bbox_diagonal_px": bbox_diag,
            "connection_edges": connection_rows,
            "connection_edge_count": len(connection_rows),
            "local_3d_edge_length_distribution": {
                "count": int(len(local_3d_lengths)),
                "median": float(np.median(local_3d_lengths)),
                "q25": float(np.quantile(local_3d_lengths, 0.25)),
                "q75": float(np.quantile(local_3d_lengths, 0.75)),
                "minimum": float(np.min(local_3d_lengths)),
                "maximum": float(np.max(local_3d_lengths)),
            },
            "connection_edge_anomaly": {
                "samples": args.samples,
                "component_centered_error_median": float(np.median([row["component_centered_point_error"] for row in anomaly_errors])),
                "component_centered_error_q25": float(np.quantile([row["component_centered_point_error"] for row in anomaly_errors], 0.25)),
                "component_centered_error_q75": float(np.quantile([row["component_centered_point_error"] for row in anomaly_errors], 0.75)),
                "component_centered_error_max": float(np.max([row["component_centered_point_error"] for row in anomaly_errors])),
                "rows": anomaly_errors,
            },
        })

    payload = {
        "experiment_gate": "E2-G.1-diagnostic",
        "interpretation_boundary": "candidate graph diagnostic only; no MolmoMotion or Relation Head claim",
        "records": records,
        "candidate_hierarchy": {
            "primary_candidate": "2d_symmetric_knn_k3",
            "secondary_candidate": "repaired_observed_3d_symmetric_knn_k3",
            "numeric_control": "observed_3d_symmetric_knn_k3",
            "formal_default_graph": "unfrozen",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"video_count": len(records), "connection_edge_counts": [item["connection_edge_count"] for item in records]}, indent=2))


if __name__ == "__main__":
    main()
