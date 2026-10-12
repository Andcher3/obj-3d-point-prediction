#!/usr/bin/env python3
"""Summarize the restricted E2-G.1 connectivity-repair experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CONDITIONS = (
    "gaussian_0.01", "gaussian_0.05", "gaussian_0.10", "gaussian_0.20",
    "outlier_0.05", "outlier_0.10", "outlier_0.20",
)
GRAPHS = ("knn_2d", "knn_3d", "knn_3d_repaired")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.results.read_text(encoding="utf-8"))
    summaries = payload["summaries"]
    videos = sorted({item["source_video"] for item in summaries})

    def get(video: str, graph: str, condition: str, query_count: int = 48) -> dict:
        return next(item for item in summaries if item["source_video"] == video and item["graph_type"] == graph
                    and item["condition"] == condition and item["query_count"] == query_count)

    structure = []
    for video in videos:
        item = get(video, "knn_3d_repaired", "oracle")
        structure.append({
            "source_video": video,
            "motion_category": item["motion_category"],
            "query_count": 48,
            "repaired_components": item["num_components"],
            "added_connection_edge_count": item["added_connection_edge_count"],
            "added_connection_edge_length_sum_px": item["added_connection_edge_length_sum_px"],
            "added_connection_edge_length_max_px": item["added_connection_edge_length_max_px"],
            "added_connection_bridge_count": item["added_connection_bridge_count"],
            "repaired_history_graph_jaccard_mean": item["repaired_history_graph_jaccard_mean"],
            "repaired_history_graph_jaccard_min": item["repaired_history_graph_jaccard_min"],
        })

    performance = []
    for condition in CONDITIONS:
        row = {"condition": condition}
        for graph in GRAPHS:
            native = [get(video, graph, condition)["graph_native_rve_auc"]["median"] for video in videos]
            shared = [get(video, graph, condition)["shared_union_rve_auc"]["median"] for video in videos]
            centered = [get(video, graph, condition)["component_centered_point_error"]["median"] for video in videos]
            row[graph] = {
                "graph_native_rve_auc_macro_mean": float(np.mean(native)),
                "shared_union_rve_auc_macro_mean": float(np.mean(shared)),
                "component_centered_point_error_macro_mean": float(np.mean(centered)),
                "video_values_graph_native": native,
                "video_values_shared_union": shared,
            }
        row["repaired_over_original_3d_shared_union_ratio"] = (
            row["knn_3d_repaired"]["shared_union_rve_auc_macro_mean"]
            / row["knn_3d"]["shared_union_rve_auc_macro_mean"]
        )
        row["repaired_over_2d_shared_union_ratio"] = (
            row["knn_3d_repaired"]["shared_union_rve_auc_macro_mean"]
            / row["knn_2d"]["shared_union_rve_auc_macro_mean"]
        )
        performance.append(row)

    density = []
    for graph in GRAPHS:
        for condition in CONDITIONS:
            ratios = []
            for video in videos:
                value_48 = get(video, graph, condition, 48)["shared_union_rve_auc"]["median"]
                value_64 = get(video, graph, condition, 64)["shared_union_rve_auc"]["median"]
                ratios.append(value_64 / value_48 if value_48 > 0 else float("nan"))
            density.append({
                "graph_type": graph,
                "condition": condition,
                "ratio_64_over_48_median": float(np.nanmedian(ratios)),
                "increase_count": int(sum(value > 1.0 for value in ratios)),
                "comparison_count": len(ratios),
                "video_values": ratios,
            })

    all_repaired_edges = [item["added_connection_edge_count"] for item in structure]
    payload_out = {
        "experiment_gate": "E2-G.1",
        "interpretation_boundary": "TrackCraft3R pseudo-GT plus controlled perturbation; no MolmoMotion or Relation Head prediction claim",
        "source_video_count": len(videos),
        "motion_categories": sorted({item["motion_category"] for item in structure}),
        "category_limitation": "rigid and articulated only; no reliable deformable video",
        "repair_rule": "3D symmetric kNN k=3 components connected by shortest observed-2D symmetric kNN k=3 crossing edges, deterministic Kruskal tie-break",
        "structure_query48": structure,
        "performance_query48": performance,
        "density_48_to_64": density,
        "repair_summary": {
            "videos_requiring_added_edges": int(sum(count > 0 for count in all_repaired_edges)),
            "videos_already_connected": int(sum(count == 0 for count in all_repaired_edges)),
            "added_edge_count_total": int(sum(all_repaired_edges)),
            "added_edge_count_values": all_repaired_edges,
            "all_repaired_graphs_connected": all(item["repaired_components"] == 1 for item in structure),
            "all_added_edges_are_bridges": all(item["added_connection_bridge_count"] == item["added_connection_edge_count"] for item in structure),
        },
        "decision": {
            "formal_default_graph": "unfrozen",
            "retain_2d_k3_as_engineering_candidate": True,
            "retain_3d_k3_as_numeric_control": True,
            "switch_to_3d": False,
            "e2_m": "blocked",
            "e3": "paused",
            "e4": "paused",
            "irls_huber_learned_gate": "paused",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "videos": len(videos),
        "added_edge_count_values": all_repaired_edges,
        "all_repaired_graphs_connected": payload_out["repair_summary"]["all_repaired_graphs_connected"],
    }, indent=2))


if __name__ == "__main__":
    main()
