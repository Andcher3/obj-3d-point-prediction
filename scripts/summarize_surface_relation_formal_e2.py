#!/usr/bin/env python3
"""Summarize formal E2-G data and graph-retention gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


NON_ORACLE = (
    "gaussian_0.01", "gaussian_0.05", "gaussian_0.10", "gaussian_0.20",
    "outlier_0.05", "outlier_0.10", "outlier_0.20",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--results-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    results = json.loads(args.results_json.read_text(encoding="utf-8"))
    summaries = results["per_video_summaries"]
    videos = sorted({item["source_video"] for item in summaries})

    def row(video: str, graph: str, query_count: int, condition: str) -> dict:
        return next(item for item in summaries if item["source_video"] == video and item["graph_type"] == graph
                    and item["query_count"] == query_count and item["condition"] == condition)

    graph_structure = []
    for video in videos:
        for graph in ("knn_2d", "knn_3d"):
            item = row(video, graph, 48, "oracle")
            graph_structure.append({
                "source_video": video,
                "motion_category": item["motion_category"],
                "graph_type": graph,
                "num_components": item["num_components"],
                "bridge_edge_ratio": item["bridge_edge_ratio"],
                "global_lambda2": item["global_lambda2"],
                "kkt_condition_number": item["kkt_condition_number"],
                "history_graph_jaccard_min": item["history_graph_jaccard_min"],
            })

    performance = []
    all_2d, all_3d = [], []
    for condition in NON_ORACLE:
        values = {graph: [] for graph in ("knn_2d", "knn_3d")}
        wins = 0
        for video in videos:
            value_2d = row(video, "knn_2d", 48, condition)["temporal_rve_auc"]["median"]
            value_3d = row(video, "knn_3d", 48, condition)["temporal_rve_auc"]["median"]
            values["knn_2d"].append(value_2d)
            values["knn_3d"].append(value_3d)
            all_2d.append(value_2d)
            all_3d.append(value_3d)
            wins += value_2d <= value_3d
        mean_2d = float(np.mean(values["knn_2d"]))
        mean_3d = float(np.mean(values["knn_3d"]))
        performance.append({
            "condition": condition,
            "macro_mean_2d": mean_2d,
            "macro_mean_3d": mean_3d,
            "ratio_2d_over_3d": mean_2d / mean_3d,
            "videos_2d_not_worse": wins,
            "video_count": len(videos),
        })

    density_ratios = []
    density_increase_count = 0
    for video in videos:
        for condition in NON_ORACLE:
            value_32 = row(video, "knn_2d", 32, condition)["temporal_rve_auc"]["median"]
            value_48 = row(video, "knn_2d", 48, condition)["temporal_rve_auc"]["median"]
            ratio = value_48 / value_32
            density_ratios.append(ratio)
            density_increase_count += int(ratio > 1.0)

    match_records = []
    for bundle_path in sorted(args.bundle_dir.glob("*.npz")):
        with np.load(bundle_path, allow_pickle=False) as bundle:
            errors = np.asarray(bundle["query_t0_match_error_px"], dtype=np.float64)
            match_records.append({
                "source_video": str(bundle["source_video"].item()),
                "max_px": float(errors.max()),
                "median_px": float(np.median(errors)),
            })

    structure_2d = [item for item in graph_structure if item["graph_type"] == "knn_2d"]
    structure_3d = [item for item in graph_structure if item["graph_type"] == "knn_3d"]
    overall_ratio = float(np.mean(all_2d) / np.mean(all_3d))
    video_mean_wins = 0
    for video in videos:
        mean_2d = np.mean([row(video, "knn_2d", 48, condition)["temporal_rve_auc"]["median"] for condition in NON_ORACLE])
        mean_3d = np.mean([row(video, "knn_3d", 48, condition)["temporal_rve_auc"]["median"] for condition in NON_ORACLE])
        video_mean_wins += int(mean_2d <= mean_3d)

    gates = {
        "formal_data_gate_passed": bool(audit["formal_e2_ready"] and results["formal_data_gate_passed"]),
        "two_d_connected_on_all_videos": all(item["num_components"] == 1 for item in structure_2d),
        "two_d_no_bridge_edges_on_all_videos": all(item["bridge_edge_ratio"] == 0.0 for item in structure_2d),
        "two_d_macro_not_worse_than_3d_by_more_than_5_percent": overall_ratio <= 1.05,
        "two_d_not_worse_on_majority_of_videos": video_mean_wins >= (len(videos) // 2 + 1),
        "query_32_to_48_no_systematic_error_increase": density_increase_count < len(density_ratios) / 2,
    }
    gates["retain_2d_k3_as_default"] = all(gates[key] for key in (
        "two_d_connected_on_all_videos",
        "two_d_macro_not_worse_than_3d_by_more_than_5_percent",
        "two_d_not_worse_on_majority_of_videos",
        "query_32_to_48_no_systematic_error_increase",
    ))

    payload = {
        "experiment_gate": "E2-G",
        "interpretation_boundary": "TrackCraft3R pseudo-GT plus controlled perturbation; no MolmoMotion dense prediction claim",
        "source_video_count": len(videos),
        "motion_categories": sorted({item["motion_category"] for item in graph_structure}),
        "category_limitation": "rigid and articulated only; no reliable deformable video",
        "t0_track_matching": match_records,
        "graph_structure_query48": graph_structure,
        "condition_performance_query48": performance,
        "overall_macro_ratio_2d_over_3d": overall_ratio,
        "videos_2d_not_worse_by_condition_mean": video_mean_wins,
        "three_d_disconnected_video_count": sum(item["num_components"] > 1 for item in structure_3d),
        "density_32_to_48_ratio_median": float(np.median(density_ratios)),
        "density_32_to_48_increase_count": density_increase_count,
        "density_32_to_48_comparison_count": len(density_ratios),
        "gates": gates,
        "e2_m_status": "blocked_fixed_8_query_molmomotion_interface",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"gates": gates, "overall_macro_ratio_2d_over_3d": overall_ratio}, indent=2))


if __name__ == "__main__":
    main()
