#!/usr/bin/env python3
"""E2-lite graph comparison for the surface-relation feasibility study."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from surface_relation_feasibility import (
    connected_components,
    incidence_matrix,
    infer_video_group,
    solve_component_gauge,
)


GRAPH_TYPES = ("knn_2d", "knn_3d", "hybrid_lite")
NOISE_CONDITIONS = (
    "oracle",
    "gaussian_0.01",
    "gaussian_0.05",
    "gaussian_0.10",
    "gaussian_0.20",
    "outlier_0.05",
    "outlier_0.10",
    "outlier_0.20",
)


def array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()[:16]


def project_points(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    z = np.clip(points[..., 2:3], 1e-8, None)
    projected = points[..., :2] / z
    projected[..., 0] = projected[..., 0] * intrinsics[0, 0] + intrinsics[0, 2]
    projected[..., 1] = projected[..., 1] * intrinsics[1, 1] + intrinsics[1, 2]
    return projected


def symmetric_knn_edges(points: np.ndarray, k: int) -> list[tuple[int, int]]:
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    edges: set[tuple[int, int]] = set()
    for source in range(len(points)):
        for target in np.argsort(distances[source])[: min(k, len(points) - 1)]:
            edges.add(tuple(sorted((source, int(target)))))
    return sorted(edges)


def hybrid_edges(points_2d: np.ndarray, points_3d: np.ndarray, k: int) -> list[tuple[int, int]]:
    num_nodes = len(points_2d)
    distances_2d = np.linalg.norm(points_2d[:, None, :] - points_2d[None, :, :], axis=-1)
    distances_3d = np.linalg.norm(points_3d[:, None, :] - points_3d[None, :, :], axis=-1)
    depth_jumps = np.abs(points_3d[:, None, 2] - points_3d[None, :, 2])
    np.fill_diagonal(distances_2d, np.inf)
    np.fill_diagonal(distances_3d, np.inf)
    np.fill_diagonal(depth_jumps, np.inf)

    candidate_k = min(num_nodes - 1, max(k + 2, 2 * k))
    candidate_pairs = []
    for source in range(num_nodes):
        for target in np.argsort(distances_2d[source])[:candidate_k]:
            candidate_pairs.append((source, int(target)))
    candidate_depth = np.asarray([depth_jumps[source, target] for source, target in candidate_pairs])
    finite_depth = candidate_depth[np.isfinite(candidate_depth)]
    median_depth = float(np.median(finite_depth)) if len(finite_depth) else 0.0
    mad_depth = float(np.median(np.abs(finite_depth - median_depth))) if len(finite_depth) else 0.0
    depth_threshold = median_depth + 2.5 * max(mad_depth, 1e-8)

    local_3d_threshold = np.empty(num_nodes, dtype=np.float64)
    for node in range(num_nodes):
        kth = np.sort(distances_3d[node])[: min(k, num_nodes - 1)]
        local_3d_threshold[node] = 1.5 * float(kth[-1])

    edges: set[tuple[int, int]] = set()
    for source in range(num_nodes):
        accepted = []
        scale_2d = max(float(np.min(distances_2d[source])), 1e-8)
        scale_3d = max(float(np.min(distances_3d[source])), 1e-8)
        for target in np.argsort(distances_2d[source])[:candidate_k]:
            target = int(target)
            if depth_jumps[source, target] > depth_threshold:
                continue
            if distances_3d[source, target] > max(local_3d_threshold[source], local_3d_threshold[target]):
                continue
            score = distances_2d[source, target] / scale_2d + distances_3d[source, target] / scale_3d
            accepted.append((float(score), target))
        for _, target in sorted(accepted)[:k]:
            edges.add(tuple(sorted((source, target))))
    return sorted(edges)


def build_graph(graph_type: str, points_2d: np.ndarray, points_3d: np.ndarray, k: int) -> list[tuple[int, int]]:
    if graph_type == "knn_2d":
        return symmetric_knn_edges(points_2d, k)
    if graph_type == "knn_3d":
        return symmetric_knn_edges(points_3d, k)
    if graph_type == "hybrid_lite":
        return hybrid_edges(points_2d, points_3d, k)
    raise ValueError(graph_type)


def bridge_edges(num_nodes: int, edges: list[tuple[int, int]]) -> set[tuple[int, int]]:
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(num_nodes)]
    for edge_index, (source, target) in enumerate(edges):
        adjacency[source].append((target, edge_index))
        adjacency[target].append((source, edge_index))
    discovery = [-1] * num_nodes
    low = [-1] * num_nodes
    bridges: set[tuple[int, int]] = set()
    time = 0

    def visit(node: int, parent_edge: int) -> None:
        nonlocal time
        discovery[node] = low[node] = time
        time += 1
        for neighbor, edge_index in adjacency[node]:
            if edge_index == parent_edge:
                continue
            if discovery[neighbor] == -1:
                visit(neighbor, edge_index)
                low[node] = min(low[node], low[neighbor])
                if low[neighbor] > discovery[node]:
                    bridges.add(tuple(sorted((node, neighbor))))
            else:
                low[node] = min(low[node], discovery[neighbor])

    for node in range(num_nodes):
        if discovery[node] == -1:
            visit(node, -1)
    return bridges


def laplacian_spectrum(num_nodes: int, edges: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for source, target in edges:
        adjacency[source, target] = 1.0
        adjacency[target, source] = 1.0
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    return np.linalg.eigvalsh(laplacian), laplacian


def kkt_condition_number(num_nodes: int, edges: list[tuple[int, int]]) -> float:
    if not edges:
        return 1.0
    incidence = np.asarray(incidence_matrix(num_nodes, edges).tolist(), dtype=np.float64)
    components = connected_components(num_nodes, edges)
    gauge = np.zeros((num_nodes, len(components)), dtype=np.float64)
    for component_index, component in enumerate(components):
        gauge[component, component_index] = 1.0
    laplacian = incidence.T @ incidence
    kkt = np.block([[laplacian, gauge], [gauge.T, np.zeros((len(components), len(components)))]] )
    return float(np.linalg.cond(kkt))


def graph_diagnostics(
    edges: list[tuple[int, int]],
    history_3d: np.ndarray,
    history_2d: np.ndarray,
    graph_type: str,
    k: int,
) -> dict:
    num_nodes = history_3d.shape[1]
    components = connected_components(num_nodes, edges)
    largest_component = max(components, key=len)
    bridges = bridge_edges(num_nodes, edges)
    eigenvalues, _ = laplacian_spectrum(num_nodes, edges)
    positive = eigenvalues[eigenvalues > 1e-9]
    global_lambda2 = float(eigenvalues[1]) if num_nodes > 1 else 0.0
    spectral_condition = float(positive[-1] / positive[0]) if len(positive) else 1.0

    largest_edges = [
        (largest_component.index(source), largest_component.index(target))
        for source, target in edges
        if source in largest_component and target in largest_component
    ]
    largest_eigenvalues, _ = laplacian_spectrum(len(largest_component), largest_edges)
    largest_lambda2 = float(largest_eigenvalues[1]) if len(largest_component) > 1 else 0.0

    history_graphs = [set(build_graph(graph_type, history_2d[t], history_3d[t], k)) for t in range(len(history_3d))]
    reference = set(edges)
    jaccards = []
    for historical in history_graphs:
        union = reference | historical
        jaccards.append(len(reference & historical) / len(union) if union else 1.0)

    if edges:
        edge_lengths = np.stack(
            [np.linalg.norm(history_3d[:, target] - history_3d[:, source], axis=-1) for source, target in edges],
            axis=1,
        )
        edge_means = np.maximum(edge_lengths.mean(axis=0), 1e-8)
        edge_cv = edge_lengths.std(axis=0) / edge_means
        normalized_range = (edge_lengths.max(axis=0) - edge_lengths.min(axis=0)) / edge_means
    else:
        edge_cv = np.asarray([np.inf])
        normalized_range = np.asarray([np.inf])
    return {
        "num_nodes": num_nodes,
        "num_edges": len(edges),
        "num_components": len(components),
        "largest_component_coverage": len(largest_component) / num_nodes,
        "bridge_edge_count": len(bridges),
        "bridge_edge_ratio": len(bridges) / len(edges) if edges else 1.0,
        "global_lambda2": global_lambda2,
        "largest_component_lambda2": largest_lambda2,
        "spectral_condition_number": spectral_condition,
        "kkt_condition_number": kkt_condition_number(num_nodes, edges),
        "history_graph_jaccard_mean": float(np.mean(jaccards)),
        "history_graph_jaccard_min": float(np.min(jaccards)),
        "history_edge_length_cv_median": float(np.median(edge_cv)),
        "history_edge_length_range_median": float(np.median(normalized_range)),
    }


def trajectory_metrics(
    prediction: torch.Tensor, target: torch.Tensor, baseline: torch.Tensor, edges: list[tuple[int, int]]
) -> dict:
    incidence = incidence_matrix(prediction.shape[1], edges)
    relative_prediction = torch.einsum("en,tnd->ted", incidence, prediction)
    relative_target = torch.einsum("en,tnd->ted", incidence, target)
    horizon_rve = np.asarray(
        torch.linalg.vector_norm(relative_prediction - relative_target, dim=-1).mean(dim=1).detach().cpu().tolist(),
        dtype=np.float64,
    )
    horizon = np.linspace(0.0, 1.0, len(horizon_rve))
    trapezoid = getattr(np, "trapezoid", np.trapz)
    temporal_auc = float(trapezoid(horizon_rve, x=horizon))
    baseline_relative = torch.einsum("en,tnd->ted", incidence, baseline)
    baseline_horizon_rve = np.asarray(
        torch.linalg.vector_norm(baseline_relative - relative_target, dim=-1).mean(dim=1).detach().cpu().tolist(),
        dtype=np.float64,
    )
    baseline_auc = float(trapezoid(baseline_horizon_rve, x=horizon))
    recovery = 1.0 - temporal_auc / baseline_auc if baseline_auc > 1e-12 else 0.0
    absolute_error = torch.linalg.vector_norm(prediction - target, dim=-1)
    return {
        "horizon_rve": horizon_rve.tolist(),
        "temporal_rve_auc": temporal_auc,
        "baseline_temporal_rve_auc": baseline_auc,
        "auc_recovery_fraction": recovery,
        "failed": recovery <= 0.0,
        "ade": float(absolute_error.mean()),
        "fde": float(absolute_error[-1].mean()),
        "common_translation_shift": float(
            torch.linalg.vector_norm(prediction.mean(dim=1) - baseline.mean(dim=1), dim=-1).max()
        ),
    }


def aggregate_runs_by_seed(runs: list[dict], group_keys: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for run in runs:
        key = tuple(str(run[name]) for name in (*group_keys, "seed_index"))
        grouped[key].append(run)
    aggregated = []
    for key, items in sorted(grouped.items()):
        mean_auc = float(np.mean([item["temporal_rve_auc"] for item in items]))
        mean_baseline_auc = float(np.mean([item["baseline_temporal_rve_auc"] for item in items]))
        recovery = 1.0 - mean_auc / mean_baseline_auc if mean_baseline_auc > 1e-12 else 0.0
        record = {name: value for name, value in zip((*group_keys, "seed_index"), key)}
        record.update(
            {
                "trajectory_config_count": len(items),
                "horizon_rve": np.mean([item["horizon_rve"] for item in items], axis=0).tolist(),
                "temporal_rve_auc": mean_auc,
                "baseline_temporal_rve_auc": mean_baseline_auc,
                "auc_recovery_fraction": recovery,
                "failed": recovery <= 0.0,
                "common_translation_shift": float(max(item["common_translation_shift"] for item in items)),
            }
        )
        aggregated.append(record)
    return aggregated


def noisy_relation(
    oracle_delta: torch.Tensor,
    condition: str,
    edge_scale: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    if condition == "oracle":
        return oracle_delta
    noisy = oracle_delta.clone()
    if condition.startswith("gaussian_"):
        fraction = float(condition.split("_")[1])
        sampled = rng.normal(0.0, fraction * edge_scale, size=tuple(noisy.shape))
        noisy += torch.tensor(sampled.tolist(), dtype=torch.float64)
        return noisy
    if condition.startswith("outlier_"):
        fraction = float(condition.split("_")[1])
        count = max(1, round(fraction * noisy.shape[1]))
        selected = rng.choice(noisy.shape[1], size=count, replace=False)
        sampled = rng.normal(0.0, 0.5 * edge_scale, size=(noisy.shape[0], count, 3))
        selected_indices = [int(index) for index in selected.tolist()]
        noisy[:, selected_indices, :] += torch.tensor(sampled.tolist(), dtype=torch.float64)
        return noisy
    raise ValueError(condition)


def quantile_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    q25, median, q75 = np.quantile(array, [0.25, 0.5, 0.75])
    return {
        "median": float(median),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def summarize_runs(runs: list[dict], group_keys: tuple[str, ...]) -> dict:
    grouped: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for run in runs:
        grouped[tuple(str(run[key]) for key in group_keys)].append(run)
    output = {}
    for group, items in sorted(grouped.items()):
        cursor = output
        for value in group[:-1]:
            cursor = cursor.setdefault(value, {})
        recoveries = [item["auc_recovery_fraction"] for item in items]
        aucs = [item["temporal_rve_auc"] for item in items]
        horizon = np.asarray([item["horizon_rve"] for item in items], dtype=np.float64)
        cursor[group[-1]] = {
            "run_count": len(items),
            "auc_recovery": quantile_summary(recoveries),
            "temporal_rve_auc": quantile_summary(aucs),
            "failure_rate": float(np.mean([item["failed"] for item in items])),
            "worst_run": min(items, key=lambda item: item["auc_recovery_fraction"]),
            "horizon_rve_median": np.median(horizon, axis=0).tolist(),
            "horizon_rve_q25": np.quantile(horizon, 0.25, axis=0).tolist(),
            "horizon_rve_q75": np.quantile(horizon, 0.75, axis=0).tolist(),
        }
    return output


def load_samples(data_root: Path) -> list[dict]:
    samples = []
    for path in sorted(data_root.rglob("*.npz")):
        with np.load(path) as loaded:
            required = {"future_3d", "pseudo_gt", "points_3d_history", "points_2d_at_t0", "intrinsics"}
            if not required.issubset(loaded.files):
                continue
            future = loaded["future_3d"][:, :9, :].transpose(1, 0, 2).astype(np.float64)
            pseudo_gt = loaded["pseudo_gt"].transpose(1, 0, 2).astype(np.float64)
            history = loaded["points_3d_history"].astype(np.float64)
            points_2d = loaded["points_2d_at_t0"].astype(np.float64)
            intrinsics = loaded["intrinsics"].astype(np.float64)
        video_group = infer_video_group(path)
        graph_config_id = array_hash(history, points_2d, intrinsics)
        supervision_config_id = array_hash(history, pseudo_gt, points_2d)
        trajectory_config_id = array_hash(history, pseudo_gt, future, points_2d)
        samples.append(
            {
                "path": path.relative_to(data_root).as_posix(),
                "video_group": video_group,
                "variant": path.parent.parent.name,
                "graph_config_id": graph_config_id,
                "supervision_config_id": supervision_config_id,
                "trajectory_config_id": trajectory_config_id,
                "future": future,
                "pseudo_gt": pseudo_gt,
                "history": history,
                "points_2d": points_2d,
                "intrinsics": intrinsics,
            }
        )
    return samples


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("experiences_results"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiences_results/surface-relation-e2-lite-2026-09-22"))
    parser.add_argument("--base-seed", type=int, default=20260922)
    parser.add_argument("--num-seeds", type=int, default=20)
    args = parser.parse_args()

    samples = load_samples(args.data_root)
    if not samples:
        raise RuntimeError(f"No compatible npz files found under {args.data_root}")

    graph_representatives: dict[str, dict] = {}
    trajectory_representatives: dict[str, dict] = {}
    variant_members: dict[str, list[str]] = defaultdict(list)
    for sample in samples:
        graph_representatives.setdefault(sample["graph_config_id"], sample)
        trajectory_representatives.setdefault(sample["trajectory_config_id"], sample)
        variant_members[sample["trajectory_config_id"]].append(sample["path"])

    diagnostics_records = []
    graph_cache: dict[tuple[str, str, int], list[tuple[int, int]]] = {}
    for graph_config_id, sample in graph_representatives.items():
        history_2d = project_points(sample["history"].copy(), sample["intrinsics"])
        for graph_type in GRAPH_TYPES:
            for k in (2, 3, 4):
                edges = build_graph(graph_type, sample["points_2d"], sample["history"][-1], k)
                graph_cache[(graph_config_id, graph_type, k)] = edges
                diagnostics_records.append(
                    {
                        "video_group": sample["video_group"],
                        "variant": sample["variant"],
                        "graph_config_id": graph_config_id,
                        "graph_type": graph_type,
                        "k": k,
                        **graph_diagnostics(edges, sample["history"], history_2d, graph_type, k),
                    }
                )

    runs = []
    graph_index = {name: index for index, name in enumerate(GRAPH_TYPES)}
    condition_index = {name: index for index, name in enumerate(NOISE_CONDITIONS)}
    for sample_index, sample in enumerate(trajectory_representatives.values()):
        p0 = torch.as_tensor(sample["future"], dtype=torch.float64)
        pgt = torch.as_tensor(sample["pseudo_gt"], dtype=torch.float64)
        for graph_type in GRAPH_TYPES:
            for k in (2, 3, 4):
                edges = graph_cache[(sample["graph_config_id"], graph_type, k)]
                if not edges:
                    continue
                incidence = incidence_matrix(p0.shape[1], edges)
                oracle_delta = torch.einsum("en,tnd->ted", incidence, pgt - p0)
                target_relations = torch.einsum("en,tnd->ted", incidence, pgt)
                edge_scale = float(torch.median(torch.linalg.vector_norm(target_relations, dim=-1)))
                for condition in NOISE_CONDITIONS:
                    seed_count = 1 if condition == "oracle" else args.num_seeds
                    for seed_index in range(seed_count):
                        seed_sequence = np.random.SeedSequence(
                            [
                                args.base_seed,
                                sample_index,
                                graph_index[graph_type],
                                k,
                                condition_index[condition],
                                seed_index,
                            ]
                        )
                        rng = np.random.default_rng(seed_sequence)
                        proposed = noisy_relation(oracle_delta, condition, edge_scale, rng)
                        correction, _ = solve_component_gauge(proposed, edges, p0.shape[1])
                        metrics = trajectory_metrics(p0 + correction, pgt, p0, edges)
                        runs.append(
                            {
                                "video_group": sample["video_group"],
                                "variant": sample["variant"],
                                "trajectory_config_id": sample["trajectory_config_id"],
                                "member_file_count": len(variant_members[sample["trajectory_config_id"]]),
                                "graph_type": graph_type,
                                "k": k,
                                "condition": condition,
                                "seed_index": seed_index,
                                **metrics,
                            }
                        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "e2_lite_runs.jsonl").open("w", encoding="utf-8") as handle:
        for run in runs:
            handle.write(json.dumps(run, ensure_ascii=False) + "\n")

    diagnostic_payload = {
        "graph_config_count": len(graph_representatives),
        "records": diagnostics_records,
    }
    write_json(output_dir / "graph_diagnostics.json", diagnostic_payload)

    source_video_seed_runs = aggregate_runs_by_seed(
        runs, ("video_group", "graph_type", "k", "condition")
    )
    macro_seed_runs = aggregate_runs_by_seed(
        source_video_seed_runs, ("graph_type", "k", "condition")
    )
    summaries = {
        "by_source_video": summarize_runs(
            source_video_seed_runs, ("video_group", "graph_type", "k", "condition")
        ),
        "macro_by_source_video": summarize_runs(macro_seed_runs, ("graph_type", "k", "condition")),
        "by_trajectory_variant": summarize_runs(
            runs, ("trajectory_config_id", "graph_type", "k", "condition")
        ),
    }
    write_json(output_dir / "noise_robustness_summary.json", summaries)

    manifest = {
        "base_seed": args.base_seed,
        "num_noise_seeds": args.num_seeds,
        "source_file_count": len(samples),
        "source_video_count": len({sample["video_group"] for sample in samples}),
        "graph_config_count": len(graph_representatives),
        "supervision_config_count": len({sample["supervision_config_id"] for sample in samples}),
        "trajectory_config_count": len(trajectory_representatives),
        "graph_types": list(GRAPH_TYPES),
        "k_values": [2, 3, 4],
        "conditions": list(NOISE_CONDITIONS),
        "temporal_auc_definition": "trapezoidal integral of per-horizon mean edge RVE over normalized horizon [0,1]",
        "hybrid_definition": "2D kNN candidates filtered by observed depth jump and observed 3D distance; no pixel-mask boundary map is available",
        "time_alignment": "future_3d[:9] matched to pseudo_gt; timestamp correspondence remains unverified",
        "variant_members": dict(sorted(variant_members.items())),
        "run_count": len(runs),
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
