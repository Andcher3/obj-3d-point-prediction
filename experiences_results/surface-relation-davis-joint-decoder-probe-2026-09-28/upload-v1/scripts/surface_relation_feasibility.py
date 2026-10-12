#!/usr/bin/env python3
"""CPU feasibility checks for component-aware surface-relation integration."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def incidence_matrix(num_nodes: int, edges: list[tuple[int, int]], *, dtype=torch.float64) -> torch.Tensor:
    matrix = torch.zeros((len(edges), num_nodes), dtype=dtype)
    for edge_index, (source, target) in enumerate(edges):
        matrix[edge_index, source] = -1.0
        matrix[edge_index, target] = 1.0
    return matrix


def connected_components(num_nodes: int, edges: Iterable[tuple[int, int]]) -> list[list[int]]:
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    for source, target in edges:
        adjacency[source].append(target)
        adjacency[target].append(source)
    components: list[list[int]] = []
    unseen = set(range(num_nodes))
    while unseen:
        start = min(unseen)
        queue = deque([start])
        unseen.remove(start)
        component: list[int] = []
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor in adjacency[node]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        components.append(sorted(component))
    return components


def component_matrix(num_nodes: int, components: list[list[int]], *, dtype=torch.float64) -> torch.Tensor:
    matrix = torch.zeros((num_nodes, len(components)), dtype=dtype)
    for component_index, component in enumerate(components):
        matrix[component, component_index] = 1.0
    return matrix


def solve_component_gauge(
    delta_relations: torch.Tensor,
    edges: list[tuple[int, int]],
    num_nodes: int,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Solve min ||W^0.5(BU-delta)||^2 with zero mean per component."""
    if not edges:
        correction = torch.zeros(
            (*delta_relations.shape[:-2], num_nodes, delta_relations.shape[-1]),
            dtype=delta_relations.dtype,
            device=delta_relations.device,
        )
        return correction, {"components": num_nodes, "condition_number": 1.0, "projection_residual": 0.0}

    dtype = delta_relations.dtype
    device = delta_relations.device
    incidence = incidence_matrix(num_nodes, edges, dtype=dtype).to(device)
    components = connected_components(num_nodes, edges)
    gauge = component_matrix(num_nodes, components, dtype=dtype).to(device)
    if weights is None:
        weights = torch.ones(len(edges), dtype=dtype, device=device)
    laplacian = incidence.T @ (weights[:, None] * incidence)
    weighted_gauge = gauge
    zeros = torch.zeros((len(components), len(components)), dtype=dtype, device=device)
    kkt = torch.cat(
        [torch.cat([laplacian, weighted_gauge], dim=1), torch.cat([weighted_gauge.T, zeros], dim=1)],
        dim=0,
    )
    rhs_nodes = torch.einsum("ne,...ed->...nd", incidence.T * weights[None, :], delta_relations)
    rhs_gauge = torch.zeros(
        (*rhs_nodes.shape[:-2], len(components), rhs_nodes.shape[-1]), dtype=dtype, device=device
    )
    rhs = torch.cat([rhs_nodes, rhs_gauge], dim=-2)
    solution = torch.linalg.solve(kkt, rhs)
    correction = solution[..., :num_nodes, :]
    projected = torch.einsum("en,...nd->...ed", incidence, correction)
    residual = torch.linalg.vector_norm(projected - delta_relations, dim=-1).mean()
    singular_values = torch.linalg.svdvals(kkt)
    condition_number = float((singular_values.max() / singular_values.min()).detach().cpu())
    diagnostics = {
        "components": len(components),
        "condition_number": condition_number,
        "projection_residual": float(residual.detach().cpu()),
    }
    return correction, diagnostics


def center_by_components(points: torch.Tensor, components: list[list[int]]) -> torch.Tensor:
    centered = points.clone()
    for component in components:
        centered[..., component, :] -= centered[..., component, :].mean(dim=-2, keepdim=True)
    return centered


def make_graph(num_nodes: int, component_sizes: list[int], rng: random.Random) -> list[tuple[int, int]]:
    if sum(component_sizes) != num_nodes:
        raise ValueError("component sizes must sum to num_nodes")
    edges: set[tuple[int, int]] = set()
    offset = 0
    for size in component_sizes:
        nodes = list(range(offset, offset + size))
        rng.shuffle(nodes)
        for index in range(1, len(nodes)):
            parent = nodes[rng.randrange(index)]
            edges.add(tuple(sorted((nodes[index], parent))))
        target_edges = min(size * 3, size * (size - 1) // 2)
        while len([edge for edge in edges if offset <= edge[0] < offset + size]) < target_edges:
            source, target = rng.sample(range(offset, offset + size), 2)
            edges.add(tuple(sorted((source, target))))
        offset += size
    return sorted(edges)


def run_e0(seed: int) -> dict:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    cases = []
    for num_nodes, component_sizes in ((8, [8]), (32, [13, 19]), (128, [31, 43, 54])):
        edges = make_graph(num_nodes, component_sizes, rng)
        components = connected_components(num_nodes, edges)
        p0 = torch.randn((4, num_nodes, 3), dtype=torch.float64)
        pgt = torch.randn((4, num_nodes, 3), dtype=torch.float64)
        incidence = incidence_matrix(num_nodes, edges)
        target_correction = pgt - p0
        delta = torch.einsum("en,tnd->ted", incidence, target_correction)
        recovered, diagnostics = solve_component_gauge(delta, edges, num_nodes)
        prediction = p0 + recovered
        centered_error = torch.sqrt(
            torch.mean((center_by_components(prediction, components) - center_by_components(pgt, components)) ** 2)
        ).item()
        zero, _ = solve_component_gauge(torch.zeros_like(delta), edges, num_nodes)
        zero_max = zero.abs().max().item()
        gauge_error = max(
            recovered[:, component, :].mean(dim=1).abs().max().item() for component in components
        )

        permutation = torch.randperm(num_nodes)
        inverse = torch.argsort(permutation)
        old_to_new = torch.empty(num_nodes, dtype=torch.long)
        old_to_new[permutation] = torch.arange(num_nodes)
        permuted_edges = sorted(
            tuple(sorted((int(old_to_new[source]), int(old_to_new[target])))) for source, target in edges
        )
        permuted_delta = torch.einsum(
            "en,tnd->ted", incidence_matrix(num_nodes, permuted_edges), target_correction[:, permutation]
        )
        permuted_recovered, _ = solve_component_gauge(permuted_delta, permuted_edges, num_nodes)
        permutation_error = (permuted_recovered[:, inverse] - recovered).abs().max().item()
        cases.append(
            {
                "num_nodes": num_nodes,
                "component_sizes": component_sizes,
                "num_edges": len(edges),
                "centered_rmse": centered_error,
                "zero_correction_max_abs": zero_max,
                "gauge_max_abs_mean": gauge_error,
                "permutation_max_abs": permutation_error,
                **diagnostics,
            }
        )

    num_nodes = 8
    edges = make_graph(num_nodes, [8], rng)
    delta = torch.randn((2, len(edges), 3), dtype=torch.float64, requires_grad=True)
    direction = torch.randn((2, num_nodes, 3), dtype=torch.float64)
    correction, _ = solve_component_gauge(delta, edges, num_nodes)
    objective = (correction * direction).sum()
    analytic = torch.autograd.grad(objective, delta)[0]
    epsilon = 1e-6
    checks = []
    for flat_index in (0, delta.numel() // 3, delta.numel() - 1):
        plus = delta.detach().clone().reshape(-1)
        minus = delta.detach().clone().reshape(-1)
        plus[flat_index] += epsilon
        minus[flat_index] -= epsilon
        plus_value = (solve_component_gauge(plus.reshape_as(delta), edges, num_nodes)[0] * direction).sum()
        minus_value = (solve_component_gauge(minus.reshape_as(delta), edges, num_nodes)[0] * direction).sum()
        numeric = (plus_value - minus_value) / (2.0 * epsilon)
        expected = analytic.reshape(-1)[flat_index]
        relative_error = float((numeric - expected).abs() / torch.clamp(expected.abs(), min=1e-12))
        checks.append({"flat_index": flat_index, "relative_error": relative_error})

    thresholds = {
        "centered_rmse": 1e-5,
        "zero_correction_max_abs": 1e-7,
        "permutation_max_abs": 1e-6,
        "gauge_max_abs_mean": 1e-7,
        "finite_difference_relative_error": 1e-3,
    }
    passed = all(
        case["centered_rmse"] < thresholds["centered_rmse"]
        and case["zero_correction_max_abs"] < thresholds["zero_correction_max_abs"]
        and case["permutation_max_abs"] < thresholds["permutation_max_abs"]
        and case["gauge_max_abs_mean"] < thresholds["gauge_max_abs_mean"]
        for case in cases
    ) and all(math.isfinite(item["relative_error"]) and item["relative_error"] < 1e-3 for item in checks)
    return {"passed": passed, "seed": seed, "thresholds": thresholds, "cases": cases, "gradient_checks": checks}


def knn_edges(points: np.ndarray, k: int) -> list[tuple[int, int]]:
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    edges: set[tuple[int, int]] = set()
    for source in range(len(points)):
        for target in np.argsort(distances[source])[:k]:
            edges.add(tuple(sorted((source, int(target)))))
    return sorted(edges)


def infer_video_group(path: Path) -> str:
    text = str(path).lower()
    if "drink_bottle" in text or "bottle" in text:
        return "drink_bottle"
    if "watering_can" in text:
        return "watering_can"
    if "sewing_scissors" in text or "scissors" in text:
        return "sewing_scissors"
    return path.parent.parent.name


def point_metrics(
    prediction: torch.Tensor, target: torch.Tensor, baseline: torch.Tensor, edges: list[tuple[int, int]]
) -> dict[str, float]:
    components = connected_components(prediction.shape[-2], edges)
    incidence = incidence_matrix(prediction.shape[-2], edges)
    absolute = torch.linalg.vector_norm(prediction - target, dim=-1)
    relative_prediction = torch.einsum("en,tnd->ted", incidence, prediction)
    relative_target = torch.einsum("en,tnd->ted", incidence, target)
    relative = torch.linalg.vector_norm(relative_prediction - relative_target, dim=-1)
    centered_prediction = center_by_components(prediction, components)
    centered_target = center_by_components(target, components)
    centered = torch.linalg.vector_norm(centered_prediction - centered_target, dim=-1)
    return {
        "ade": float(absolute.mean()),
        "fde": float(absolute[-1].mean()),
        "centered_point_error": float(centered.mean()),
        "relative_vector_error": float(relative.mean()),
        "structural_drift_auc": float(relative.mean(dim=1).mean()),
        "common_translation_shift": float(
            torch.linalg.vector_norm(prediction.mean(dim=1) - baseline.mean(dim=1), dim=-1).max()
        ),
    }


def aggregate_records(records: list[dict], key: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record[key]].append(record)
    result = {}
    for group_name, items in sorted(groups.items()):
        conditions = sorted(items[0]["conditions"])
        result[group_name] = {
            "file_count": len(items),
            "conditions": {
                condition: {
                    metric: float(np.mean([item["conditions"][condition][metric] for item in items]))
                    for metric in items[0]["conditions"][condition]
                    if isinstance(items[0]["conditions"][condition][metric], (int, float))
                }
                for condition in conditions
            },
        }
    return result


def run_e1(data_root: Path, seed: int, k: int) -> dict:
    files = sorted(data_root.rglob("*.npz"))
    rng = np.random.default_rng(seed)
    records = []
    for path in files:
        with np.load(path) as loaded:
            required = {"future_3d", "pseudo_gt", "points_3d_history"}
            if not required.issubset(loaded.files):
                continue
            p0_np = loaded["future_3d"][:, :9, :].transpose(1, 0, 2)
            pgt_np = loaded["pseudo_gt"].transpose(1, 0, 2)
            observed = loaded["points_3d_history"][-1]
        p0 = torch.as_tensor(p0_np, dtype=torch.float64)
        pgt = torch.as_tensor(pgt_np, dtype=torch.float64)
        edges = knn_edges(observed, k)
        incidence = incidence_matrix(p0.shape[1], edges)
        oracle_delta = torch.einsum("en,tnd->ted", incidence, pgt - p0)
        oracle_correction, oracle_diagnostics = solve_component_gauge(oracle_delta, edges, p0.shape[1])
        condition_predictions: dict[str, torch.Tensor] = {"baseline": p0, "oracle": p0 + oracle_correction}
        edge_lengths = torch.linalg.vector_norm(torch.einsum("en,tnd->ted", incidence, pgt), dim=-1)
        scale = float(torch.median(edge_lengths))
        for noise_fraction in (0.01, 0.05, 0.10, 0.20):
            noisy_delta = oracle_delta + torch.as_tensor(
                rng.normal(0.0, noise_fraction * scale, size=oracle_delta.shape), dtype=torch.float64
            )
            correction, _ = solve_component_gauge(noisy_delta, edges, p0.shape[1])
            condition_predictions[f"gaussian_{noise_fraction:.2f}"] = p0 + correction
        for outlier_fraction in (0.05, 0.10, 0.20):
            corrupted = oracle_delta.clone()
            count = max(1, round(outlier_fraction * len(edges)))
            selected_edges = rng.choice(len(edges), size=count, replace=False)
            corrupted[:, selected_edges, :] += torch.as_tensor(
                rng.normal(0.0, 0.5 * scale, size=(corrupted.shape[0], count, 3)), dtype=torch.float64
            )
            correction, _ = solve_component_gauge(corrupted, edges, p0.shape[1])
            condition_predictions[f"outlier_{outlier_fraction:.2f}"] = p0 + correction
        delete_count = max(1, round(0.2 * len(edges)))
        kept_indices = sorted(set(range(len(edges))) - set(rng.choice(len(edges), size=delete_count, replace=False).tolist()))
        kept_edges = [edges[index] for index in kept_indices]
        kept_delta = oracle_delta[:, kept_indices]
        correction, _ = solve_component_gauge(kept_delta, kept_edges, p0.shape[1])
        condition_predictions["delete_edges_0.20"] = p0 + correction

        split = p0.shape[1] // 2
        disconnected_edges = [
            edge for edge in edges if (edge[0] < split and edge[1] < split) or (edge[0] >= split and edge[1] >= split)
        ]
        if disconnected_edges:
            edge_lookup = {edge: index for index, edge in enumerate(edges)}
            disconnected_delta = oracle_delta[:, [edge_lookup[edge] for edge in disconnected_edges]]
            correction, _ = solve_component_gauge(disconnected_delta, disconnected_edges, p0.shape[1])
            condition_predictions["disconnect_halves"] = p0 + correction

        conditions = {name: point_metrics(value, pgt, p0, edges) for name, value in condition_predictions.items()}
        baseline_rve = conditions["baseline"]["relative_vector_error"]
        for condition in conditions.values():
            condition["rve_recovery_fraction"] = (
                1.0 - condition["relative_vector_error"] / baseline_rve if baseline_rve > 1e-12 else 0.0
            )
        records.append(
            {
                "path": str(path.relative_to(data_root)),
                "video_group": infer_video_group(path),
                "num_nodes": int(p0.shape[1]),
                "num_edges": len(edges),
                "edge_length_median": scale,
                "oracle_diagnostics": oracle_diagnostics,
                "conditions": conditions,
            }
        )
    if not records:
        raise RuntimeError(f"No compatible npz files found under {data_root}")
    video_aggregate = aggregate_records(records, "video_group")
    video_names = sorted(video_aggregate)
    macro_conditions = {}
    for condition in sorted(records[0]["conditions"]):
        sample_metrics = video_aggregate[video_names[0]]["conditions"][condition]
        macro_conditions[condition] = {
            metric: float(np.mean([video_aggregate[name]["conditions"][condition][metric] for name in video_names]))
            for metric in sample_metrics
        }
    oracle_recovery = macro_conditions["oracle"]["rve_recovery_fraction"]
    noise_recovery = macro_conditions["gaussian_0.05"]["rve_recovery_fraction"]
    return {
        "passed": oracle_recovery > 0.95 and noise_recovery > 0.50,
        "seed": seed,
        "graph": {"type": "observed_3d_symmetric_knn", "k": k},
        "alignment": "first 9 of 30 predicted steps matched to 9 pseudo-GT steps, following the existing metric behavior",
        "file_count": len(records),
        "video_group_count": len(video_aggregate),
        "pass_thresholds": {"oracle_rve_recovery": 0.95, "gaussian_0.05_rve_recovery": 0.50},
        "macro_by_video": {"conditions": macro_conditions},
        "by_video": video_aggregate,
        "records": records,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("experiences_results"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiences_results/surface-relation-feasibility-2026-09-22"))
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--knn-k", type=int, default=3)
    parser.add_argument("--stage", choices=("e0", "e1", "all"), default="all")
    args = parser.parse_args()

    summary = {"stage": args.stage, "seed": args.seed}
    if args.stage in ("e0", "all"):
        e0 = run_e0(args.seed)
        write_json(args.output_dir / "e0_solver_unit_tests.json", e0)
        summary["e0_passed"] = e0["passed"]
    if args.stage in ("e1", "all"):
        e1 = run_e1(args.data_root, args.seed, args.knn_k)
        write_json(args.output_dir / "e1_oracle_noise.json", e1)
        summary["e1_passed"] = e1["passed"]
        summary["e1_files"] = e1["file_count"]
        summary["e1_video_groups"] = e1["video_group_count"]
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
