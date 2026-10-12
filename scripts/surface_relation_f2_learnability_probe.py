#!/usr/bin/env python3
"""F2: frozen-backbone relation learnability probe on one F0 video.

The probe is deliberately small and temporal-holdout based. It compares an
untrained geometry baseline, a geometry-only edge MLP, an H+geometry node
residual, and an H+geometry edge Relation Head. TrackCraft3R remains a
pseudo-GT source; this script does not establish real-world accuracy.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from surface_relation_feasibility import solve_component_gauge


def symmetric_knn(uv: np.ndarray, k: int = 3) -> list[tuple[int, int]]:
    d = np.linalg.norm(uv[:, None] - uv[None, :], axis=-1)
    d = d + np.eye(len(uv), dtype=np.float64) * 1e9
    directed = [set(np.argsort(d[i])[:k].tolist()) for i in range(len(uv))]
    return sorted(
        {
            tuple(sorted((i, int(j))))
            for i, js in enumerate(directed)
            for j in js
            if i != int(j)
        }
    )


def transform_frame0_to_camera(
    tracks_frame0: np.ndarray, extrinsics_w2c: np.ndarray, camera_index: int
) -> np.ndarray:
    f0_to_world = np.linalg.inv(np.asarray(extrinsics_w2c[0], dtype=np.float64))
    w_to_cam = np.asarray(extrinsics_w2c[camera_index], dtype=np.float64)
    xyz1 = np.concatenate(
        [tracks_frame0.astype(np.float64), np.ones((*tracks_frame0.shape[:2], 1))],
        axis=-1,
    )
    return (xyz1 @ f0_to_world.T @ w_to_cam.T)[..., :3]


class EdgeMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NodeMLP(nn.Module):
    def __init__(self, input_dim: int = 19, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    edges: list[tuple[int, int]],
    scale: float,
) -> dict[str, float]:
    ei = torch.as_tensor([i for i, _ in edges], device=pred.device)
    ej = torch.as_tensor([j for _, j in edges], device=pred.device)
    point = torch.linalg.vector_norm(pred - target, dim=-1)
    centered_pred = pred - pred.mean(dim=1, keepdim=True)
    centered_target = target - target.mean(dim=1, keepdim=True)
    centered = torch.linalg.vector_norm(centered_pred - centered_target, dim=-1)
    pred_edge = pred[:, ej] - pred[:, ei]
    target_edge = target[:, ej] - target[:, ei]
    edge_err = torch.linalg.vector_norm(pred_edge - target_edge, dim=-1)
    edge_norm = torch.linalg.vector_norm(target_edge, dim=-1)
    edge_rel = edge_err / edge_norm.clamp_min(1e-8)
    shape_rel = torch.abs(
        torch.linalg.vector_norm(pred_edge, dim=-1)
        - torch.linalg.vector_norm(target_edge, dim=-1)
    ) / edge_norm.clamp_min(1e-8)
    return {
        "raw_point_ade_m": float(point.mean().detach()),
        "raw_point_fde_m": float(point[-1].mean().detach()),
        "centered_point_error_m": float(centered.mean().detach()),
        "centered_point_error_normalized": float((centered / scale).mean().detach()),
        "edge_vector_error_m": float(edge_err.mean().detach()),
        "edge_relative_error": float(edge_rel.mean().detach()),
        "edge_relative_error_median": float(torch.median(edge_rel).detach()),
        "edge_length_shape_relative_error": float(shape_rel.mean().detach()),
        "edge_length_shape_relative_error_median": float(
            torch.median(shape_rel).detach()
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trackcraft-npz", type=Path, required=True)
    ap.add_argument("--motionforesight-npz", type=Path, required=True)
    ap.add_argument("--user-npz", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    tc = np.load(args.trackcraft_npz, allow_pickle=True)
    mf = np.load(args.motionforesight_npz, allow_pickle=True)
    user = np.load(args.user_npz, allow_pickle=True)
    tracks = np.asarray(tc["trackcraft3r_tracks"], dtype=np.float64)
    p0_all = np.asarray(mf["p0_query"], dtype=np.float64)
    h_all = np.asarray(mf["h_output_latent"], dtype=np.float64)
    uv = np.asarray(tc["query_uv_original"], dtype=np.float64)
    if tracks.shape != p0_all.shape or h_all.shape[:2] != tracks.shape[:2]:
        raise RuntimeError(
            f"shape mismatch tracks={tracks.shape}, p0={p0_all.shape}, H={h_all.shape}"
        )
    obs_count = int(np.asarray(user["observed_frame_count"]).item())
    gt_all = transform_frame0_to_camera(
        tracks, np.asarray(user["extrinsics_w2c"], dtype=np.float64), obs_count - 1
    )
    future = np.arange(obs_count, tracks.shape[0])
    p0 = torch.as_tensor(p0_all[future], dtype=torch.float64)
    h = torch.as_tensor(h_all[future], dtype=torch.float64)
    gt = torch.as_tensor(gt_all[future], dtype=torch.float64)
    edges = symmetric_knn(uv, args.k)
    ei = torch.as_tensor([i for i, _ in edges], dtype=torch.long)
    ej = torch.as_tensor([j for _, j in edges], dtype=torch.long)
    edge_p0 = p0[:, ej] - p0[:, ei]
    edge_gt = gt[:, ej] - gt[:, ei]
    scale = float(np.linalg.norm(gt_all[obs_count - 1].ptp(axis=0)))
    scale = max(scale, 1e-8)

    train_idx = np.arange(0, min(9, len(future)))
    val_idx = np.arange(len(train_idx), min(len(train_idx) + 3, len(future)))
    test_idx = np.arange(min(len(train_idx) + 3, len(future)), len(future))
    splits = {
        "train": train_idx.tolist(),
        "validation": val_idx.tolist(),
        "test": test_idx.tolist(),
    }

    def evaluate_model(name: str, seed: int) -> dict:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if name == "geometry_only":
            pred = p0
            return {
                "name": name,
                "seed": seed,
                "train_loss": None,
                "validation": metrics(pred[val_idx], gt[val_idx], edges, scale),
                "test": metrics(pred[test_idx], gt[test_idx], edges, scale),
                "parameter_count": 0,
            }

        if name == "geometry_edge_mlp":
            model: nn.Module = EdgeMLP(3).double()
        elif name == "node_residual":
            model = NodeMLP(19).double()
        elif name == "h_geometry_edge":
            model = EdgeMLP(2 * h.shape[-1] + 3).double()
        else:
            raise ValueError(name)

        opt = torch.optim.Adam(model.parameters(), lr=0.03)
        train_loss = float("nan")
        for _ in range(args.steps):
            opt.zero_grad()
            if name == "geometry_edge_mlp":
                raw = model(edge_p0[train_idx])
                correction, _ = solve_component_gauge(
                    raw, edges, p0.shape[1]
                )
                pred_train = p0[train_idx] + correction
            elif name == "h_geometry_edge":
                hi = h[train_idx][:, ei]
                hj = h[train_idx][:, ej]
                feat = torch.cat(
                    [hi + hj, torch.abs(hi - hj), edge_p0[train_idx]], dim=-1
                )
                raw = model(feat)
                correction, _ = solve_component_gauge(
                    raw, edges, p0.shape[1]
                )
                pred_train = p0[train_idx] + correction
            else:
                feat = torch.cat([h[train_idx], p0[train_idx]], dim=-1)
                pred_train = p0[train_idx] + model(feat)
            centered_pred = pred_train - pred_train.mean(dim=1, keepdim=True)
            centered_gt = gt[train_idx] - gt[train_idx].mean(dim=1, keepdim=True)
            loss = (centered_pred - centered_gt).pow(2).mean()
            loss.backward()
            opt.step()
            train_loss = float(loss.detach())

        with torch.no_grad():
            if name == "geometry_edge_mlp":
                raw_all = model(edge_p0)
                correction, _ = solve_component_gauge(
                    raw_all, edges, p0.shape[1]
                )
                pred_all = p0 + correction
            elif name == "h_geometry_edge":
                hi = h[:, ei]
                hj = h[:, ej]
                feat = torch.cat([hi + hj, torch.abs(hi - hj), edge_p0], dim=-1)
                raw_all = model(feat)
                correction, _ = solve_component_gauge(
                    raw_all, edges, p0.shape[1]
                )
                pred_all = p0 + correction
            else:
                pred_all = p0 + model(torch.cat([h, p0], dim=-1))
            val = metrics(pred_all[val_idx], gt[val_idx], edges, scale)
            test = metrics(pred_all[test_idx], gt[test_idx], edges, scale)
        return {
            "name": name,
            "seed": seed,
            "train_loss": train_loss,
            "validation": val,
            "test": test,
            "parameter_count": sum(p.numel() for p in model.parameters()),
        }

    methods = [
        "geometry_only",
        "geometry_edge_mlp",
        "node_residual",
        "h_geometry_edge",
    ]
    rows = [
        evaluate_model(method, 1000 + seed)
        for seed in range(args.seeds)
        for method in methods
    ]

    aggregate: dict[str, dict] = {}
    for method in methods:
        method_rows = [r for r in rows if r["name"] == method]
        aggregate[method] = {}
        for split in ("validation", "test"):
            aggregate[method][split] = {}
            keys = method_rows[0][split].keys()
            for key in keys:
                vals = np.asarray([r[split][key] for r in method_rows], dtype=float)
                aggregate[method][split][key] = {
                    "median": float(np.median(vals)),
                    "p10": float(np.quantile(vals, 0.10)),
                    "p90": float(np.quantile(vals, 0.90)),
                }
        aggregate[method]["parameter_count"] = method_rows[0]["parameter_count"]

    # A conservative pilot comparison: H+geometry edge must beat both trained
    # alternatives on held-out edge RVE in at least 3/5 seeds to be called a
    # positive learnability signal.
    h_rows = [r for r in rows if r["name"] == "h_geometry_edge"]
    baseline_rows = [r for r in rows if r["name"] == "geometry_only"]
    geom_rows = [r for r in rows if r["name"] == "geometry_edge_mlp"]
    node_rows = [r for r in rows if r["name"] == "node_residual"]
    wins_vs_baseline = [
        h["test"]["edge_relative_error"] < b["test"]["edge_relative_error"]
        for h, b in zip(h_rows, baseline_rows)
    ]
    wins_vs_node = [
        h["test"]["edge_relative_error"] < n["test"]["edge_relative_error"]
        for h, n in zip(h_rows, node_rows)
    ]
    wins_vs_trained_geometry = [
        h["test"]["edge_relative_error"] < g["test"]["edge_relative_error"]
        for h, g in zip(h_rows, geom_rows)
    ]
    required_wins = max(3, (args.seeds + 1) // 2)
    h_advantage = bool(
        sum(wins_vs_baseline) >= required_wins
        and sum(wins_vs_node) >= required_wins
    )
    payload = {
        "status": "PASS",
        "scope": "F2 frozen-backbone single-video temporal holdout probe",
        "video_id": "9069bd26-0139-4e88-9957-9f9489b35a91",
        "pseudo_gt_note": "TrackCraft3R is pseudo-GT; this is a learnability probe, not external accuracy validation.",
        "protocol": {
            "future_slots": int(len(future)),
            "splits": splits,
            "graph": f"2D symmetric kNN k={args.k}",
            "seeds": args.seeds,
            "steps_per_seed": args.steps,
            "loss": "centered-coordinate MSE on train horizons",
        },
        "methods": {
            "geometry_only": "untrained MotionForesight p0",
            "geometry_edge_mlp": "edge MLP on baseline edge vector only + constrained L2",
            "node_residual": "node MLP on [H, p0]",
            "h_geometry_edge": "edge Relation Head on [H_i+H_j, |H_i-H_j|, baseline edge]",
        },
        "rows": rows,
        "aggregate": aggregate,
        "decision": {
            "required_wins": required_wins,
            "h_geometry_edge_test_edge_rve_wins_vs_geometry_only": int(
                sum(wins_vs_baseline)
            ),
            "h_geometry_edge_test_edge_rve_wins_vs_node_residual": int(
                sum(wins_vs_node)
            ),
            "h_geometry_edge_test_edge_rve_wins_vs_trained_geometry_edge": int(
                sum(wins_vs_trained_geometry)
            ),
            "h_geometry_edge_advantage_gate": h_advantage,
            "interpretation": (
                "Positive pilot signal; proceed to F3 only if this advantage is "
                "reproduced on independent videos."
                if h_advantage
                else "No stable H/edge advantage; do not promote Relation Head."
            ),
        },
        "limitations": [
            "Only one complete F0 video is available.",
            "Temporal holdout does not replace video-level generalization.",
            "No external 3D ground truth is used.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
