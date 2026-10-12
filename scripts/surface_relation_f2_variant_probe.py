#!/usr/bin/env python3
"""F2 variant probe: centroid-preserving node residual and dynamic edge head."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from surface_relation_feasibility import solve_component_gauge
from surface_relation_f2_redesigned_pilot import (
    metrics,
    standardize,
    standardize_fit,
    symmetric_knn,
    transform_frame0_to_camera,
    ZeroInitMLP,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trackcraft-npz", type=Path, required=True)
    ap.add_argument("--motionforesight-npz", type=Path, required=True)
    ap.add_argument("--user-npz", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--train-count", type=int, default=8)
    ap.add_argument("--val-count", type=int, default=2)
    args = ap.parse_args()

    tc = np.load(args.trackcraft_npz, allow_pickle=True)
    mf = np.load(args.motionforesight_npz, allow_pickle=True)
    user = np.load(args.user_npz, allow_pickle=True)
    tracks = np.asarray(tc["trackcraft3r_tracks"], dtype=np.float64)
    p0_all = np.asarray(mf["p0_query"], dtype=np.float64)
    h_all = np.asarray(mf["h_output_latent"], dtype=np.float64)
    uv = np.asarray(tc["query_uv_original"], dtype=np.float64)
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
    edge_vel = torch.zeros_like(edge_p0)
    edge_vel[1:] = edge_p0[1:] - edge_p0[:-1]
    edge_acc = torch.zeros_like(edge_p0)
    edge_acc[2:] = edge_vel[2:] - edge_vel[1:-1]
    tau = torch.linspace(0.0, 1.0, len(future), dtype=torch.float64)
    scale = max(float(np.linalg.norm(gt_all[obs_count - 1].ptp(axis=0))), 1e-8)
    if args.train_count + args.val_count >= len(future):
        raise ValueError("train_count + val_count must leave at least one test slot")
    train_idx = np.arange(0, args.train_count)
    val_idx = np.arange(args.train_count, args.train_count + args.val_count)
    test_idx = np.arange(args.train_count + args.val_count, len(future))

    node_raw = torch.cat(
        [h, p0, tau[:, None, None].expand(-1, p0.shape[1], 1)], dim=-1
    )
    edge_dyn_raw = torch.cat(
        [
            h[:, ei] + h[:, ej],
            torch.abs(h[:, ei] - h[:, ej]),
            edge_p0,
            edge_vel,
            edge_acc,
            tau[:, None, None].expand(-1, len(edges), 1),
        ],
        dim=-1,
    )
    node_mean, node_std = standardize_fit(
        node_raw[train_idx].reshape(-1, node_raw.shape[-1])
    )
    edge_mean, edge_std = standardize_fit(
        edge_dyn_raw[train_idx].reshape(-1, edge_dyn_raw.shape[-1])
    )
    node_x = standardize(node_raw, node_mean, node_std)
    edge_x = standardize(edge_dyn_raw, edge_mean, edge_std)

    def objective(pred: torch.Tensor, indices: np.ndarray) -> torch.Tensor:
        pred_c = pred - pred.mean(dim=1, keepdim=True)
        gt_c = gt[indices] - gt[indices].mean(dim=1, keepdim=True)
        pred_e = pred[:, ej] - pred[:, ei]
        return 0.5 * (pred_c - gt_c).pow(2).mean() + 0.5 * (
            pred_e - edge_gt[indices]
        ).pow(2).mean()

    def fit(name: str, seed: int) -> dict:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if name == "node_centered":
            model = ZeroInitMLP(node_x.shape[-1], 32).double()
            x = node_x
        elif name == "h_edge_dynamic":
            model = ZeroInitMLP(edge_x.shape[-1], 32).double()
            x = edge_x
        else:
            raise ValueError(name)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        best_state, best_val = None, float("inf")
        for step in range(args.steps):
            opt.zero_grad()
            raw = model(x[train_idx])
            if name == "node_centered":
                correction = raw - raw.mean(dim=1, keepdim=True)
                pred_train = p0[train_idx] + correction
            else:
                correction, _ = solve_component_gauge(raw, edges, p0.shape[1])
                pred_train = p0[train_idx] + correction
            loss = objective(pred_train, train_idx)
            loss.backward()
            opt.step()
            if step % 10 == 0 or step == args.steps - 1:
                with torch.no_grad():
                    val_raw = model(x[val_idx])
                    if name == "node_centered":
                        val_corr = val_raw - val_raw.mean(dim=1, keepdim=True)
                    else:
                        val_corr, _ = solve_component_gauge(
                            val_raw, edges, p0.shape[1]
                        )
                    val_pred = p0[val_idx] + val_corr
                    val_loss = float(objective(val_pred, val_idx))
                    if val_loss < best_val:
                        best_val = val_loss
                        best_state = {
                            k: v.detach().clone()
                            for k, v in model.state_dict().items()
                        }
        model.load_state_dict(best_state)
        with torch.no_grad():
            all_raw = model(x)
            if name == "node_centered":
                all_corr = all_raw - all_raw.mean(dim=1, keepdim=True)
            else:
                all_corr, _ = solve_component_gauge(
                    all_raw, edges, p0.shape[1]
                )
            pred = p0 + all_corr
        return {
            "method": name,
            "seed": seed,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "learned_gate": float(model.gate.detach()),
            "best_validation_objective": best_val,
            "validation": metrics(pred[val_idx], gt[val_idx], edges, scale),
            "test": metrics(pred[test_idx], gt[test_idx], edges, scale),
        }

    baseline = {
        "method": "baseline",
        "seed": None,
        "parameter_count": 0,
        "validation": metrics(p0[val_idx], gt[val_idx], edges, scale),
        "test": metrics(p0[test_idx], gt[test_idx], edges, scale),
    }
    rows = [baseline]
    for seed in range(args.seeds):
        rows.extend(
            [
                fit("node_centered", 7000 + seed),
                fit("h_edge_dynamic", 7000 + seed),
            ]
        )

    aggregate = {}
    for method in ("baseline", "node_centered", "h_edge_dynamic"):
        mr = [r for r in rows if r["method"] == method]
        aggregate[method] = {}
        for split in ("validation", "test"):
            keys = mr[0][split].keys()
            aggregate[method][split] = {
                k: {
                    "median": float(np.median([r[split][k] for r in mr])),
                    "p10": float(np.quantile([r[split][k] for r in mr], 0.1)),
                    "p90": float(np.quantile([r[split][k] for r in mr], 0.9)),
                }
                for k in keys
            }
        aggregate[method]["parameter_count"] = mr[0]["parameter_count"]

    base = baseline["test"]
    def wins(name: str) -> int:
        return sum(
            r["test"]["edge_relative_error"] < base["edge_relative_error"]
            and r["test"]["centered_error_normalized"]
            <= base["centered_error_normalized"] * 1.02
            and r["test"]["raw_ade_m"] <= base["raw_ade_m"] * 1.02
            for r in rows
            if r["method"] == name
        )

    payload = {
        "status": "PASS",
        "scope": "F2 variant pilot targeting temporal extrapolation and centroid drift",
        "video_id": "9069bd26-0139-4e88-9957-9f9489b35a91",
        "protocol": {
            "future_slots": int(len(future)),
            "train": train_idx.tolist(),
            "validation": val_idx.tolist(),
            "test": test_idx.tolist(),
            "graph": f"2D symmetric kNN k={args.k}",
            "seeds": args.seeds,
            "head_initialization": "zero-init output; baseline-preserving residual",
            "loss": "0.5 centered-coordinate MSE + 0.5 edge-vector MSE",
        },
        "variants": {
            "node_centered": "H+p0+time node residual with per-horizon zero-mean correction",
            "h_edge_dynamic": "H pair + edge position/velocity/acceleration + time edge head",
        },
        "rows": rows,
        "aggregate": aggregate,
        "decision": {
            "node_centered_gate_wins": wins("node_centered"),
            "h_edge_dynamic_gate_wins": wins("h_edge_dynamic"),
            "required_wins": max(3, (args.seeds + 1) // 2),
            "interpretation": (
                "At least one variant deserves multi-video F2 expansion."
                if max(wins("node_centered"), wins("h_edge_dynamic"))
                >= max(3, (args.seeds + 1) // 2)
                else "Neither variant passes the single-video safety gate."
            ),
        },
        "limitations": [
            "Only one complete F0 video is available.",
            "TrackCraft3R remains pseudo-GT.",
            "Temporal holdout does not establish video-level generalization.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
