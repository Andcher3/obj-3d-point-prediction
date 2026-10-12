#!/usr/bin/env python3
"""F2 query-held-out probe over all future horizons."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from surface_relation_feasibility import solve_component_gauge
from surface_relation_f2_redesigned_pilot import (
    ZeroInitMLP,
    metrics,
    standardize,
    standardize_fit,
    symmetric_knn,
    transform_frame0_to_camera,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trackcraft-npz", type=Path, required=True)
    ap.add_argument("--motionforesight-npz", type=Path, required=True)
    ap.add_argument("--user-npz", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--train-query-count", type=int, default=32)
    ap.add_argument("--k", type=int, default=3)
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
    n = p0.shape[1]
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

    node_raw = torch.cat(
        [h, p0, tau[:, None, None].expand(-1, n, 1)], dim=-1
    )
    edge_raw = torch.cat(
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

    def evaluate(
        pred: torch.Tensor,
        held_nodes: np.ndarray,
        held_edge_mask: np.ndarray,
    ) -> dict[str, float]:
        held = torch.as_tensor(held_nodes, dtype=torch.long)
        ei_h = torch.as_tensor(
            [i for (i, _), keep in zip(edges, held_edge_mask) if keep],
            dtype=torch.long,
        )
        ej_h = torch.as_tensor(
            [j for (_, j), keep in zip(edges, held_edge_mask) if keep],
            dtype=torch.long,
        )
        point = torch.linalg.vector_norm(pred[:, held] - gt[:, held], dim=-1)
        pc = pred - pred.mean(dim=1, keepdim=True)
        gc = gt - gt.mean(dim=1, keepdim=True)
        centered = torch.linalg.vector_norm(pc[:, held] - gc[:, held], dim=-1)
        pe = pred[:, ej_h] - pred[:, ei_h]
        ge = gt[:, ej_h] - gt[:, ei_h]
        eerr = torch.linalg.vector_norm(pe - ge, dim=-1)
        enorm = torch.linalg.vector_norm(ge, dim=-1).clamp_min(1e-8)
        erel = eerr / enorm
        shape = torch.abs(
            torch.linalg.vector_norm(pe, dim=-1)
            - torch.linalg.vector_norm(ge, dim=-1)
        ) / enorm
        return {
            "held_query_count": int(len(held_nodes)),
            "held_edge_count": int(ei_h.numel()),
            "raw_ade_m": float(point.mean()),
            "raw_fde_m": float(point[-1].mean()),
            "centered_error_normalized": float((centered / scale).mean()),
            "edge_relative_error": float(erel.mean()),
            "edge_relative_error_median": float(torch.median(erel)),
            "edge_shape_relative_error": float(shape.mean()),
        }

    rows = []
    for seed in range(args.seeds):
        torch.manual_seed(8000 + seed)
        np.random.seed(8000 + seed)
        random.seed(8000 + seed)
        train_nodes = np.sort(
            np.random.default_rng(8000 + seed).choice(
                n, size=args.train_query_count, replace=False
            )
        )
        held_nodes = np.asarray(
            sorted(set(range(n)).difference(set(train_nodes))), dtype=np.int64
        )
        train_mask = np.zeros(n, dtype=bool)
        train_mask[train_nodes] = True
        held_edge_mask = np.asarray(
            [not (train_mask[i] and train_mask[j]) for i, j in edges], dtype=bool
        )
        train_edge_mask = ~held_edge_mask
        train_edge_idx = np.where(train_edge_mask)[0]
        train_edge_t = torch.as_tensor(train_edge_idx, dtype=torch.long)
        node_mean, node_std = standardize_fit(
            node_raw[:, train_nodes].reshape(-1, node_raw.shape[-1])
        )
        edge_mean, edge_std = standardize_fit(
            edge_raw[:, train_edge_idx].reshape(-1, edge_raw.shape[-1])
        )
        node_x = standardize(node_raw, node_mean, node_std)
        edge_x = standardize(edge_raw, edge_mean, edge_std)

        baseline_test = evaluate(p0, held_nodes, held_edge_mask)
        rows.append(
            {
                "method": "baseline",
                "seed": seed,
                "train_query_count": int(len(train_nodes)),
                "held_queries": held_nodes.tolist(),
                "metrics": baseline_test,
            }
        )

        def objective_node(pred: torch.Tensor) -> torch.Tensor:
            held = torch.as_tensor(train_nodes, dtype=torch.long)
            pc = pred - pred.mean(dim=1, keepdim=True)
            gc = gt - gt.mean(dim=1, keepdim=True)
            ptrain = pc[:, held]
            gtrain = gc[:, held]
            return (ptrain - gtrain).pow(2).mean()

        def objective_edge(raw: torch.Tensor) -> torch.Tensor:
            return (raw[:, train_edge_t] - (edge_gt - edge_p0)[:, train_edge_t]).pow(2).mean()

        for method in ("node_centered", "h_edge_dynamic"):
            if method == "node_centered":
                model = ZeroInitMLP(node_x.shape[-1], 32).double()
                x = node_x
            else:
                model = ZeroInitMLP(edge_x.shape[-1], 32).double()
                x = edge_x
            opt = torch.optim.Adam(model.parameters(), lr=0.01)
            best_loss, best_state = float("inf"), None
            for _ in range(args.steps):
                opt.zero_grad()
                raw = model(x)
                if method == "node_centered":
                    corr = raw - raw.mean(dim=1, keepdim=True)
                    loss = objective_node(p0 + corr)
                else:
                    loss = objective_edge(raw)
                loss.backward()
                opt.step()
                lv = float(loss.detach())
                if lv < best_loss:
                    best_loss = lv
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(best_state)
            with torch.no_grad():
                raw = model(x)
                if method == "node_centered":
                    pred = p0 + raw - raw.mean(dim=1, keepdim=True)
                else:
                    corr, _ = solve_component_gauge(raw, edges, n)
                    pred = p0 + corr
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "train_query_count": int(len(train_nodes)),
                    "held_queries": held_nodes.tolist(),
                    "parameter_count": sum(p.numel() for p in model.parameters()),
                    "learned_gate": float(model.gate.detach()),
                    "best_training_loss": best_loss,
                    "metrics": evaluate(pred, held_nodes, held_edge_mask),
                }
            )

    aggregate = {}
    for method in ("baseline", "node_centered", "h_edge_dynamic"):
        mr = [r for r in rows if r["method"] == method]
        keys = mr[0]["metrics"].keys()
        aggregate[method] = {
            k: {
                "median": float(np.median([r["metrics"][k] for r in mr])),
                "p10": float(np.quantile([r["metrics"][k] for r in mr], 0.1)),
                "p90": float(np.quantile([r["metrics"][k] for r in mr], 0.9)),
            }
            for k in keys
        }

    base = [r for r in rows if r["method"] == "baseline"]
    wins = {}
    for method in ("node_centered", "h_edge_dynamic"):
        wins[method] = sum(
            r["metrics"]["edge_relative_error"] < b["metrics"]["edge_relative_error"]
            and r["metrics"]["centered_error_normalized"]
            <= b["metrics"]["centered_error_normalized"] * 1.02
            and r["metrics"]["raw_ade_m"] <= b["metrics"]["raw_ade_m"] * 1.02
            for r, b in zip(
                [x for x in rows if x["method"] == method], base
            )
        )
    required = max(3, (args.seeds + 1) // 2)
    payload = {
        "status": "PASS",
        "scope": "F2 query-held-out all-horizon probe",
        "video_id": "9069bd26-0139-4e88-9957-9f9489b35a91",
        "protocol": {
            "future_slots": int(len(future)),
            "train_query_count": args.train_query_count,
            "held_query_count": int(n - args.train_query_count),
            "graph": f"2D symmetric kNN k={args.k}",
            "seeds": args.seeds,
            "all_horizons_used_for_training": True,
            "loss": {
                "node_centered": "centered-coordinate MSE on train queries",
                "h_edge_dynamic": "edge correction MSE on train-train edges",
            },
        },
        "rows": rows,
        "aggregate": aggregate,
        "decision": {
            "wins_vs_baseline": wins,
            "required_wins": required,
            "interpretation": (
                "Query-held-out relation correction shows a positive single-video signal."
                if max(wins.values()) >= required
                else "No query-held-out relation correction advantage."
            ),
        },
        "limitations": [
            "One video only; this is not video-level generalization.",
            "TrackCraft3R remains pseudo-GT.",
            "The query split uses all future horizons for training and tests spatial query transfer.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
