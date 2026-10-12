#!/usr/bin/env python3
"""Redesigned F2 pilot with identity-preserving relation adapters.

This version fixes the previous pilot's main comparison issues:
* all learned heads start exactly at the MotionForesight baseline;
* training uses the same centered-coordinate/edge-relational objective for
  every learned head;
* feature standardization is fit on the training horizons only;
* validation selects the checkpoint, while the last horizons are held out;
* raw ADE/FDE are reported separately and are not mixed into the structural
  loss.

The current repository contains one complete F0 video, so this remains a
single-video temporal pilot and cannot establish video-level generalization.
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
    d += np.eye(len(uv), dtype=np.float64) * 1e9
    return sorted(
        {
            tuple(sorted((i, int(j))))
            for i in range(len(uv))
            for j in np.argsort(d[i])[:k]
            if i != int(j)
        }
    )


def transform_frame0_to_camera(
    tracks_frame0: np.ndarray, extrinsics_w2c: np.ndarray, camera_index: int
) -> np.ndarray:
    f0_to_world = np.linalg.inv(extrinsics_w2c[0].astype(np.float64))
    world_to_camera = extrinsics_w2c[camera_index].astype(np.float64)
    xyz1 = np.concatenate(
        [tracks_frame0.astype(np.float64), np.ones((*tracks_frame0.shape[:2], 1))],
        axis=-1,
    )
    return (xyz1 @ f0_to_world.T @ world_to_camera.T)[..., :3]


class ZeroInitMLP(nn.Module):
    """MLP whose output is exactly zero at initialization."""

    def __init__(self, input_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 3),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.gate = nn.Parameter(torch.ones((), dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate * self.net(x)


def standardize_fit(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return mean, std


def standardize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    edges: list[tuple[int, int]],
    scale: float,
) -> dict[str, float]:
    ei = torch.as_tensor([i for i, _ in edges], dtype=torch.long)
    ej = torch.as_tensor([j for _, j in edges], dtype=torch.long)
    point = torch.linalg.vector_norm(pred - target, dim=-1)
    pred_c = pred - pred.mean(dim=1, keepdim=True)
    target_c = target - target.mean(dim=1, keepdim=True)
    centered = torch.linalg.vector_norm(pred_c - target_c, dim=-1)
    pred_e = pred[:, ej] - pred[:, ei]
    target_e = target[:, ej] - target[:, ei]
    e_err = torch.linalg.vector_norm(pred_e - target_e, dim=-1)
    e_norm = torch.linalg.vector_norm(target_e, dim=-1).clamp_min(1e-8)
    e_rel = e_err / e_norm
    shape = torch.abs(
        torch.linalg.vector_norm(pred_e, dim=-1)
        - torch.linalg.vector_norm(target_e, dim=-1)
    ) / e_norm
    return {
        "raw_ade_m": float(point.mean()),
        "raw_fde_m": float(point[-1].mean()),
        "centered_error_m": float(centered.mean()),
        "centered_error_normalized": float((centered / scale).mean()),
        "edge_vector_error_m": float(e_err.mean()),
        "edge_relative_error": float(e_rel.mean()),
        "edge_relative_error_median": float(torch.median(e_rel)),
        "edge_shape_relative_error": float(shape.mean()),
        "edge_shape_relative_error_median": float(torch.median(shape)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trackcraft-npz", type=Path, required=True)
    ap.add_argument("--motionforesight-npz", type=Path, required=True)
    ap.add_argument("--user-npz", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    tc = np.load(args.trackcraft_npz, allow_pickle=True)
    mf = np.load(args.motionforesight_npz, allow_pickle=True)
    user = np.load(args.user_npz, allow_pickle=True)
    cache_mode = "p0_formal9" in mf.files and "pseudo_gt_aligned" in mf.files
    if cache_mode:
        p0_all = np.asarray(mf["p0_formal9"], dtype=np.float64)
        h_all = np.asarray(mf["h_output_latent_formal9"], dtype=np.float64)
        gt_all = np.asarray(mf["pseudo_gt_aligned"], dtype=np.float64)
        uv = np.asarray(mf["query_uv_t0_source"], dtype=np.float64)
        future = np.arange(p0_all.shape[0])
    else:
        tracks = np.asarray(tc["trackcraft3r_tracks"], dtype=np.float64)
        p0_all = np.asarray(mf["p0_query"], dtype=np.float64)
        h_all = np.asarray(mf["h_output_latent"], dtype=np.float64)
        uv = np.asarray(tc["query_uv_original"], dtype=np.float64)
        obs_count = int(np.asarray(user["observed_frame_count"]).item())
        if tracks.shape != p0_all.shape or h_all.shape[:2] != tracks.shape[:2]:
            raise RuntimeError(
                f"shape mismatch tracks={tracks.shape}, p0={p0_all.shape}, H={h_all.shape}"
            )
        gt_all = transform_frame0_to_camera(
            tracks,
            np.asarray(user["extrinsics_w2c"], dtype=np.float64),
            obs_count - 1,
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
    tau = torch.linspace(0.0, 1.0, len(future), dtype=torch.float64)
    bbox_scale = max(float(np.linalg.norm(gt_all[future[0]].ptp(axis=0))), 1e-8)

    # The final five slots are held out; the middle two select the checkpoint.
    train_idx = np.arange(0, len(future) - 7)
    val_idx = np.arange(len(future) - 7, len(future) - 5)
    test_idx = np.arange(len(future) - 5, len(future))
    splits = {
        "train": train_idx.tolist(),
        "validation": val_idx.tolist(),
        "test": test_idx.tolist(),
    }

    # Fit feature normalization on train horizons only.
    node_raw = torch.cat([h, p0, tau[:, None, None].expand(-1, p0.shape[1], 1)], -1)
    edge_raw = torch.cat(
        [
            h[:, ei] + h[:, ej],
            torch.abs(h[:, ei] - h[:, ej]),
            edge_p0,
            tau[:, None, None].expand(-1, len(edges), 1),
        ],
        -1,
    )
    node_mean, node_std = standardize_fit(node_raw[train_idx].reshape(-1, node_raw.shape[-1]))
    edge_mean, edge_std = standardize_fit(edge_raw[train_idx].reshape(-1, edge_raw.shape[-1]))
    node_x = standardize(node_raw, node_mean, node_std)
    edge_x = standardize(edge_raw, edge_mean, edge_std)

    def objective(pred: torch.Tensor, indices: np.ndarray) -> torch.Tensor:
        pred_c = pred - pred.mean(dim=1, keepdim=True)
        target_c = gt[indices] - gt[indices].mean(dim=1, keepdim=True)
        ei_local = edge_p0[indices]
        # Coordinate and edge losses share equal weight. Both are centered/
        # relational, so no method gets a free global translation signal.
        pi = pred[:, ei]
        pj = pred[:, ej]
        edge_pred = pj - pi
        edge_target = edge_gt[indices]
        return 0.5 * (pred_c - target_c).pow(2).mean() + 0.5 * (
            edge_pred - edge_target
        ).pow(2).mean()

    def fit_method(name: str, seed: int) -> dict:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if name == "geometry_edge":
            model = ZeroInitMLP(edge_x.shape[-1] - (2 * h.shape[-1]), hidden=32).double()
            # geometry-only features: baseline edge vector and normalized tau
            x = standardize(
                torch.cat([edge_p0, tau[:, None, None].expand(-1, len(edges), 1)], -1),
                *standardize_fit(
                    torch.cat(
                        [edge_p0[train_idx], tau[train_idx, None, None].expand(-1, len(edges), 1)],
                        -1,
                    ).reshape(-1, 4)
                ),
            )
            feature_kind = "geometry+time"
        elif name == "node_residual":
            model = ZeroInitMLP(node_x.shape[-1], hidden=32).double()
            x = node_x
            feature_kind = "H+p0+time"
        elif name == "h_edge":
            model = ZeroInitMLP(edge_x.shape[-1], hidden=32).double()
            x = edge_x
            feature_kind = "H-pair+p0-edge+time"
        else:
            raise ValueError(name)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        best_state = None
        best_val = float("inf")
        train_history = []

        for step in range(args.steps):
            optimizer.zero_grad()
            if name == "node_residual":
                pred_train = p0[train_idx] + model(x[train_idx])
            else:
                raw = model(x[train_idx])
                corr, _ = solve_component_gauge(raw, edges, p0.shape[1])
                pred_train = p0[train_idx] + corr
            loss = objective(pred_train, train_idx)
            loss.backward()
            optimizer.step()
            train_history.append(float(loss.detach()))
            if step % 10 == 0 or step == args.steps - 1:
                with torch.no_grad():
                    if name == "node_residual":
                        pred_val = p0[val_idx] + model(x[val_idx])
                    else:
                        raw_val = model(x[val_idx])
                        corr_val, _ = solve_component_gauge(raw_val, edges, p0.shape[1])
                        pred_val = p0[val_idx] + corr_val
                    val_loss = float(objective(pred_val, val_idx))
                    if val_loss < best_val:
                        best_val = val_loss
                        best_state = {
                            k: v.detach().clone() for k, v in model.state_dict().items()
                        }
        model.load_state_dict(best_state)
        with torch.no_grad():
            if name == "node_residual":
                pred_all = p0 + model(x)
            else:
                raw_all = model(x)
                corr_all, _ = solve_component_gauge(raw_all, edges, p0.shape[1])
                pred_all = p0 + corr_all
        return {
            "method": name,
            "seed": seed,
            "feature_kind": feature_kind,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "learned_gate": float(model.gate.detach()),
            "best_validation_objective": best_val,
            "train_objective_final": train_history[-1],
            "validation": metrics(pred_all[val_idx], gt[val_idx], edges, bbox_scale),
            "test": metrics(pred_all[test_idx], gt[test_idx], edges, bbox_scale),
        }

    baseline_row = {
        "method": "baseline",
        "seed": None,
        "feature_kind": "none",
        "parameter_count": 0,
        "validation": metrics(p0[val_idx], gt[val_idx], edges, bbox_scale),
        "test": metrics(p0[test_idx], gt[test_idx], edges, bbox_scale),
    }
    rows = [baseline_row]
    for seed in range(args.seeds):
        for method in ("geometry_edge", "node_residual", "h_edge"):
            rows.append(fit_method(method, 5000 + seed))

    aggregate = {}
    for method in ("baseline", "geometry_edge", "node_residual", "h_edge"):
        method_rows = [r for r in rows if r["method"] == method]
        aggregate[method] = {}
        for split in ("validation", "test"):
            keys = method_rows[0][split].keys()
            aggregate[method][split] = {
                key: {
                    "median": float(np.median([r[split][key] for r in method_rows])),
                    "p10": float(np.quantile([r[split][key] for r in method_rows], 0.10)),
                    "p90": float(np.quantile([r[split][key] for r in method_rows], 0.90)),
                }
                for key in keys
            }
        aggregate[method]["parameter_count"] = method_rows[0]["parameter_count"]

    baseline_test = baseline_row["test"]
    h_rows = [r for r in rows if r["method"] == "h_edge"]
    h_wins = [
        r["test"]["edge_relative_error"] < baseline_test["edge_relative_error"]
        and r["test"]["centered_error_normalized"]
        <= baseline_test["centered_error_normalized"] * 1.02
        and r["test"]["raw_ade_m"] <= baseline_test["raw_ade_m"] * 1.02
        for r in h_rows
    ]
    positive_gate = bool(sum(h_wins) >= max(3, (args.seeds + 1) // 2))

    payload = {
        "status": "PASS",
        "scope": "F2 redesigned identity-preserving single-video temporal pilot",
        "video_id": args.output.stem,
        "pseudo_gt_note": "TrackCraft3R remains pseudo-GT; no external 3D ground truth.",
        "protocol": {
            "future_slots": int(len(future)),
            "splits": splits,
            "graph": f"2D symmetric kNN k={args.k}",
            "seeds": args.seeds,
            "steps_per_seed": args.steps,
            "loss": "0.5 centered-coordinate MSE + 0.5 edge-vector MSE",
            "head_initialization": "zero final layer; exact baseline at step zero",
            "checkpoint_selection": "minimum validation objective",
            "feature_normalization": "fit on training horizons only",
        },
        "rows": rows,
        "aggregate": aggregate,
        "decision": {
            "h_edge_test_gate_wins": int(sum(h_wins)),
            "required_wins": max(3, (args.seeds + 1) // 2),
            "h_edge_positive_pilot_gate": positive_gate,
            "interpretation": (
                "The corrected pilot supports expanding to independent videos."
                if positive_gate
                else "The corrected pilot does not support a Relation Head advantage; "
                "do not start F3."
            ),
        },
        "limitations": [
            "Only one complete F0 video is available.",
            "Temporal holdout is not video-level generalization.",
            "Pseudo-GT quality limits the strength of the conclusion.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
