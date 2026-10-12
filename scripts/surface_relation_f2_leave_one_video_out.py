#!/usr/bin/env python3
"""Multi-video leave-one-video-out F2 for relation corrections.

The primary estimand is structural transfer across videos.  A held-out video
is never used to fit feature normalization, model parameters, or checkpoint
selection.  Coordinates are normalized by an observed-only scale from each
video, while raw ADE remains a secondary diagnostic because a relation
correction is gauge-fixed and cannot recover arbitrary global translation.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from surface_relation_feasibility import component_matrix, connected_components, incidence_matrix


def symmetric_knn(uv: np.ndarray, k: int = 3) -> list[tuple[int, int]]:
    d = np.linalg.norm(uv[:, None] - uv[None, :], axis=-1)
    d += np.eye(len(uv), dtype=np.float64) * 1e9
    edges = {
        tuple(sorted((i, int(j))))
        for i in range(len(uv))
        for j in np.argsort(d[i])[:k]
        if i != int(j)
    }
    return sorted(edges)


class ZeroInitMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh(), nn.Linear(hidden, 3))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def fit_stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return x.mean(0, keepdim=True), x.std(0, keepdim=True).clamp_min(1e-6)


def zscore(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def build_projector(edges: list[tuple[int, int]], num_nodes: int) -> torch.Tensor:
    """Precompute the fixed zero-mean least-squares edge-to-node map."""
    if not edges:
        return torch.zeros((num_nodes, 0), dtype=torch.float64)
    incidence = incidence_matrix(num_nodes, edges, dtype=torch.float64)
    components = connected_components(num_nodes, edges)
    gauge = component_matrix(num_nodes, components, dtype=torch.float64)
    lap = incidence.T @ incidence
    zeros = torch.zeros((len(components), len(components)), dtype=torch.float64)
    kkt = torch.cat(
        [torch.cat([lap, gauge], 1), torch.cat([gauge.T, zeros], 1)], 0
    )
    inv = torch.linalg.inv(kkt)
    return inv[:num_nodes, :num_nodes] @ incidence.T


def metric(
    pred: torch.Tensor,
    target: torch.Tensor,
    edges: list[tuple[int, int]],
    scale: float,
    valid_nodes: np.ndarray,
) -> dict[str, float | int]:
    valid = torch.as_tensor(valid_nodes, dtype=torch.bool)
    pred = pred[:, valid]
    target = target[:, valid]
    point = torch.linalg.vector_norm(pred - target, dim=-1)
    pred_c = pred - pred.mean(1, keepdim=True)
    target_c = target - target.mean(1, keepdim=True)
    centered = torch.linalg.vector_norm(pred_c - target_c, dim=-1)
    valid_index = {int(old): i for i, old in enumerate(np.flatnonzero(valid_nodes))}
    kept = [(valid_index[i], valid_index[j]) for i, j in edges if i in valid_index and j in valid_index]
    if kept:
        ei = torch.as_tensor([i for i, _ in kept], dtype=torch.long)
        ej = torch.as_tensor([j for _, j in kept], dtype=torch.long)
        pe = pred[:, ej] - pred[:, ei]
        te = target[:, ej] - target[:, ei]
        eerr = torch.linalg.vector_norm(pe - te, dim=-1)
        enorm = torch.linalg.vector_norm(te, dim=-1).clamp_min(1e-8)
        erel = eerr / enorm
        shape = (torch.linalg.vector_norm(pe, dim=-1) - torch.linalg.vector_norm(te, dim=-1)).abs() / enorm
        edge_abs = eerr.mean()
        edge_length_abs = (
            torch.linalg.vector_norm(pe, dim=-1) - torch.linalg.vector_norm(te, dim=-1)
        ).abs().mean()
        edge_rel = float(erel.mean())
        shape_rel = float(shape.mean())
        edge_count = len(kept)
    else:
        edge_rel = float("nan")
        shape_rel = float("nan")
        edge_abs = torch.tensor(float("nan"))
        edge_length_abs = torch.tensor(float("nan"))
        edge_count = 0
    return {
        "raw_ade_m": float(point.mean() * scale),
        "raw_fde_m": float(point[-1].mean() * scale),
        "centered_error_normalized": float((centered / 1.0).mean()),
        "centered_error_m": float(centered.mean() * scale),
        "edge_relative_error": edge_rel,
        "edge_shape_relative_error": shape_rel,
        "edge_vector_error_normalized": float(edge_abs),
        "edge_length_error_normalized": float(edge_length_abs),
        "valid_query_count": int(valid.sum()),
        "edge_count": edge_count,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--video-ids", nargs="+", required=True)
    ap.add_argument("--anchor-threshold", type=float, default=5.0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument(
        "--posthoc-shrink",
        type=float,
        default=1.0,
        help="multiply learned corrections only at evaluation time; training remains unchanged",
    )
    ap.add_argument(
        "--select-gate",
        action="store_true",
        help="select a correction gate from {0,.25,.5,.75,1} on training-video validation slots",
    )
    args = ap.parse_args()

    videos: dict[str, dict] = {}
    for vid in args.video_ids:
        d = args.bundle_root / vid
        cache = np.load(d / "p0_h_cache_aligned.npz", allow_pickle=True)
        p0 = np.asarray(cache["p0_formal9"], dtype=np.float64)
        h = np.asarray(cache["h_output_latent_formal9"], dtype=np.float64)
        gt = np.asarray(cache["pseudo_gt_aligned"], dtype=np.float64)
        valid = np.asarray(cache["pseudo_gt_validity"], dtype=bool).all(axis=0)
        anchor = np.asarray(cache["query_anchor_match_error_px"], dtype=np.float64)
        if args.anchor_threshold >= 0:
            valid &= anchor <= args.anchor_threshold
        uv = np.asarray(cache["query_uv_t0_source"], dtype=np.float64)
        if not valid.any():
            raise RuntimeError(f"{vid}: no valid query after quality filter")
        # Observed-only scale: no future target is used to normalize inputs.
        obs_scale = max(float(np.linalg.norm(p0[0].ptp(axis=0))), 1e-8)
        p0n = p0 / obs_scale
        gtn = gt / obs_scale
        edges = symmetric_knn(uv, args.k)
        videos[vid] = {
            "p0": torch.as_tensor(p0n, dtype=torch.float64),
            "gt": torch.as_tensor(gtn, dtype=torch.float64),
            "h": torch.as_tensor(h, dtype=torch.float64),
            "valid": valid,
            "edges": edges,
            "projector": None,
            "scale": obs_scale,
            "anchor_quantiles": np.quantile(anchor, [0.5, 0.9, 1.0]).tolist(),
        }

    # All methods share the same 0..6 train and 7..8 validation horizons.
    train_t = np.arange(0, 7)
    val_t = np.arange(7, 9)
    rows: list[dict] = []
    for held_out in args.video_ids:
        train_vids = [v for v in args.video_ids if v != held_out]
        test = videos[held_out]
        projector_cache: dict[tuple[tuple[int, int], ...], torch.Tensor] = {}

        def get_projector(edges: list[tuple[int, int]], n: int) -> torch.Tensor:
            key = tuple(edges)
            if key not in projector_cache:
                projector_cache[key] = build_projector(edges, n)
            return projector_cache[key]
        def build_node(v: dict, t: np.ndarray) -> torch.Tensor:
            p = v["p0"][t]
            c = p - p.mean(1, keepdim=True)
            tau = torch.as_tensor(t / 8.0, dtype=torch.float64)[:, None, None].expand(-1, p.shape[1], 1)
            return torch.cat([v["h"][t], c, tau], -1)

        def build_edge(v: dict, t: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            p = v["p0"][t]
            h = v["h"][t]
            ei = torch.as_tensor([i for i, _ in v["edges"]], dtype=torch.long)
            ej = torch.as_tensor([j for _, j in v["edges"]], dtype=torch.long)
            ep = p[:, ej] - p[:, ei]
            ex = torch.cat(
                [
                    h[:, ei] + h[:, ej],
                    (h[:, ei] - h[:, ej]).abs(),
                    ep,
                    torch.as_tensor(t / 8.0, dtype=torch.float64)[:, None, None].expand(-1, len(v["edges"]), 1),
                ],
                -1,
            )
            return ex, ei, ej

        node_train = torch.cat(
            [
                build_node(videos[v], train_t)[:, videos[v]["valid"]].reshape(-1, 20)
                for v in train_vids
            ],
            0,
        )
        edge_train_parts = []
        geom_train_parts = []
        for vname in train_vids:
            vv = videos[vname]
            ex0, ei0, ej0 = build_edge(vv, train_t)
            good = vv["valid"][ei0.numpy()] & vv["valid"][ej0.numpy()]
            edge_train_parts.append(ex0[:, good].reshape(-1, 36))
            p0 = vv["p0"][train_t]
            geom_train_parts.append(
                torch.cat(
                    [
                        p0[:, ej0[good]] - p0[:, ei0[good]],
                        torch.as_tensor(train_t / 8.0, dtype=torch.float64)[:, None, None].expand(
                            -1, int(good.sum()), 1
                        ),
                    ],
                    -1,
                ).reshape(-1, 4)
            )
        edge_train = torch.cat(edge_train_parts, 0)
        geom_train = torch.cat(geom_train_parts, 0)
        node_mean, node_std = fit_stats(node_train)
        edge_mean, edge_std = fit_stats(edge_train)
        geom_mean, geom_std = fit_stats(geom_train)

        # Per-video tensors are normalized by training-video statistics only.
        train_records = []
        for vid in train_vids:
            v = videos[vid]
            node_x = zscore(build_node(v, train_t), node_mean, node_std)
            edge_x, ei, ej = build_edge(v, train_t)
            good = v["valid"][ei.numpy()] & v["valid"][ej.numpy()]
            edge_x = zscore(edge_x[:, good], edge_mean, edge_std)
            p = v["p0"][train_t]
            geom_x = torch.cat(
                [
                    p[:, ej[good]] - p[:, ei[good]],
                    torch.as_tensor(train_t / 8.0, dtype=torch.float64)[:, None, None].expand(
                        -1, int(good.sum()), 1
                    ),
                ],
                -1,
            )
            geom_x = zscore(geom_x, geom_mean, geom_std)
            train_records.append((vid, node_x, edge_x, geom_x, ei[good], ej[good]))

        def objective(
            pred: torch.Tensor,
            target: torch.Tensor,
            ei: torch.Tensor,
            ej: torch.Tensor,
            valid_mask: np.ndarray,
        ) -> torch.Tensor:
            keep = torch.as_tensor(valid_mask, dtype=torch.bool)
            pc = pred[:, keep] - pred[:, keep].mean(1, keepdim=True)
            tc = target[:, keep] - target[:, keep].mean(1, keepdim=True)
            return 0.5 * (pc - tc).pow(2).mean() + 0.5 * (
                (pred[:, ej] - pred[:, ei]) - (target[:, ej] - target[:, ei])
            ).pow(2).mean()

        def predict(
            name: str,
            model: nn.Module | None,
            v: dict,
            t: np.ndarray,
            shrink: float,
        ) -> torch.Tensor:
            p = v["p0"][t]
            if name == "baseline":
                return p
            if name == "node_centered":
                x = zscore(build_node(v, t), node_mean, node_std)
                corr = model(x)
                keep = torch.as_tensor(v["valid"], dtype=torch.bool)
                corr = corr - corr[:, keep].mean(1, keepdim=True)
                return p + shrink * corr
            x, ei_all, ej_all = build_edge(v, t)
            good = v["valid"][ei_all.numpy()] & v["valid"][ej_all.numpy()]
            if name == "geometry_edge":
                x = torch.cat(
                    [
                        p[:, ej_all[good]] - p[:, ei_all[good]],
                        torch.as_tensor(t / 8.0, dtype=torch.float64)[:, None, None].expand(
                            -1, int(good.sum()), 1
                        ),
                    ],
                    -1,
                )
                x = zscore(x, geom_mean, geom_std)
            else:
                x = zscore(x[:, good], edge_mean, edge_std)
            kept_edges = [e for e, keep in zip(v["edges"], good.tolist()) if keep]
            projector = get_projector(kept_edges, p.shape[1])
            delta = torch.einsum("ne,...ed->...nd", projector, model(x))
            return p + shrink * delta

        methods = ["baseline", "geometry_edge", "node_centered", "h_edge"]
        for method in methods:
            method_rows = []
            for seed in range(args.seeds):
                torch.manual_seed(7000 + seed)
                random.seed(7000 + seed)
                np.random.seed(7000 + seed)
                if method == "baseline":
                    model = None
                elif method == "node_centered":
                    model = ZeroInitMLP(20).double()
                elif method == "geometry_edge":
                    model = ZeroInitMLP(4).double()
                else:
                    model = ZeroInitMLP(36).double()
                if model is not None:
                    opt = torch.optim.Adam(model.parameters(), lr=0.01)
                    best = None
                    best_val = float("inf")
                    for _ in range(args.steps):
                        opt.zero_grad()
                        total = torch.zeros((), dtype=torch.float64)
                        for vid, nx, ex, gx, ei, ej in train_records:
                            v = videos[vid]
                            if method == "node_centered":
                                pred = v["p0"][train_t] + model(nx)
                                keep = torch.as_tensor(v["valid"], dtype=torch.bool)
                                corr = pred - v["p0"][train_t]
                                corr = corr - corr[:, keep].mean(1, keepdim=True)
                                pred = v["p0"][train_t] + corr
                            else:
                                if method == "geometry_edge":
                                    raw = model(gx)
                                else:
                                    raw = model(ex)
                                kept_edges = list(zip(ei.tolist(), ej.tolist()))
                                projector = get_projector(kept_edges, v["p0"].shape[1])
                                delta = torch.einsum("ne,...ed->...nd", projector, raw)
                                pred = v["p0"][train_t] + delta
                            total = total + objective(pred, v["gt"][train_t], ei, ej, v["valid"])
                        total = total / len(train_records)
                        total.backward()
                        opt.step()
                        with torch.no_grad():
                            val_total = torch.zeros((), dtype=torch.float64)
                            for vid in train_vids:
                                v = videos[vid]
                                pv = predict(method, model, v, val_t, 1.0)
                                _, ei_v, ej_v = build_edge(v, val_t)
                                good_v = v["valid"][ei_v.numpy()] & v["valid"][ej_v.numpy()]
                                val_total = val_total + objective(
                                    pv,
                                    v["gt"][val_t],
                                    ei_v[good_v],
                                    ej_v[good_v],
                                    v["valid"],
                                )
                            val_total = val_total / len(train_vids)
                        if float(val_total) < best_val:
                            best_val = float(val_total)
                            best = {k: value.detach().clone() for k, value in model.state_dict().items()}
                    model.load_state_dict(best)
                selected_gate = args.posthoc_shrink
                gate_scores = {}
                if args.select_gate and method != "baseline":
                    candidates = [0.0, 0.25, 0.5, 0.75, 1.0]
                    with torch.no_grad():
                        for gate in candidates:
                            score = 0.0
                            for vid in train_vids:
                                v = videos[vid]
                                pv = predict(method, model, v, val_t, gate)
                                _, ei_v, ej_v = build_edge(v, val_t)
                                good_v = v["valid"][ei_v.numpy()] & v["valid"][ej_v.numpy()]
                                score += float(
                                    objective(
                                        pv,
                                        v["gt"][val_t],
                                        ei_v[good_v],
                                        ej_v[good_v],
                                        v["valid"],
                                    )
                                )
                            gate_scores[str(gate)] = score / len(train_vids)
                    selected_gate = min(candidates, key=lambda gate: gate_scores[str(gate)])
                pred = predict(method, model, test, np.arange(9), selected_gate)
                method_rows.append({
                    "seed": seed,
                    "test": metric(pred, test["gt"], test["edges"], test["scale"], test["valid"]),
                    "best_validation_objective": None if method == "baseline" else best_val,
                    "selected_gate": selected_gate,
                    "gate_validation_scores": gate_scores,
                })
            rows.append({"held_out_video": held_out, "method": method, "rows": method_rows})

    aggregate = {}
    for method in ["baseline", "geometry_edge", "node_centered", "h_edge"]:
        all_metrics = [r["test"] for row in rows if row["method"] == method for r in row["rows"]]
        aggregate[method] = {}
        for key in [
            "raw_ade_m",
            "centered_error_normalized",
            "edge_relative_error",
            "edge_shape_relative_error",
            "edge_vector_error_normalized",
            "edge_length_error_normalized",
        ]:
            vals = np.asarray([m[key] for m in all_metrics], dtype=float)
            aggregate[method][key] = {
                "median": float(np.nanmedian(vals)),
                "mean": float(np.nanmean(vals)),
                "per_video": {
                    vid: float(np.nanmedian([
                        r["test"][key] for row in rows if row["method"] == method and row["held_out_video"] == vid for r in row["rows"]
                    ]))
                    for vid in args.video_ids
                },
            }

    payload = {
        "status": "PASS",
        "scope": "F2 multi-video leave-one-video-out structural transfer",
        "protocol": {
            "videos": args.video_ids,
            "held_out_unit": "video",
            "train_time_slots": train_t.tolist(),
            "validation_time_slots": val_t.tolist(),
            "test_time_slots": list(range(9)),
            "graph": f"2D symmetric kNN k={args.k}",
            "anchor_threshold_px": args.anchor_threshold,
            "posthoc_shrink": args.posthoc_shrink,
            "select_gate_on_train_validation": args.select_gate,
            "seeds": args.seeds,
            "steps": args.steps,
            "normalization": "per-video observed-only p0 extent; feature statistics fit on training videos and train slots",
            "primary_metrics": [
                "centered_error_normalized",
                "edge_vector_error_normalized",
                "edge_length_error_normalized",
            ],
        },
        "video_diagnostics": {
            vid: {
                "valid_query_count": int(videos[vid]["valid"].sum()),
                "anchor_quantiles_px": videos[vid]["anchor_quantiles"],
                "edge_count": len(videos[vid]["edges"]),
            }
            for vid in args.video_ids
        },
        "rows": rows,
        "aggregate": aggregate,
        "decision": {
            "note": "This is a descriptive transfer experiment; no universal success claim is made from three videos.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
