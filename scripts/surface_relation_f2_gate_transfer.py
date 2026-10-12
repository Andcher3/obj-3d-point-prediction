#!/usr/bin/env python3
"""Bounded F2 gate-transfer probe. No transfer label is read before model selection.

The existing PointOdyssey DA3 input was computed with full-video context, so
this is a cross-video diagnostic, NOT a valid observed-RGB-only benchmark.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from surface_relation_f1_baseline_structure_audit import transform_frame0_to_camera
from surface_relation_f2_leave_one_video_out import metric, symmetric_knn
from surface_relation_f2_observed_relations import (
    OddEdgeMLP,
    features,
    fit_normalizer,
    load_video,
    loss,
    predict as predict_edge,
    weighted_projector,
)


class NodeResidual(nn.Module):
    """Pointwise comparator with approximately the same parameters as edge MLP."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 3))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def node_features(v: dict) -> torch.Tensor:
    p, h, history = v["p0"], v["h"], v["history"]
    t, n = p.shape[:2]
    last = history[-1]
    recent = (history[-1] - history[-3]) / 2
    long = (history[-1] - history[0]) / 6
    parts = [h, p - p.mean(1, keepdim=True)]
    for x in (last, recent, long):
        parts.append((x - x.mean(0, keepdim=True))[None].expand(t, -1, -1))
    parts.append(torch.arange(t, dtype=torch.float64)[:, None, None].expand(t, n, 1) / 8)
    return torch.cat(parts, -1)


def fit_node_normalizer(videos: dict[str, dict], ids: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    sample = torch.cat([node_features(videos[vid])[:7].reshape(-1, 29) for vid in ids])
    return sample.mean(0), sample.std(0).clamp_min(1e-6)


def predict_node(v: dict, model: NodeResidual, times: np.ndarray, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    residual = model((node_features(v)[times] - mean) / std)
    keep = torch.as_tensor(v["valid"], dtype=torch.bool)
    residual = residual - residual[:, keep].mean(1, keepdim=True)
    return v["p0"][times] + residual


def edge_shift(v: dict) -> float:
    """Label-free difference between frozen future edges and observed t0 edges."""
    ei, ej = v["ei"], v["ej"]
    pred = v["p0"][:, ej] - v["p0"][:, ei]
    t0 = v["history"][-1, ej] - v["history"][-1, ei]
    return float(torch.linalg.vector_norm(pred - t0[None], dim=-1).mean())


def load_transfer(cache_dir: Path, full_dir: Path, query_track_path: Path) -> tuple[dict, dict]:
    cache = np.load(cache_dir / "p0_h_cache_aligned.npz", allow_pickle=True)
    user = np.load(full_dir / "full22_user_res392.npz", allow_pickle=True)
    future_track = np.load(full_dir / "full22_tracks_res392.npz")
    observed_track = np.load(query_track_path)
    uv_mismatch = float(np.max(np.abs(observed_track["query_uv_original"] - future_track["query_uv_original"])))
    if uv_mismatch > 1e-4:
        raise ValueError(f"Query order does not match: {cache_dir.name}: {uv_mismatch}")
    history = transform_frame0_to_camera(
        np.asarray(observed_track["trackcraft3r_tracks"], dtype=np.float64),
        np.asarray(user["extrinsics_w2c"], dtype=np.float64), 6,
    )
    p0 = np.asarray(cache["p0_formal9"], dtype=np.float64)
    h = np.asarray(cache["h_output_latent_formal9"], dtype=np.float64)
    teacher = np.asarray(cache["pseudo_gt_aligned"], dtype=np.float64)
    teacher_t0 = np.asarray(cache["pseudo_gt_t0_aligned"], dtype=np.float64)
    # Per-query teacher displacement is anchored at the observed-only t0.
    target = history[-1][None] + (teacher - teacher_t0[None])
    anchor = np.asarray(cache["query_anchor_match_error_px"], dtype=np.float64)
    valid = np.asarray(cache["pseudo_gt_validity"], dtype=bool).all(0) & (anchor <= 5)
    if history.shape != (7, 48, 3) or p0.shape != (9, 48, 3) or h.shape != (9, 48, 16) or int(valid.sum()) < 40:
        raise ValueError(f"Invalid transfer geometry or validity: {cache_dir.name}")
    if not all(np.isfinite(x).all() for x in (history, p0, h, target)):
        raise ValueError(f"Nonfinite transfer data: {cache_dir.name}")
    scale = max(float(np.linalg.norm(np.ptp(p0[0], axis=0))), 1e-8)
    graph = symmetric_knn(np.asarray(cache["query_uv_t0_source"], dtype=np.float64), 3)
    edges = [(i, j) for i, j in graph if valid[i] and valid[j]]
    ei, ej = np.asarray(edges, dtype=np.int64).T
    v = {
        "id": cache_dir.name, "p0": torch.as_tensor(p0 / scale, dtype=torch.float64),
        "gt": torch.as_tensor(target / scale, dtype=torch.float64),
        "h": torch.as_tensor(h, dtype=torch.float64),
        "history": torch.as_tensor(history / scale, dtype=torch.float64),
        "valid": valid, "edges_all": graph, "edges": edges,
        "ei": torch.as_tensor(ei, dtype=torch.long),
        "ej": torch.as_tensor(ej, dtype=torch.long), "scale": scale,
    }
    v["projector_plain"] = weighted_projector(edges, torch.ones(len(edges), dtype=torch.float64), 48)
    audit = {
        "query_uv_max_abs_difference_px": uv_mismatch,
        "observed_only_trackcraft_frame_count": len(history),
        "future_teacher_frame_count": int(future_track["trackcraft3r_tracks"].shape[0]),
        "valid_queries": int(valid.sum()), "edges": len(edges),
        "t0_old_cache_vs_new_observed_median_m": float(np.median(np.linalg.norm(
            history[-1] - np.asarray(cache["observed_only_trackcraft_t0"], dtype=np.float64), axis=-1
        ))),
        "t0_new_observed_vs_teacher_median_m": float(np.median(np.linalg.norm(history[-1] - teacher_t0, axis=-1))),
        "anchor_max_px": float(anchor.max()),
        "observed_only_score": edge_shift(v),
        "scale_m": scale,
    }
    return v, audit


def train_model(
    name: str, videos: dict[str, dict], ids: list[str], seed: int, steps: int,
) -> tuple[nn.Module, torch.Tensor, torch.Tensor, float, int]:
    torch.manual_seed(seed)
    if name == "edge":
        mean, std, directed = fit_normalizer(videos, ids, "history", np.arange(7))
        model = OddEdgeMLP(len(mean), directed).double()
        predict = lambda v, t: predict_edge(v, model, "history", t, mean, std, 1.)
    else:
        mean, std = fit_node_normalizer(videos, ids)
        edge_dim = features(videos[ids[0]], "history")[0].shape[-1]
        edge_params = (edge_dim + 1) * 32 + (32 + 1) * 3
        node_dim = node_features(videos[ids[0]]).shape[-1]
        hidden = round((edge_params - 3) / (node_dim + 4))
        model = NodeResidual(node_dim, hidden).double()
        predict = lambda v, t: predict_node(v, model, t, mean, std)
    param_count = sum(p.numel() for p in model.parameters())
    assert float((predict(videos[ids[0]], np.arange(1)) - videos[ids[0]]["p0"][:1]).abs().max()) == 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_state = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        best_val = float(np.mean([float(loss(videos[i], predict(videos[i], np.arange(7, 9)), np.arange(7, 9), False)) for i in ids]))
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        objective = sum(loss(videos[i], predict(videos[i], np.arange(7)), np.arange(7), False) for i in ids) / len(ids)
        objective.backward()
        optimizer.step()
        if step % 10 == 0 or step == steps:
            with torch.no_grad():
                val = float(np.mean([float(loss(videos[i], predict(videos[i], np.arange(7, 9)), np.arange(7, 9), False)) for i in ids]))
            if val < best_val:
                best_val, best_state = val, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, mean, std, best_val, param_count


def diagnostics(v: dict, correction: torch.Tensor, gates: dict[str, float]) -> dict:
    rows = {}
    for label, g in gates.items():
        pred = v["p0"] + g * correction
        rows[label] = metric(pred, v["gt"], v["edges_all"], v["scale"], v["valid"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dev-root", type=Path, required=True)
    parser.add_argument("--transfer-root", type=Path, required=True)
    parser.add_argument("--tracks-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    args = parser.parse_args()
    spec = json.loads(args.protocol.read_text(encoding="utf-8"))
    if args.steps != spec["training_steps"] or spec["fixed_gate"] != 0.25:
        raise ValueError("Protocol diverged from locked configuration")
    torch.set_num_threads(1)
    ids = spec["development_video_ids"]
    videos = {i: load_video(args.dev_root / i, 5., 3, "anchored_displacement") for i in ids}
    dev_scores = {i: edge_shift(videos[i]) for i in ids}
    threshold = float(np.median(list(dev_scores.values())))
    def gates(v: dict) -> dict[str, float]:
        return {"baseline": 0., "fixed_0.25": 0.25,
                "abstain_or_0.25": 0.25 if edge_shift(v) >= threshold else 0.}

    # Historical cross-fitted corrections, used only for development diagnostics.
    historical = np.load(
        "experiences_results/surface-relation-f2-direction-gate-2026-09-27/corrections.npz"
    )
    if list(historical["video_ids"]) != ids or list(historical["methods"])[0] != "history":
        raise ValueError("Cross-fitted development arrays do not match protocol")
    dev_diagnostics = {}
    for vi, vid in enumerate(ids):
        v = videos[vid]
        gs = gates(v)
        dev_diagnostics[vid] = {
            "baseline_edge_shift": dev_scores[vid], "gates": gs,
            "history_crossfit": [diagnostics(v, torch.as_tensor(historical["corrections"][vi, 0, seed], dtype=torch.float64), gs) for seed in range(5)],
        }

    # No transfer cache or label has been read prior to training and gate freeze.
    models: dict[str, list] = {"edge": [], "node": []}
    train_rows = []
    for name in ("edge", "node"):
        for seed in spec["seeds"]:
            model, mean, std, best_val, param_count = train_model(name, videos, ids, seed, args.steps)
            models[name].append((model, mean, std))
            train_rows.append({"model": name, "seed": seed, "best_val": best_val, "parameters": param_count})
            print(json.dumps({"stage": "train", "model": name, "seed": seed, "best_val": best_val}), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({name: [
        {"state": model.state_dict(), "normalizer_mean": mean, "normalizer_std": std}
        for model, mean, std in pack] for name, pack in models.items()}, args.output_dir / "trained_heads.pt")
    freeze = {"threshold_from_development_only": threshold, "development_observed_scores": dev_scores,
              "fixed_gate": 0.25, "training_rows": train_rows, "development_diagnostics": dev_diagnostics,
              "protocol": str(args.protocol)}
    (args.output_dir / "frozen_before_transfer.json").write_text(json.dumps(freeze, indent=2), encoding="utf-8")

    external = {}
    for vid in spec["transfer_video_ids"]:
        v, audit = load_transfer(
            args.transfer_root / "f2_bundles" / vid,
            args.transfer_root / vid, args.tracks_root / f"{vid}.npz",
        )
        gs = gates(v)
        runs = {}
        for name, pack in models.items():
            runs[name] = []
            for (model, mean, std), seed in zip(pack, spec["seeds"]):
                with torch.no_grad():
                    full = (predict_edge(v, model, "history", np.arange(9), mean, std, 1.)
                            if name == "edge" else predict_node(v, model, np.arange(9), mean, std))
                    runs[name].append({"seed": seed, "metrics": diagnostics(v, full - v["p0"], gs)})
        external[vid] = {"audit": audit, "gates": gs,
                         "baseline": metric(v["p0"], v["gt"], v["edges_all"], v["scale"], v["valid"]),
                         "runs": runs}
        print(json.dumps({"stage": "transfer", "video": vid, "gates": gs,
                          "edge_scores": [r["metrics"]["abstain_or_0.25"]["edge_vector_error_normalized"] for r in runs["edge"]]}), flush=True)

    result = {"status": "COMPLETED_EXPLORATORY_CROSS_RESOLUTION", "protocol": str(args.protocol),
              "freeze": "frozen_before_transfer.json", "development": freeze,
              "transfer": external, "known_limits": spec["known_limits"]}
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
