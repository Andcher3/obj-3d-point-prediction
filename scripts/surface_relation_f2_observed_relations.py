#!/usr/bin/env python3
"""LOO probe for observed edge history, shared object state and safe correction.

Every feature and confidence used at inference is observed-only or from the
frozen MotionForesight prediction.  Future TrackCraft trajectories are labels
and evaluation targets only.  The held-out video is never used for fitting,
normalization, checkpoint choice or correction-gate selection.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from surface_relation_f2_leave_one_video_out import metric, symmetric_knn
from surface_relation_feasibility import component_matrix, connected_components, incidence_matrix


METHODS = ("history", "shared", "safe")
GATES = (0.0, 0.25, 0.5, 1.0)


class OddEdgeMLP(nn.Module):
    """Antisymmetric with respect to reversing an edge's directed features."""

    def __init__(self, dim: int, directed: slice, hidden: int = 32):
        super().__init__()
        self.directed = directed
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 3))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reverse = x.clone()
        reverse[..., self.directed] *= -1
        return 0.5 * (self.net(x) - self.net(reverse))


def weighted_projector(edges: list[tuple[int, int]], weights: torch.Tensor, n: int) -> torch.Tensor:
    if not edges:
        return torch.zeros((n, 0), dtype=torch.float64)
    b = incidence_matrix(n, edges, dtype=torch.float64)
    comps = connected_components(n, edges)
    c = component_matrix(n, comps, dtype=torch.float64)
    bw = b.T * weights[None, :]
    lap = bw @ b
    kkt = torch.cat(
        [torch.cat([lap, c], 1), torch.cat([c.T, torch.zeros((len(comps), len(comps)), dtype=torch.float64)], 1)],
        0,
    )
    rhs = torch.cat([bw, torch.zeros((len(comps), len(edges)), dtype=torch.float64)], 0)
    return torch.linalg.solve(kkt, rhs)[:n]


def load_video(directory: Path, threshold: float, k: int, target_mode: str) -> dict:
    cache = np.load(directory / "p0_h_cache_aligned.npz", allow_pickle=True)
    user = np.load(directory / "observed7_user.npz", allow_pickle=True)
    sparse_path = directory / "observed7_query_tracks.npz"
    dense = None if sparse_path.exists() else np.load(directory / "observed7_trackcraft_dense.npz")
    p0 = np.asarray(cache["p0_formal9"], dtype=np.float64)
    gt = np.asarray(cache["pseudo_gt_aligned"], dtype=np.float64)
    h = np.asarray(cache["h_output_latent_formal9"], dtype=np.float64)
    anchor = np.asarray(cache["query_anchor_match_error_px"], dtype=np.float64)
    valid = np.asarray(cache["pseudo_gt_validity"], dtype=bool).all(axis=0) & (anchor <= threshold)
    if sparse_path.exists():
        with np.load(sparse_path) as sparse:
            track_map = np.asarray(sparse["trackcraft3r_tracks"], dtype=np.float64)
        if track_map.shape != (7, 48, 3):
            raise ValueError(f"{directory.name}: invalid sparse observed trajectory shape {track_map.shape}")
    else:
        uv_dense = np.asarray(cache["query_uv_frame0_dense"], dtype=np.int64)
        track_map = np.asarray(dense["track_map"], dtype=np.float64)[:, uv_dense[:, 1], uv_dense[:, 0]]
    w2c = np.asarray(user["extrinsics_w2c"], dtype=np.float64)[-1]
    history = track_map @ w2c[:3, :3].T + w2c[:3, 3]
    recorded_t0 = np.asarray(cache["observed_only_trackcraft_t0"], dtype=np.float64)
    if target_mode == "anchored_displacement":
        # Same-query teacher displacement, anchored to the observed-only t0.
        # This changes supervision and evaluation only; it is never an input.
        teacher_t0 = np.asarray(cache["pseudo_gt_t0_aligned"], dtype=np.float64)
        gt = gt - teacher_t0[None] + recorded_t0[None]
    t0_difference = float(np.max(np.linalg.norm(history[-1] - recorded_t0, axis=-1)))
    if history.shape != (7, 48, 3) or t0_difference > 1e-5 or not np.isfinite(history).all():
        raise ValueError(f"{directory.name}: invalid observed-only history, t0 difference={t0_difference}")
    if p0.shape != (9, 48, 3) or gt.shape != p0.shape or h.shape[:2] != p0.shape[:2]:
        raise ValueError(f"{directory.name}: incompatible F2 arrays")
    if valid.sum() < 40:
        raise ValueError(f"{directory.name}: insufficient valid queries")
    scale = max(float(np.linalg.norm(np.ptp(p0[0], axis=0))), 1e-8)
    p0, gt, history = p0 / scale, gt / scale, history / scale
    all_edges = symmetric_knn(np.asarray(cache["query_uv_t0_source"], dtype=np.float64), k)
    edges = [(i, j) for i, j in all_edges if valid[i] and valid[j]]
    ei, ej = np.array(edges, dtype=np.int64).T
    edge_lengths = np.linalg.norm(history[-1, ej] - history[-1, ei], axis=-1)
    median_length = max(float(np.median(edge_lengths)), 1e-8)
    geometry_confidence = np.minimum(1.0, 2.0 * median_length / np.maximum(edge_lengths, 1e-8))
    anchor_confidence = np.exp(-0.5 * (anchor / threshold) ** 2)
    confidence = np.clip(geometry_confidence * np.sqrt(anchor_confidence[ei] * anchor_confidence[ej]), 0.05, 1.0)
    video = {
        "id": directory.name,
        "p0": torch.as_tensor(p0, dtype=torch.float64),
        "gt": torch.as_tensor(gt, dtype=torch.float64),
        "h": torch.as_tensor(h, dtype=torch.float64),
        "history": torch.as_tensor(history, dtype=torch.float64),
        "valid": valid,
        "edges_all": all_edges,
        "edges": edges,
        "ei": torch.as_tensor(ei, dtype=torch.long),
        "ej": torch.as_tensor(ej, dtype=torch.long),
        "confidence": torch.as_tensor(confidence, dtype=torch.float64),
        "scale": scale,
        "t0_difference_m": t0_difference,
        "t0_teacher_fit_median_m": float(np.median(np.linalg.norm(np.asarray(cache["pseudo_gt_t0_aligned"]) - recorded_t0, axis=-1))),
    }
    video["projector_plain"] = weighted_projector(edges, torch.ones(len(edges), dtype=torch.float64), 48)
    video["projector_weighted"] = weighted_projector(edges, video["confidence"], 48)
    return video


def features(v: dict, method: str) -> tuple[torch.Tensor, slice]:
    h, p, hist = v["h"], v["p0"], v["history"]
    ei, ej = v["ei"], v["ej"]
    t, e = p.shape[0], len(v["edges"])
    hpair = torch.cat([h[:, ei] + h[:, ej], (h[:, ei] - h[:, ej]).abs()], -1)
    edge_now = p[:, ej] - p[:, ei]
    edge_t0 = (hist[-1, ej] - hist[-1, ei])[None].expand(t, -1, -1)
    edge_recent = ((hist[-1, ej] - hist[-3, ej]) - (hist[-1, ei] - hist[-3, ei]))[None].expand(t, -1, -1) / 2.0
    edge_long = ((hist[-1, ej] - hist[0, ej]) - (hist[-1, ei] - hist[0, ei]))[None].expand(t, -1, -1) / 6.0
    directed = torch.cat([edge_now, edge_t0, edge_recent, edge_long], -1)
    parts = [hpair, directed]
    direction_slice = slice(hpair.shape[-1], hpair.shape[-1] + directed.shape[-1])
    if method in ("shared", "safe"):
        keep = torch.as_tensor(v["valid"], dtype=torch.bool)
        shared_h = h[:, keep].mean(1, keepdim=True).expand(-1, e, -1)
        shared_velocity = ((hist[-1, keep] - hist[0, keep]) / 6.0).mean(0)
        parts.extend([shared_h, shared_velocity[None, None].expand(t, e, -1)])
    parts.append(torch.arange(t, dtype=torch.float64)[:, None, None].expand(t, e, 1) / 8.0)
    if method == "safe":
        parts.append(v["confidence"][None, :, None].expand(t, -1, -1))
    return torch.cat(parts, -1), direction_slice


def fit_normalizer(videos: dict[str, dict], train_ids: list[str], method: str, train_t: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, slice]:
    parts = []
    direction_slice = None
    for vid in train_ids:
        x, direction_slice = features(videos[vid], method)
        parts.append(x[train_t].reshape(-1, x.shape[-1]))
    joined = torch.cat(parts, 0)
    mean = joined.mean(0)
    std = joined.std(0).clamp_min(1e-6)
    mean[direction_slice] = 0.0
    std[direction_slice] = joined[:, direction_slice].square().mean(0).sqrt().clamp_min(1e-6)
    return mean, std, direction_slice


def predict(v: dict, model: OddEdgeMLP, method: str, t: np.ndarray, mean: torch.Tensor, std: torch.Tensor, gate: float) -> torch.Tensor:
    x, _ = features(v, method)
    proposal = model((x[t] - mean) / std)
    if method == "safe":
        proposal = 0.10 * torch.tanh(proposal / 0.10) * v["confidence"][None, :, None]
        projector = v["projector_weighted"]
    else:
        projector = v["projector_plain"]
    delta = torch.einsum("ne,ted->tnd", projector, proposal)
    return v["p0"][t] + gate * delta


def loss(v: dict, prediction: torch.Tensor, t: np.ndarray, safe: bool) -> torch.Tensor:
    keep = torch.as_tensor(v["valid"], dtype=torch.bool)
    target = v["gt"][t]
    pc = prediction[:, keep] - prediction[:, keep].mean(1, keepdim=True)
    tc = target[:, keep] - target[:, keep].mean(1, keepdim=True)
    ei, ej = v["ei"], v["ej"]
    edge_error = ((prediction[:, ej] - prediction[:, ei]) - (target[:, ej] - target[:, ei])).square().mean(-1)
    if safe:
        weight = v["confidence"]
        edge_loss = (edge_error * weight[None]).sum() / (len(t) * weight.sum())
    else:
        edge_loss = edge_error.mean()
    output = 0.5 * (pc - tc).square().mean() + 0.5 * edge_loss
    if safe:
        output = output + 0.05 * (prediction - v["p0"][t]).square().mean()
    return output


def run(args: argparse.Namespace) -> dict:
    torch.set_num_threads(1)
    videos = {vid: load_video(args.bundle_root / vid, args.anchor_threshold, args.k, args.target) for vid in args.video_ids}
    train_t = np.arange(7)
    val_t = np.arange(7, 9)
    test_t = np.arange(9)
    rows = []
    for held_out in args.video_ids:
        train_ids = [vid for vid in args.video_ids if vid != held_out]
        for method in METHODS:
            mean, std, directed = fit_normalizer(videos, train_ids, method, train_t)
            for seed in range(args.seeds):
                torch.manual_seed(9000 + seed)
                model = OddEdgeMLP(len(mean), directed).double()
                # Test antisymmetry and exact zero correction at initialization.
                probe, _ = features(videos[train_ids[0]], method)
                probe = (probe[:1, :1] - mean) / std
                reversed_probe = probe.clone()
                reversed_probe[..., directed] *= -1
                with torch.no_grad():
                    if float(model(probe).abs().max()) != 0.0 or float((model(probe) + model(reversed_probe)).abs().max()) > 1e-12:
                        raise AssertionError("zero correction or edge antisymmetry failed")
                opt = torch.optim.Adam(model.parameters(), lr=args.lr)
                best_state = copy.deepcopy(model.state_dict())
                with torch.no_grad():
                    best_val = np.mean([float(loss(videos[vid], predict(videos[vid], model, method, val_t, mean, std, 1.0), val_t, method == "safe")) for vid in train_ids])
                for step in range(1, args.steps + 1):
                    opt.zero_grad()
                    total = sum(loss(videos[vid], predict(videos[vid], model, method, train_t, mean, std, 1.0), train_t, method == "safe") for vid in train_ids) / len(train_ids)
                    total.backward()
                    opt.step()
                    if step % 10 == 0 or step == args.steps:
                        with torch.no_grad():
                            val = np.mean([float(loss(videos[vid], predict(videos[vid], model, method, val_t, mean, std, 1.0), val_t, method == "safe")) for vid in train_ids])
                        if val < best_val:
                            best_val = val
                            best_state = copy.deepcopy(model.state_dict())
                model.load_state_dict(best_state)
                with torch.no_grad():
                    gate_scores = {}
                    for gate in GATES:
                        gate_scores[str(gate)] = float(np.mean([float(loss(videos[vid], predict(videos[vid], model, method, val_t, mean, std, gate), val_t, method == "safe")) for vid in train_ids]))
                    gate = min(GATES, key=lambda g: gate_scores[str(g)])
                    v = videos[held_out]
                    full = metric(predict(v, model, method, test_t, mean, std, 1.0), v["gt"], v["edges_all"], v["scale"], v["valid"])
                    selected = metric(predict(v, model, method, test_t, mean, std, gate), v["gt"], v["edges_all"], v["scale"], v["valid"])
                rows.append({"held_out_video": held_out, "method": method, "seed": seed, "selected_gate": gate, "gate_validation_scores": gate_scores, "best_validation_loss": best_val, "test_full": full, "test_selected": selected})
                print(json.dumps({"video": held_out[:4], "method": method, "seed": seed, "gate": gate, "centered": selected["centered_error_normalized"], "edge": selected["edge_vector_error_normalized"]}), flush=True)
    baseline = {vid: metric(v["p0"], v["gt"], v["edges_all"], v["scale"], v["valid"]) for vid, v in videos.items()}
    keys = ("raw_ade_m", "centered_error_normalized", "edge_vector_error_normalized", "edge_relative_error", "edge_shape_relative_error")
    aggregate = {"baseline": {key: float(np.median([base[key] for base in baseline.values()])) for key in keys}}
    for method in METHODS:
        aggregate[method] = {}
        for variant in ("test_full", "test_selected"):
            aggregate[method][variant] = {key: float(np.median([row[variant][key] for row in rows if row["method"] == method])) for key in keys}
            aggregate[method][variant]["wins_vs_baseline_per_video_centered"] = int(sum(bool(np.median([row[variant]["centered_error_normalized"] for row in rows if row["method"] == method and row["held_out_video"] == vid]) < baseline[vid]["centered_error_normalized"]) for vid in args.video_ids))
            aggregate[method][variant]["wins_vs_baseline_per_video_edge"] = int(sum(bool(np.median([row[variant]["edge_vector_error_normalized"] for row in rows if row["method"] == method and row["held_out_video"] == vid]) < baseline[vid]["edge_vector_error_normalized"]) for vid in args.video_ids))
    return {
        "protocol": {"video_ids": args.video_ids, "graph": f"2D symmetric kNN k={args.k}", "train_slots": train_t.tolist(), "validation_slots": val_t.tolist(), "held_out_test_slots": test_t.tolist(), "seeds": args.seeds, "steps": args.steps, "gates": GATES, "safe_edge_bound_normalized": 0.10, "safe_regularization": 0.05, "feature_source": "observed7_trackcraft_dense, observed7_user, frozen p0_h cache", "target_source": "TrackCraft3R future pseudo-GT", "target_mode": args.target, "held_out_video_used_for_selection": False},
        "video_diagnostics": {vid: {"valid_queries": int(v["valid"].sum()), "edges": len(v["edges"]), "confidence_min": float(v["confidence"].min()), "confidence_median": float(v["confidence"].median()), "confidence_below_half": int((v["confidence"] < 0.5).sum()), "observed_t0_match_max_m": v["t0_difference_m"], "teacher_t0_fit_median_m": v["t0_teacher_fit_median_m"]} for vid, v in videos.items()},
        "baseline_per_video": baseline,
        "aggregate": aggregate,
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--video-ids", nargs="+", required=True)
    ap.add_argument("--anchor-threshold", type=float, default=5.0)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--target", choices=("aligned", "anchored_displacement"), default="aligned")
    args = ap.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["aggregate"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
