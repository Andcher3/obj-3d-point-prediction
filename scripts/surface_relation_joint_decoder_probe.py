#!/usr/bin/env python3
"""Exploratory four-video test of a joint relational trajectory decoder.

Query graph and all inference features are observed-only or from the frozen
MotionForesight prediction. Teacher validity affects loss/evaluation only.
No test-label gate, edge integration, or baseline-selecting checkpoint exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from surface_relation_f2_gate_transfer import node_features
from surface_relation_f2_leave_one_video_out import metric
from surface_relation_f2_observed_relations import load_video


class JointDecoder(nn.Module):
    """Same-size graph or no-neighbor decoder; both mix all predicted slots."""

    def __init__(self, relation: bool, width: int = 16):
        super().__init__()
        self.relation = relation
        self.width = width
        self.encode = nn.Linear(29, width)
        self.context = nn.Linear(3 * width + 6, width)
        self.temporal = nn.Conv1d(2 * width, 2 * width, 3, padding=1)
        self.readout = nn.Sequential(
            nn.Linear(2 * width, width), nn.Tanh(), nn.Linear(width, 3)
        )
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, v: dict, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.encode((v["x"] - mean) / std))
        t, n, w = x.shape
        global_state = x.mean(1, keepdim=True).expand(-1, n, -1)
        if self.relation:
            ei, ej = v["ei"], v["ej"]
            baseline_edge = v["p0"][:, ej] - v["p0"][:, ei]
            observed_edge = (v["history"][-1, ej] - v["history"][-1, ei])[None].expand(t, -1, -1)
            global_edge = global_state[:, ei]
            input_i = torch.cat([x[:, ei], x[:, ej], global_edge, baseline_edge, observed_edge], -1)
            input_j = torch.cat([x[:, ej], x[:, ei], global_edge, -baseline_edge, -observed_edge], -1)
            msg_i = torch.tanh(self.context(input_i))
            msg_j = torch.tanh(self.context(input_j))
            aggregated = x.new_zeros((t, n, w))
            aggregated = aggregated.index_add(1, ei, msg_i).index_add(1, ej, msg_j)
            degree = torch.bincount(torch.cat([ei, ej]), minlength=n).clamp_min(1)
            message = aggregated / degree[None, :, None]
        else:
            # Equal parameter count; uses point-local future summary instead of neighbors.
            local_summary = x.mean(0, keepdim=True).expand(t, -1, -1)
            centered_now = v["p0"] - v["p0"].mean(1, keepdim=True)
            observed = v["history"][-1]
            centered_observed = (observed - observed.mean(0, keepdim=True))[None].expand(t, -1, -1)
            message = torch.tanh(self.context(torch.cat([
                x, local_summary, global_state, centered_now, centered_observed,
            ], -1)))
        fused = torch.cat([x, message], -1).permute(1, 2, 0)
        temporal = torch.tanh(self.temporal(fused)).permute(2, 0, 1)
        residual = self.readout(temporal)
        residual = residual - residual.mean(1, keepdim=True)
        return v["p0"] + residual


def supervised_loss(v: dict, pred: torch.Tensor, slots: np.ndarray) -> torch.Tensor:
    """Future visibility filters the labels, never graph or readout features."""
    valid = torch.as_tensor(v["valid"], dtype=torch.bool)
    target = v["gt"]
    evalid = valid[v["ei"]] & valid[v["ej"]]
    ei, ej = v["ei"][evalid], v["ej"][evalid]
    if valid.sum() < 40 or len(ei) < 50:
        raise ValueError(f"insufficient supervised support: {v['id']}")
    a, b = pred[slots][:, valid], target[slots][:, valid]
    a = a - a.mean(1, keepdim=True)
    b = b - b.mean(1, keepdim=True)
    edge_pred = pred[slots][:, ej] - pred[slots][:, ei]
    edge_true = target[slots][:, ej] - target[slots][:, ei]
    return 0.5 * (a - b).square().mean() + 0.5 * (edge_pred - edge_true).square().mean()


def load_observed_only_videos(root: Path, ids: list[str]) -> dict[str, dict]:
    videos = {}
    for vid in ids:
        v = load_video(root / vid, 5., 3, "anchored_displacement")
        # The legacy F2 loader prunes inference edges using future label
        # validity. Restore the entire observed-only graph before any forward.
        all_edges = v["edges_all"]
        v["edges"] = all_edges
        v["ei"] = torch.as_tensor([i for i, _ in all_edges], dtype=torch.long)
        v["ej"] = torch.as_tensor([j for _, j in all_edges], dtype=torch.long)
        v["x"] = node_features(v)
        if v["x"].shape != (9, 48, 29) or not torch.isfinite(v["x"]).all():
            raise ValueError(f"invalid observed-only node features: {vid}")
        videos[vid] = v
    return videos


def permutation_probe(model: JointDecoder, v: dict, mean: torch.Tensor, std: torch.Tensor) -> dict:
    rng = np.random.default_rng(837)
    order = rng.permutation(48)
    inverse = np.argsort(order)
    view = dict(v)
    for k in ("x", "p0", "history", "h", "gt"):
        if k == "history":
            view[k] = v[k][:, order]
        else:
            view[k] = v[k][:, order]
    view["ei"] = torch.as_tensor(inverse[v["ei"].numpy()], dtype=torch.long)
    view["ej"] = torch.as_tensor(inverse[v["ej"].numpy()], dtype=torch.long)
    # Future-label validity is intentionally unavailable in the forward.
    view["valid"] = np.logical_not(v["valid"])
    with torch.no_grad():
        original = model(v, mean, std)
        permuted = model(view, mean, std)
    error = float((permuted - original[:, order]).abs().max())
    validity_error = float((model(v, mean, std) - model({**v, "valid": np.logical_not(v["valid"])}, mean, std)).abs().max())
    if error > 1e-10 or validity_error != 0:
        raise AssertionError(f"permutation/future-validity independence failed: {error}, {validity_error}")
    return {"permutation_max_error": error, "future_validity_forward_max_error": validity_error}


def train_fold(videos: dict[str, dict], ids: list[str], held_out: str, relation: bool,
               seed: int, width: int, steps: int, lr: float) -> tuple[dict, dict]:
    train_ids = [vid for vid in ids if vid != held_out]
    training = torch.cat([videos[vid]["x"][:7].reshape(-1, 29) for vid in train_ids], 0)
    mean, std = training.mean(0), training.std(0).clamp_min(1e-6)
    torch.manual_seed(seed)
    model = JointDecoder(relation, width).double()
    zero = float((model(videos[train_ids[0]], mean, std) - videos[train_ids[0]]["p0"]).abs().max())
    if zero != 0:
        raise AssertionError("model must start at the exact baseline")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_slots = np.arange(7)
    val_slots = np.arange(7, 9)
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        objective = sum(supervised_loss(videos[vid], model(videos[vid], mean, std), train_slots)
                        for vid in train_ids) / len(train_ids)
        if not torch.isfinite(objective):
            raise ValueError(f"non-finite loss: {held_out} {relation} {seed} {step}")
        objective.backward()
        if step == 1 and not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("non-finite or missing gradient")
        optimizer.step()
    with torch.no_grad():
        train_loss = float(np.mean([float(supervised_loss(videos[vid], model(videos[vid], mean, std), train_slots)) for vid in train_ids]))
        val_loss = float(np.mean([float(supervised_loss(videos[vid], model(videos[vid], mean, std), val_slots)) for vid in train_ids]))
        held = videos[held_out]
        pred = model(held, mean, std)
        probe = permutation_probe(model, held, mean, std)
        measured = metric(pred, held["gt"], held["edges_all"], held["scale"], held["valid"])
        baseline = metric(held["p0"], held["gt"], held["edges_all"], held["scale"], held["valid"])
    if not all(np.isfinite(z) for z in measured.values() if isinstance(z, (int, float))):
        raise ValueError(f"non-finite test metrics: {held_out}")
    return {
        "held_out": held_out, "method": "joint_relational" if relation else "joint_node_control",
        "seed": seed, "train_loss": train_loss, "validation_loss": val_loss,
        "parameters": sum(p.numel() for p in model.parameters()),
        "baseline": baseline, "test": measured, "zero_initial_error": zero,
        **probe,
    }, {"state": model.state_dict(), "mean": mean, "std": std}


def aggregate(rows: list[dict], ids: list[str]) -> dict:
    metrics = ("raw_ade_m", "centered_error_normalized", "edge_vector_error_normalized")
    summary = {}
    for vid in ids:
        b = next(r["baseline"] for r in rows if r["held_out"] == vid)
        summary[vid] = {"baseline": {m: b[m] for m in metrics}}
        for method in ("joint_relational", "joint_node_control"):
            subset = [r for r in rows if r["held_out"] == vid and r["method"] == method]
            summary[vid][method] = {m: float(np.median([r["test"][m] for r in subset])) for m in metrics}
            summary[vid][method]["wins_both_vs_baseline"] = all(summary[vid][method][m] < b[m] for m in metrics[1:])
    signals = []
    for vid in ids:
        x = summary[vid]
        rel, node, base = x["joint_relational"], x["joint_node_control"], x["baseline"]
        signals.append(all(rel[m] < min(base[m], node[m]) for m in metrics[1:]) and rel["raw_ade_m"] <= 1.05 * base["raw_ade_m"])
    return {"per_video": summary, "relational_both_structural_and_raw_safety_wins": int(sum(signals)),
            "exploratory_signal_pass": bool(sum(signals) >= 3)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    spec = json.loads(args.protocol.read_text(encoding="utf-8"))
    torch.set_num_threads(1)
    ids = spec["video_ids"]
    videos = load_observed_only_videos(Path(spec["data_root"]), ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    states = {}
    for held_out in ids:
        for relation in (False, True):
            for seed in spec["seeds"]:
                record, state = train_fold(videos, ids, held_out, relation, seed,
                                           spec["hidden_width"], spec["training_steps"], spec["learning_rate"])
                rows.append(record)
                states[f"{held_out}_{record['method']}_{seed}"] = state
                print(json.dumps({"video": held_out[:4], "method": record["method"], "seed": seed,
                                  "centered": record["test"]["centered_error_normalized"],
                                  "edge": record["test"]["edge_vector_error_normalized"]}), flush=True)
    comparison = aggregate(rows, ids)
    result = {"status": "COMPLETED_EXPLORATORY", "protocol": str(args.protocol),
              "training": "full 400 steps; no checkpoint/gate selected on labels",
              "data_audit": {vid: {"observed_graph_edges": len(videos[vid]["edges"]),
                                   "valid_supervision_queries": int(videos[vid]["valid"].sum()),
                                   "observed_t0_match_max_m": videos[vid]["t0_difference_m"]} for vid in ids},
              "comparison": comparison, "rows": rows, "limits": spec["known_limits"]}
    torch.save(states, args.output_dir / "trained_folds.pt")
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "comparison": comparison}), flush=True)


if __name__ == "__main__":
    main()
