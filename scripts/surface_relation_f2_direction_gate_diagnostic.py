#!/usr/bin/env python3
"""Replay anchored F2 and diagnose edge-head direction versus correction strength.

The oracle gate uses held-out pseudo-labels *only after* training, as a
diagnostic upper bound. It is not a deployable prediction or model score.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from surface_relation_f2_leave_one_video_out import metric
from surface_relation_f2_observed_relations import (
    GATES,
    METHODS,
    OddEdgeMLP,
    features,
    fit_normalizer,
    load_video,
    loss,
    predict,
)


def direction_stats(v: dict, correction: torch.Tensor) -> dict:
    """Use the same unweighted 50/50 centered/edge MSE for all three heads."""
    p = v["p0"]
    r = v["gt"] - p
    mask = torch.as_tensor(v["valid"], dtype=torch.bool)
    ei, ej = v["ei"], v["ej"]
    dc = correction[:, mask] - correction[:, mask].mean(1, keepdim=True)
    rc = r[:, mask] - r[:, mask].mean(1, keepdim=True)
    de = correction[:, ej] - correction[:, ei]
    re = r[:, ej] - r[:, ei]

    def optimum(dcs: torch.Tensor, rcs: torch.Tensor, des: torch.Tensor, res: torch.Tensor) -> dict:
        terms = [
            ("centered", dcs, rcs),
            ("edge", des, res),
        ]
        scores = {}
        for name, d, target in terms:
            dd = float(d.square().mean())
            rr = float(target.square().mean())
            dot = float((d * target).mean())
            gate = float(np.clip(dot / dd, 0.0, 1.0)) if dd > 1e-20 else 0.0
            cos = float(dot / np.sqrt(dd * rr)) if dd > 1e-20 and rr > 1e-20 else None
            benefit = (2.0 * gate * dot - gate * gate * dd) / rr if rr > 1e-20 else 0.0
            scores[name] = {
                "oracle_gate": gate,
                "cosine": cos,
                "correction_to_needed_rms": float(np.sqrt(dd / rr)) if rr > 1e-20 else None,
                "oracle_squared_error_reduction_fraction": float(benefit),
                "full_gate_squared_error_reduction_fraction": float((2 * dot - dd) / rr) if rr > 1e-20 else 0.0,
                "dot": dot,
                "correction_energy": dd,
                "target_energy": rr,
            }
        dd = 0.5 * (scores["centered"]["correction_energy"] + scores["edge"]["correction_energy"])
        rr = 0.5 * (scores["centered"]["target_energy"] + scores["edge"]["target_energy"])
        dot = 0.5 * (scores["centered"]["dot"] + scores["edge"]["dot"])
        gate = float(np.clip(dot / dd, 0.0, 1.0)) if dd > 1e-20 else 0.0
        cos = float(dot / np.sqrt(dd * rr)) if dd > 1e-20 and rr > 1e-20 else None
        scores["combined"] = {
            "oracle_gate": gate,
            "cosine": cos,
            "correction_to_needed_rms": float(np.sqrt(dd / rr)) if rr > 1e-20 else None,
            "oracle_squared_error_reduction_fraction": float((2 * gate * dot - gate * gate * dd) / rr) if rr > 1e-20 else 0.0,
            "full_gate_squared_error_reduction_fraction": float((2 * dot - dd) / rr) if rr > 1e-20 else 0.0,
            "dot": dot,
            "correction_energy": dd,
            "target_energy": rr,
        }
        return scores

    summary = optimum(dc, rc, de, re)
    slots = [optimum(dc[t], rc[t], de[t], re[t]) for t in range(len(p))]
    return {"video": summary, "slots": slots}


def run(args: argparse.Namespace) -> dict:
    torch.set_num_threads(1)
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    proto = reference["protocol"]
    if proto["target_mode"] != "anchored_displacement" or proto["video_ids"] != args.video_ids:
        raise ValueError("reference protocol or ordered video IDs do not match")
    if (proto["steps"], proto["seeds"], proto["graph"]) != (args.steps, args.seeds, f"2D symmetric kNN k={args.k}"):
        raise ValueError("reference F2 training settings do not match")
    videos = {vid: load_video(args.bundle_root / vid, args.anchor_threshold, args.k, "anchored_displacement") for vid in args.video_ids}
    train_t, val_t, test_t = np.arange(7), np.arange(7, 9), np.arange(9)
    expected = {(row["held_out_video"], row["method"], row["seed"]): row for row in reference["rows"]}
    deltas = np.empty((len(args.video_ids), len(METHODS), args.seeds, 9, 48, 3), dtype=np.float64)
    rows: list[dict] = []
    maximum_replay_error = 0.0
    for vidx, held_out in enumerate(args.video_ids):
        train_ids = [vid for vid in args.video_ids if vid != held_out]
        for midx, method in enumerate(METHODS):
            mean, std, directed = fit_normalizer(videos, train_ids, method, train_t)
            for seed in range(args.seeds):
                torch.manual_seed(9000 + seed)
                model = OddEdgeMLP(len(mean), directed).double()
                opt = torch.optim.Adam(model.parameters(), lr=args.lr)
                best_state = copy.deepcopy(model.state_dict())
                with torch.no_grad():
                    best_val = np.mean([
                        float(loss(videos[vid], predict(videos[vid], model, method, val_t, mean, std, 1.0), val_t, method == "safe"))
                        for vid in train_ids
                    ])
                for step in range(1, args.steps + 1):
                    opt.zero_grad()
                    total = sum(
                        loss(videos[vid], predict(videos[vid], model, method, train_t, mean, std, 1.0), train_t, method == "safe")
                        for vid in train_ids
                    ) / len(train_ids)
                    total.backward()
                    opt.step()
                    if step % 10 == 0 or step == args.steps:
                        with torch.no_grad():
                            val = np.mean([
                                float(loss(videos[vid], predict(videos[vid], model, method, val_t, mean, std, 1.0), val_t, method == "safe"))
                                for vid in train_ids
                            ])
                        if val < best_val:
                            best_val = val
                            best_state = copy.deepcopy(model.state_dict())
                model.load_state_dict(best_state)
                with torch.no_grad():
                    gate_scores = {
                        str(g): float(np.mean([
                            float(loss(videos[vid], predict(videos[vid], model, method, val_t, mean, std, g), val_t, method == "safe"))
                            for vid in train_ids
                        ])) for g in GATES
                    }
                    selected_gate = min(GATES, key=lambda g: gate_scores[str(g)])
                    v = videos[held_out]
                    full = predict(v, model, method, test_t, mean, std, 1.0)
                    delta = full - v["p0"]
                    stats = direction_stats(v, delta)
                    oracle_gate = stats["video"]["combined"]["oracle_gate"]
                    actual_full = metric(full, v["gt"], v["edges_all"], v["scale"], v["valid"])
                    actual_selected = metric(v["p0"] + selected_gate * delta, v["gt"], v["edges_all"], v["scale"], v["valid"])
                    oracle_diagnostic = metric(v["p0"] + oracle_gate * delta, v["gt"], v["edges_all"], v["scale"], v["valid"])
                prior = expected[(held_out, method, seed)]
                if selected_gate != prior["selected_gate"]:
                    raise AssertionError(f"replay gate differs for {held_out}/{method}/{seed}")
                for label, actual, saved in (("full", actual_full, prior["test_full"]), ("selected", actual_selected, prior["test_selected"])):
                    for key in ("centered_error_normalized", "edge_vector_error_normalized", "raw_ade_m"):
                        error = abs(actual[key] - saved[key])
                        maximum_replay_error = max(maximum_replay_error, error)
                        if not np.isclose(actual[key], saved[key], atol=1e-8, rtol=1e-7):
                            raise AssertionError(f"replay {label}/{key} differs for {held_out}/{method}/{seed}: {actual[key]} vs {saved[key]}")
                deltas[vidx, midx, seed] = delta.numpy()
                rows.append({
                    "held_out_video": held_out,
                    "method": method,
                    "seed": seed,
                    "selected_gate": selected_gate,
                    "direction_and_oracle": stats,
                    "test_full": actual_full,
                    "test_selected": actual_selected,
                    "oracle_gate_diagnostic_only": oracle_diagnostic,
                })
                print(json.dumps({"held_out": held_out[:4], "method": method, "seed": seed, "cosine": stats["video"]["combined"]["cosine"], "oracle_gate": oracle_gate}), flush=True)

    per_video = {}
    for vid in args.video_ids:
        per_video[vid] = {}
        for method in METHODS:
            subset = [r for r in rows if r["held_out_video"] == vid and r["method"] == method]
            per_video[vid][method] = {
                "median_combined_cosine": float(np.median([r["direction_and_oracle"]["video"]["combined"]["cosine"] for r in subset])),
                "median_edge_cosine": float(np.median([r["direction_and_oracle"]["video"]["edge"]["cosine"] for r in subset])),
                "median_oracle_gate": float(np.median([r["direction_and_oracle"]["video"]["combined"]["oracle_gate"] for r in subset])),
                "zero_gate_seeds": int(sum(r["direction_and_oracle"]["video"]["combined"]["oracle_gate"] == 0 for r in subset)),
                "median_oracle_squared_error_reduction_fraction": float(np.median([r["direction_and_oracle"]["video"]["combined"]["oracle_squared_error_reduction_fraction"] for r in subset])),
                "median_full_centered": float(np.median([r["test_full"]["centered_error_normalized"] for r in subset])),
                "median_full_edge": float(np.median([r["test_full"]["edge_vector_error_normalized"] for r in subset])),
                "median_oracle_centered_diagnostic_only": float(np.median([r["oracle_gate_diagnostic_only"]["centered_error_normalized"] for r in subset])),
                "median_oracle_edge_diagnostic_only": float(np.median([r["oracle_gate_diagnostic_only"]["edge_vector_error_normalized"] for r in subset])),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.arrays_output,
        video_ids=np.asarray(args.video_ids),
        methods=np.asarray(METHODS),
        corrections=deltas,
        baseline=np.stack([videos[vid]["p0"].numpy() for vid in args.video_ids]),
        anchored_pseudo_gt=np.stack([videos[vid]["gt"].numpy() for vid in args.video_ids]),
        valid=np.stack([videos[vid]["valid"] for vid in args.video_ids]),
        scales=np.asarray([videos[vid]["scale"] for vid in args.video_ids]),
    )
    result = {
        "status": "COMPLETED",
        "scope": "F2 anchored-displacement direction and post-hoc oracle-gate diagnosis",
        "oracle_warning": "Held-out pseudo-labels select oracle gates only for diagnosis. Oracle metrics are not deployable or a new method score.",
        "protocol": {
            "reference": str(args.reference), "video_ids": args.video_ids,
            "methods": METHODS, "seeds": args.seeds, "steps": args.steps, "learning_rate": args.lr,
            "anchor_threshold_px": args.anchor_threshold, "graph_k": args.k,
            "target": "anchored_displacement", "held_out_used_for_training_or_checkpoint": False,
            "gate_objective": "0.5 centered coordinate MSE + 0.5 edge-vector MSE, equal weighting across methods",
            "gate_scope": "single scalar per held-out video and seed; per-slot gates also diagnostic",
            "gate_range": [0.0, 1.0],
            "arrays": str(args.arrays_output),
        },
        "max_replayed_metric_abs_error": maximum_replay_error,
        "baseline_per_video": reference["baseline_per_video"],
        "per_video": per_video,
        "rows": rows,
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETED", "max_replayed_metric_abs_error": maximum_replay_error, "per_video": per_video}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arrays-output", type=Path, required=True)
    parser.add_argument("--video-ids", nargs="+", required=True)
    parser.add_argument("--anchor-threshold", type=float, default=5.0)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    arguments = parser.parse_args()
    run(arguments)


if __name__ == "__main__":
    main()
