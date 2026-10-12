#!/usr/bin/env python3
"""Probe MolmoMotion sensitivity to action-text form on bundled valid inputs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


TEXT_VARIANTS = {
    "egodex_clean_surface": {
        "official": "Roll a lint roller on a blue cloth",
        "compact": "rolling a lint roller",
        "free_sentence": (
            "The person rolls the lint roller back and forth across the blue cloth "
            "to clean its surface."
        ),
        "multi_step": (
            "First roll the lint roller across the blue cloth, then lift it, reposition it, "
            "and roll over another area of the cloth."
        ),
        "contradictory": (
            "Lift the lint roller straight up and hold it still without touching the blue cloth."
        ),
    },
    "egodex_ball_base": {
        "official": "Put white ball on white 3D printed base",
        "compact": "placing a ball",
        "free_sentence": (
            "The person places the white ball onto the top of the white 3D-printed base."
        ),
        "multi_step": (
            "Pick up the white ball, move it above the white 3D-printed base, place it on the "
            "base, and then release it."
        ),
        "contradictory": "Move the white ball away from the base and place it to the side.",
    },
    "davis_car_turn": {
        "official": "A silver car follows the road and slowly turns to the right",
        "compact": "turning the car right",
        "free_sentence": (
            "The silver car continues forward along the road and gradually steers around the "
            "right-hand bend."
        ),
        "multi_step": (
            "Continue forward, enter the curve, turn right along the road, and then straighten "
            "the car after the bend."
        ),
        "contradictory": "The silver car stops immediately and reverses to the left.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local MolmoMotion checkpoint directory")
    parser.add_argument("--repo", required=True, help="Official molmo-motion repository")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--examples",
        nargs="+",
        default=["egodex_clean_surface", "egodex_ball_base", "davis_car_turn"],
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["official", "compact", "free_sentence", "multi_step", "contradictory"],
    )
    parser.add_argument("--future-horizon", type=int, default=30)
    return parser.parse_args()


def import_quickstart(repo: Path):
    path = repo / "examples" / "01_quickstart.py"
    spec = importlib.util.spec_from_file_location("molmo_motion_quickstart", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def tensor_to_list(value: torch.Tensor) -> list:
    return value.detach().float().cpu().tolist()


def trajectory_metrics(future: torch.Tensor, t0_xyz: torch.Tensor) -> dict:
    future = future.detach().float().cpu()
    t0_xyz = t0_xyz.detach().float().cpu()
    full = torch.cat([t0_xyz[:, None, :], future], dim=1)
    steps = torch.linalg.vector_norm(full[:, 1:] - full[:, :-1], dim=-1)
    final_delta = future[:, -1] - t0_xyz
    centroid_delta = future[:, -1].mean(0) - t0_xyz.mean(0)
    return {
        "finite_fraction": float(torch.isfinite(future).float().mean()),
        "mean_path_length_m": float(steps.sum(dim=1).mean()),
        "mean_final_displacement_m": float(torch.linalg.vector_norm(final_delta, dim=-1).mean()),
        "centroid_final_delta_m": tensor_to_list(centroid_delta),
    }


def compare(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    ref = reference.detach().float().cpu()
    cand = candidate.detach().float().cpu()
    delta = torch.linalg.vector_norm(cand - ref, dim=-1)
    ref_vec = ref[:, -1].mean(0) - ref[:, 0].mean(0)
    cand_vec = cand[:, -1].mean(0) - cand[:, 0].mean(0)
    denom = torch.linalg.vector_norm(ref_vec) * torch.linalg.vector_norm(cand_vec)
    cosine = float(torch.dot(ref_vec, cand_vec) / denom) if float(denom) > 1e-12 else math.nan
    return {
        "ade_to_official_m": float(delta.mean()),
        "fde_to_official_m": float(delta[:, -1].mean()),
        "centroid_direction_cosine_to_official": cosine,
    }


def plot_comparison(
    output_path: Path,
    t0_image,
    intrinsics: torch.Tensor,
    predictions: dict[str, torch.Tensor],
    project_fn,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    ax.imshow(t0_image)
    colors = plt.cm.tab10(np.linspace(0, 1, len(predictions)))
    k = intrinsics.detach().cpu().numpy()
    for color, (name, prediction) in zip(colors, predictions.items()):
        pixels = project_fn(prediction.detach().float().cpu().numpy(), k)
        centroid = pixels.mean(axis=0)
        ax.plot(centroid[:, 0], centroid[:, 1], color=color, linewidth=2.0, label=name)
        ax.scatter(centroid[-1, 0], centroid[-1, 1], color=color, s=18)
    ax.set_xlim(0, t0_image.width)
    ax.set_ylim(t0_image.height, 0)
    ax.axis("off")
    ax.legend(loc="best", fontsize=7, framealpha=0.8)
    fig.tight_layout(pad=0)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "arrays").mkdir(exist_ok=True)

    sys.path.insert(0, str(repo / "src"))
    quickstart = import_quickstart(repo)
    from molmo_motion import MolmoMotion, MolmoMotionProcessor

    load_started = time.perf_counter()
    processor = MolmoMotionProcessor.from_pretrained(args.model)
    model = MolmoMotion.from_pretrained(args.model)
    model._internal = model._internal.to(torch.bfloat16).cuda()
    model_load_seconds = time.perf_counter() - load_started
    history_size = processor.config.history_size

    records: list[dict] = []
    experiment_predictions: dict[str, dict[str, torch.Tensor]] = {}
    experiment_context: dict[str, tuple] = {}

    for example in args.examples:
        if example not in TEXT_VARIANTS:
            raise KeyError(f"No text variants defined for {example}")
        example_dir = repo / "examples" / "data" / example
        (
            meta,
            history_frames,
            points_2d_at_t0,
            points_3d_history,
            intrinsics,
            bundled_action,
        ) = quickstart.load_example(example_dir, history_size)
        if TEXT_VARIANTS[example]["official"] != bundled_action:
            raise ValueError(f"Official caption drift for {example}: {bundled_action!r}")

        experiment_predictions[example] = {}
        experiment_context[example] = (history_frames[-1], intrinsics)
        for variant in args.variants:
            action = TEXT_VARIANTS[example][variant]
            inputs = processor(
                history_frames=history_frames,
                points_2d_at_t0=points_2d_at_t0,
                points_3d_history=points_3d_history,
                action=action,
                future_horizon=args.future_horizon,
            )
            inputs = {key: value.cuda() if torch.is_tensor(value) else value for key, value in inputs.items()}

            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.predict_trajectory(**inputs)
            elapsed = time.perf_counter() - started
            future = out.future_3d.detach().float().cpu()
            if tuple(future.shape) != (points_2d_at_t0.shape[0], args.future_horizon, 3):
                raise RuntimeError(f"Unexpected output shape for {example}/{variant}: {tuple(future.shape)}")

            np.savez_compressed(
                output_dir / "arrays" / f"{example}__{variant}.npz",
                future_3d=future.numpy(),
                points_3d_history=points_3d_history.numpy(),
                points_2d_at_t0=points_2d_at_t0.numpy(),
                intrinsics=intrinsics.numpy(),
                action=np.asarray(action),
            )
            experiment_predictions[example][variant] = future
            row = {
                "example": example,
                "source_video": meta.get("video"),
                "variant": variant,
                "action": action,
                "action_characters": len(action),
                "output_shape": list(future.shape),
                "inference_seconds": elapsed,
                "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
                **trajectory_metrics(future, points_3d_history[-1]),
            }
            records.append(row)
            print(
                "RESULT "
                + json.dumps(
                    {
                        "example": example,
                        "variant": variant,
                        "shape": list(future.shape),
                        "finite_fraction": row["finite_fraction"],
                        "seconds": round(elapsed, 3),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    for example, predictions in experiment_predictions.items():
        reference = predictions["official"]
        for row in records:
            if row["example"] == example:
                row.update(compare(reference, predictions[row["variant"]]))
        t0_image, intrinsics = experiment_context[example]
        plot_comparison(
            output_dir / f"{example}__text_comparison.png",
            t0_image,
            intrinsics,
            predictions,
            quickstart.project_camera_xyz_to_pixel,
        )

    payload = {
        "checkpoint": str(Path(args.model).resolve()),
        "repository": str(repo),
        "repository_commit": "61f5b21b694ad8f854ec7ecd2400005acc73f685",
        "model_load_seconds": model_load_seconds,
        "future_horizon": args.future_horizon,
        "records": records,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"METRICS={output_dir / 'metrics.json'}")
    print(f"TOTAL_RESULTS={len(records)}")


if __name__ == "__main__":
    main()
