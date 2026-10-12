#!/usr/bin/env python3
"""Smoke-test variable query counts at the sampler and optional model processor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def sampler_smoke(counts: list[int]) -> dict:
    height, width = 96, 128
    yy, xx = np.mgrid[:height, :width]
    polygon_mask = ((xx - 64) / 40.0) ** 2 + ((yy - 48) / 30.0) ** 2 <= 1.0
    candidate = polygon_mask.copy()
    from surface_relation_formal_e2_prepare import deterministic_fps

    order = deterministic_fps(candidate, candidate, max(counts), 4.0)
    results = {str(count): {"available": len(order) >= count, "selected_count": min(count, len(order))} for count in counts}
    return {"mode": "sampler_only", "results": results}


def processor_smoke(model_name: str, repo: str, counts: list[int]) -> dict:
    import torch
    from PIL import Image

    sys.path.insert(0, str(Path(repo) / "src"))
    from molmo_motion import MolmoMotionProcessor

    processor = MolmoMotionProcessor.from_pretrained(model_name)
    frames = [Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8) + 127) for _ in range(3)]
    records = {}
    for count in counts:
        try:
            points_2d = torch.zeros((count, 2), dtype=torch.float32)
            points_3d = torch.zeros((3, count, 3), dtype=torch.float32)
            points_3d[..., 2] = 1.0
            encoded = processor(
                history_frames=frames,
                points_2d_at_t0=points_2d,
                points_3d_history=points_3d,
                action="",
                future_horizon=9,
            )
            tensor_shapes = {key: list(value.shape) for key, value in encoded.items() if torch.is_tensor(value)}
            records[str(count)] = {"accepted": True, "tensor_shapes": tensor_shapes}
        except Exception as exc:
            records[str(count)] = {"accepted": False, "error": repr(exc)}
    return {"mode": "processor", "model": model_name, "results": records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", type=int, nargs="+", default=[16, 48, 64])
    parser.add_argument("--model")
    parser.add_argument("--repo")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.model is None) != (args.repo is None):
        raise SystemExit("--model and --repo must be supplied together")
    payload = sampler_smoke(args.counts)
    if args.model:
        try:
            payload["processor"] = processor_smoke(args.model, args.repo, args.counts)
        except Exception as exc:
            payload["processor"] = {"mode": "processor", "accepted": False, "error": repr(exc)}
    else:
        payload["processor"] = {"mode": "processor", "status": "not_run", "reason": "checkpoint_and_repo_not_provided"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
