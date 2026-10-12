#!/usr/bin/env python3
"""Prepare a contiguous 23-frame offline-only DA3 context."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from prepare_davis_joint_probe_inputs import save_user_npz


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=14)
    parser.add_argument("--process-res", type=int, default=392)
    args = parser.parse_args()
    indices = np.arange(args.start_index, args.start_index + 23, dtype=np.int32)
    frames = []
    for index in indices:
        bgr = cv2.imread(str(args.frames_dir / f"{index:05d}.jpg"), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(args.frames_dir / f"{index:05d}.jpg")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained("depth-anything/DA3-BASE").to("cuda").eval()
    with torch.no_grad():
        prediction = model.inference(frames, process_res=args.process_res)
    shapes = save_user_npz(args.output, prediction, indices, observed_count=7)
    print(json.dumps({
        "status": "OFFLINE_FULL23_ONLY", "sequence": args.sequence,
        "source_frame_indices": indices.tolist(),
        "extra_context_source_frame": int(indices[-1]),
        "evaluation_source_frames": indices[:-1].tolist(), "shapes": shapes,
        "future_frames_allowed_as_prediction_condition": False,
    }, indent=2))


if __name__ == "__main__":
    main()
