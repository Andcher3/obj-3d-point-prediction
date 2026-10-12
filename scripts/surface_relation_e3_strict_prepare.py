#!/usr/bin/env python3
"""Prepare a consecutive 15-fps DA3 input for strict E3 pseudo-GT audit."""
from __future__ import annotations

import argparse, json
from pathlib import Path
import cv2
import numpy as np
import torch

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--frames-dir', type=Path, required=True)
    p.add_argument('--frame-indices', type=int, nargs='+', required=True)
    p.add_argument('--clip-start-seconds', type=float, required=True)
    p.add_argument('--fps', type=float, default=15.0)
    p.add_argument('--output-npz', type=Path, required=True)
    p.add_argument('--output-json', type=Path, required=True)
    p.add_argument('--process-res', type=int, default=518)
    args = p.parse_args()
    indices = np.asarray(args.frame_indices, dtype=np.int32)
    if len(indices) < 2 or not np.all(np.diff(indices) == 1):
        raise ValueError('frame indices must be consecutive')
    from depth_anything_3.api import DepthAnything3
    frames = [np.load(args.frames_dir / f'{i:04d}.npy') for i in indices]
    model = DepthAnything3.from_pretrained('depth-anything/DA3-BASE').to('cuda').eval()
    with torch.no_grad(): pred = model.inference(frames, process_res=args.process_res)
    rgb = np.asarray(pred.processed_images, dtype=np.uint8)
    depth = np.asarray(pred.depth, dtype=np.float32)
    intr = np.asarray(pred.intrinsics, dtype=np.float64)
    ext = np.asarray(pred.extrinsics, dtype=np.float64)
    w2c = np.tile(np.eye(4, dtype=np.float64), (len(indices), 1, 1))
    w2c[:, :3, :4] = ext
    first_c2w = np.linalg.inv(w2c[0]); w2c = np.stack([x @ first_c2w for x in w2c]).astype(np.float32)
    h, w = depth.shape[1:]
    if rgb.shape[1:3] != (h, w):
        rgb = np.stack([cv2.resize(x, (w, h), interpolation=cv2.INTER_AREA) for x in rgb])
    k = intr[0]
    jpeg = np.asarray([cv2.imencode('.jpg', cv2.cvtColor(x, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tobytes() for x in rgb], dtype=object)
    ts = args.clip_start_seconds + indices.astype(np.float64) / args.fps
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, images_jpeg_bytes=jpeg, rgb=rgb, depth_map=depth, extrinsics_w2c=w2c, fx_fy_cx_cy=np.asarray([k[0,0],k[1,1],k[0,2],k[1,2]]), frame_indices=indices, timestamps_s=ts)
    payload={'status':'PASS','frame_indices':indices.tolist(),'timestamps_s':ts.tolist(),'fps':args.fps,'rgb_shape':list(rgb.shape),'depth_shape':list(depth.shape),'future_frames_supplied':False}
    args.output_json.parent.mkdir(parents=True, exist_ok=True); args.output_json.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8'); print(json.dumps(payload,indent=2))
if __name__ == '__main__': main()
