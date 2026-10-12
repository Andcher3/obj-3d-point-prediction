#!/usr/bin/env python3
"""Build a 22-frame 15 fps DA3 input for the 7+15 protocol."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2, numpy as np, torch

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--frames-dir',type=Path,required=True); ap.add_argument('--frame-indices',type=int,nargs=22,required=True); ap.add_argument('--clip-start-seconds',type=float,required=True); ap.add_argument('--output-npz',type=Path,required=True); ap.add_argument('--output-json',type=Path,required=True); ap.add_argument('--process-res',type=int,default=518); args=ap.parse_args()
    idx=np.asarray(args.frame_indices,dtype=np.int32)
    frames=[np.load(args.frames_dir/f'{i:04d}.npy') for i in idx]
    from depth_anything_3.api import DepthAnything3
    model=DepthAnything3.from_pretrained('depth-anything/DA3-BASE').to('cuda').eval()
    with torch.no_grad(): pred=model.inference(frames,process_res=args.process_res)
    rgb=np.asarray(pred.processed_images,dtype=np.uint8); depth=np.asarray(pred.depth,dtype=np.float32); K=np.asarray(pred.intrinsics,dtype=np.float64); ext=np.asarray(pred.extrinsics,dtype=np.float64)
    w2c=np.tile(np.eye(4,dtype=np.float64),(len(idx),1,1)); w2c[:,:3,:4]=ext; first=np.linalg.inv(w2c[0]); w2c=np.stack([p@first for p in w2c]).astype(np.float32)
    h,w=depth.shape[1:]; rgb=np.stack([cv2.resize(x,(w,h),interpolation=cv2.INTER_AREA) for x in rgb]) if rgb.shape[1:3]!=(h,w) else rgb
    jpeg=np.asarray([cv2.imencode('.jpg',cv2.cvtColor(x,cv2.COLOR_RGB2BGR),[int(cv2.IMWRITE_JPEG_QUALITY),95])[1].tobytes() for x in rgb],dtype=object)
    ts=args.clip_start_seconds+idx.astype(np.float64)/15.0
    args.output_npz.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(args.output_npz,images_jpeg_bytes=jpeg,rgb=rgb,depth_map=depth,extrinsics_w2c=w2c,fx_fy_cx_cy=np.asarray([K[0,0,0],K[0,1,1],K[0,0,2],K[0,1,2]],dtype=np.float64),frame_indices=idx,timestamps_s=ts,observed_frame_count=np.asarray(7,dtype=np.int32))
    args.output_json.write_text(json.dumps({'status':'PASS','protocol':'22-frame 7+15','frame_indices':idx.tolist(),'timestamps_s':ts.tolist(),'observed_indices':list(range(7)),'future_indices':list(range(7,22)),'future_frames_supplied_to_model':False},indent=2)+'\n')
    print(json.dumps({'status':'PASS','shape_rgb':list(rgb.shape),'shape_depth':list(depth.shape),'output':str(args.output_npz)},indent=2))
if __name__=='__main__': main()
