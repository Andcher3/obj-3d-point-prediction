#!/usr/bin/env python3
"""Project official interleaved Pj0-anchored tracks for DAVIS visualization."""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from build_davis_visualization_projection_bundle import project

def cam0_to_frame_matrices(w2c):
    return np.einsum("tij,jk->tik", w2c, np.linalg.inv(w2c[0]))

def main() -> None:
    p=argparse.ArgumentParser();p.add_argument("--video-dir",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();d=a.video_dir
    m=json.loads((d/"query_frame0_manifest.json").read_text());q=np.asarray(m["query_sets"]["48"],np.float32)
    with np.load(d/"observed7_query_tracks.npz") as o,np.load(d/"full22_query_tracks.npz") as t,np.load(d/"observed7_user.npz",allow_pickle=True) as ou,np.load(d/"full23_user.npz",allow_pickle=True) as tu:
        op=np.asarray(o["trackcraft3r_tracks"],float);tp=np.asarray(t["trackcraft3r_tracks"],float)
        ow=np.asarray(ou["extrinsics_w2c"]);tw=np.asarray(tu["extrinsics_w2c"])[:22]
        ouv,ov=project(op,cam0_to_frame_matrices(ow),np.asarray(ou["fx_fy_cx_cy"]))
        tuv,tv=project(tp,cam0_to_frame_matrices(tw),np.asarray(tu["fx_fy_cx_cy"]))
        processed=np.asarray(tu["depth_map"].shape[1:3],np.int32)
    np.savez_compressed(a.output,video_id=np.asarray(m["video_id"]),source_frame_indices=np.arange(14,36,dtype=np.int32),observed_source_frame_indices=np.arange(14,21,dtype=np.int32),query_uv_frame0_processed=q,query_uv_frame0_source=np.asarray(m["query_uv_frame0_source_image"],np.float32),teacher_uv_processed=tuv.astype(np.float32),observed_uv_processed=ouv.astype(np.float32),teacher_valid=tv,observed_valid=ov,processed_resolution_hw=processed,query_reference_source_frame_index=np.int32(14),output_camera_source_frame_index=np.int32(20),teacher_protocol=np.asarray("official_interleaved23_two_12frame_runs_pj0_anchored"),observed_protocol=np.asarray("nonofficial_observed7_pj0_anchored"),projection_coordinate_transform=np.asarray("w2c_j @ inverse(w2c_0) from frame0-camera tracks to frame-j camera"),preview_mp4_fps_used=np.asarray(False))
    print(json.dumps({"video_id":m["video_id"],"teacher_shape":list(tp.shape),"observed_shape":list(op.shape)}))
if __name__=="__main__":main()
