"""CPU teacher-vs-GT evaluation. No future fitting or pointwise alignment."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def stats(values):
    x = np.asarray(values).reshape(-1)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"count": 0, "mean": None, "median": None, "p90": None}
    return {"count": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)), "p90": float(np.quantile(x,.9))}


def compare(pred, truth, t0, diagonal, scale):
    pred = np.asarray(pred, np.float64) * scale
    truth = np.asarray(truth, np.float64)
    i,j = np.triu_indices(pred.shape[1],1)
    pe, ge = pred[:,i]-pred[:,j], truth[:,i]-truth[:,j]
    pl, gl = np.linalg.norm(pe,axis=-1), np.linalg.norm(ge,axis=-1)
    absolute = np.linalg.norm(pred-truth,axis=-1)
    centered = np.linalg.norm((pred-pred.mean(1,keepdims=True))-(truth-truth.mean(1,keepdims=True)),axis=-1)
    edge = np.linalg.norm(pe-ge,axis=-1)/diagonal
    length = np.abs(pl-gl)/np.maximum(gl, diagonal*1e-6)
    change = np.linalg.norm((pe-pe[t0])-(ge-ge[t0]),axis=-1)/diagonal
    groups = {}
    for name, sl in [("observed",slice(0,t0+1)), ("future",slice(t0+1,None)), ("t0",slice(t0,t0+1))]:
        groups[name] = {"absolute_point_error_scene_units": stats(absolute[sl]),
                        "centered_point_error_bbox_fraction": stats(centered[sl]/diagonal),
                        "relative_vector_error_bbox_fraction": stats(edge[sl]),
                        "relative_pair_length_error_fraction": stats(length[sl]),
                        "edge_change_error_bbox_fraction_diagnostic": stats(change[sl])}
    return groups, {"point": absolute/diagonal, "edge": edge, "length": length, "i": i, "j": j}


def scene_metrics(raw, gt, manifest):
    pred = raw["tracks_3d"].astype(np.float64)
    count = len(pred)
    truth = gt["tracks_first_camera"][:count]
    t0 = manifest["query_model_frame_index"]
    queries = len(manifest["query_uv_normalized_01"])
    if pred.shape != truth.shape or pred.shape[1:] != (queries,3) or queries < 2 or not np.isfinite(pred).all():
        raise ValueError("Bad teacher/GT shape or numbers")
    np.testing.assert_array_equal(raw["source_frame_indices"], gt["source_frame_indices"][:count])
    np.testing.assert_allclose(raw["query_uv_normalized_01"], manifest["query_uv_normalized_01"], atol=1e-7)
    box = gt["target_bboxes_world"][t0]
    # Rotation-invariant box diagonal from published eight world-space corners.
    diagonal = float(np.max(np.linalg.norm(box[:,None]-box[None],axis=-1)))
    i,j = np.triu_indices(queries,1)
    gl = np.linalg.norm(truth[t0,i]-truth[t0,j],axis=-1)
    pl = np.linalg.norm(pred[t0,i]-pred[t0,j],axis=-1)
    if np.any(gl <= diagonal*1e-8) or np.any(pl <= diagonal*1e-8):
        raise ValueError("Degenerate t0 point pairs; do not manufacture scale")
    scale = float(np.median(gl/pl))
    groups, arrays = compare(pred, truth, t0, diagonal, scale)
    raw_groups, _ = compare(pred, truth, t0, diagonal, 1.)
    visibility = gt["visibility"][:count]
    visible_edges = visibility[:,i] & visibility[:,j]
    splits = {}
    for name, point_mask, edge_mask in [("visible", visibility, visible_edges),
                                         ("occluded", ~visibility, ~visible_edges)]:
        # Future-only split; hidden valid GT is retained.
        splits[name] = {"point_error_bbox_fraction": stats(arrays["point"][t0+1:][point_mask[t0+1:]]),
                        "relative_vector_error_bbox_fraction": stats(arrays["edge"][t0+1:][edge_mask[t0+1:]]),
                        "relative_length_error_fraction": stats(arrays["length"][t0+1:][edge_mask[t0+1:]])}
    gt_uv = gt["tracks_2d"][:count]
    uv_errors = {}
    for key in ["tracks_2d_source_pixels", "tracks_3d_projected_source_pixels"]:
        error = np.linalg.norm(raw[key]-gt_uv,axis=-1)
        uv_errors[key] = {"t0_pixels": stats(error[t0]), "future_visible_pixels": stats(error[t0+1:][visibility[t0+1:]]),
                          "future_all_finite_pixels": stats(error[t0+1:])}
    return {"scene_id": manifest["scene_id"], "frame_count": count, "query_count":queries, "pair_count":len(i),
            "target_bbox_diagonal_scene_units": diagonal, "one_t0_global_scale": scale,
            "scale_adjusted": groups, "raw_model_scale": raw_groups, "future_visibility_splits": splits,
            "image_errors": uv_errors, "pointwise_t0_alignment": False, "future_fit": False,
            "gt_unit": "simulator scene units", "absolute_error_includes_shape_and_camera_errors": True}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base",type=Path,required=True)
    ap.add_argument("--context",type=int,choices=[7,22],default=22)
    args=ap.parse_args()
    preparation=json.loads((args.base/"logs"/"gt-preparation-status.json").read_text())
    results=[]
    for scene in preparation["scenes"]:
        if scene["status"] != "READY":
            continue
        directory=args.base/"inputs"/scene["scene_id"]
        output=args.base/"outputs"/scene["scene_id"]
        raw_path=output/f"{scene['scene_id']}_{args.context}frames_raw.npz"
        metadata_path=output/f"{scene['scene_id']}_{args.context}frames_metadata.json"
        if not raw_path.exists() or not metadata_path.exists():
            raise FileNotFoundError("Missing actual inference; never treat preparation as a model result")
        manifest_path=directory/"queries.json"
        metadata=json.loads(metadata_path.read_text())
        if hashlib.sha256(manifest_path.read_bytes()).hexdigest()!=metadata["input_manifest_sha256"]:
            raise ValueError("Inference query manifest changed")
        gt_path=directory/"gt_tracks.npz"
        if hashlib.sha256(gt_path.read_bytes()).hexdigest()!=metadata["gt_sha256"]:
            raise ValueError("GT changed since inference")
        raw=np.load(raw_path,allow_pickle=False)
        gt=np.load(gt_path,allow_pickle=False)
        result=scene_metrics(raw,gt,json.loads(manifest_path.read_text()))
        (output/f"gt_comparison_{args.context}frames.json").write_text(json.dumps(result,indent=2)+"\n")
        results.append(result)
    if not results:
        raise ValueError("No model outputs")
    macro={}
    for metric in results[0]["scale_adjusted"]["future"]:
        means=[r["scale_adjusted"]["future"][metric]["mean"] for r in results]
        macro[metric]=stats([m for m in means if m is not None])
    report={"experiment":"G0 / experiment1", "model_inference_completed":True,"scene_count":len(results),
            "context":args.context,"per_scene":results,"future_scene_macro":macro,
            "training_performed":False,"relation_method_validated":False,
            "source_exclusions": [s for s in preparation["scenes"] if s["status"]!="READY"]}
    (args.base/"logs"/f"experiment1_gt_result_{args.context}frames.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"scene_count":len(results),"future_scene_macro":macro},indent=2))


if __name__=="__main__":
    main()
