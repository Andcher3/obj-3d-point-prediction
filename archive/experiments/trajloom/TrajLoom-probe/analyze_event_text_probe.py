import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
REMOTE_ROOT = Path("/root/autodl-tmp/projects/TrajLoom-probe")
STEMS = ["10790317943435558913", "gear_v2_9-wrist_front_rgb_img-0"]
TARGET_ROIS = {
    # Rectangles are defined on frame 0, where query_xy is sampled.
    "10790317943435558913": (320, 576, 128, 416),
    "gear_v2_9-wrist_front_rgb_img-0": (384, 480, 64, 256),
}


def artifact(base: Path, stem: str, name: str) -> Path:
    return base / stem / name


def load_case(stem: str, key: str, seed: int):
    if key == "no_text":
        base = REMOTE_ROOT / "outputs" / "matrix" / stem / f"seed_{seed}"
    else:
        base = REMOTE_ROOT / "outputs" / "event_text_probe" / stem / f"{key}_seed{seed}"
    pred = np.load(artifact(base, stem, "pred_tracks.npy")).astype(np.float32)
    gt = np.load(artifact(base, stem, "gt_tracks.npy")).astype(np.float32)
    vis = np.load(artifact(base, stem, "gt_visibility.npy")).astype(bool)
    meta = json.loads(artifact(base, stem, "meta.json").read_text())
    return pred, gt, vis, meta


def metrics(pred, gt, vis, point_mask=None):
    active = vis.sum(axis=0) >= max(4, int(0.25 * vis.shape[0]))
    if point_mask is not None:
        active &= point_mask
    valid = vis & active[None]
    err = np.linalg.norm(pred - gt, axis=-1)
    ade = float(err[valid].mean())
    last_valid = vis[-1] & active
    fde = float(err[-1, last_valid].mean())
    delta = pred[-1, active] - pred[0, active]
    return {
        "active_points": int(active.sum()),
        "ade_px": ade,
        "fde_px": fde,
        "mean_final_displacement_xy_px": delta.mean(axis=0).tolist(),
        "mean_final_displacement_norm_px": float(np.linalg.norm(delta, axis=-1).mean()),
    }


result = {
    "protocol": {
        "fixed_seed_text_contrast": 42,
        "correct_text_seeds": [42, 43, 44],
        "future_frames": 81,
        "solver_steps": 20,
    },
    "videos": {},
}

for stem in STEMS:
    cases = {}
    arrays = {}
    for key, seeds in (("no_text", [42, 43, 44]), ("correct", [42, 43, 44]), ("opposite", [42, 43, 44])):
        for seed in seeds:
            pred, gt, vis, meta = load_case(stem, key, seed)
            base = (REMOTE_ROOT / "outputs" / ("matrix" if key == "no_text" else "event_text_probe") /
                    stem / (f"seed_{seed}" if key == "no_text" else f"{key}_seed{seed}"))
            query_xy = np.load(artifact(base, stem, "query_xy.npy"))
            x0, x1, y0, y1 = TARGET_ROIS[stem]
            target_mask = ((query_xy[:, 0] >= x0) & (query_xy[:, 0] <= x1) &
                           (query_xy[:, 1] >= y0) & (query_xy[:, 1] <= y1))
            label = f"{key}_seed{seed}"
            arrays[label] = pred
            cases[label] = {
                "caption": meta.get("caption", ""),
                "full_grid": metrics(pred, gt, vis),
                "target_roi": metrics(pred, gt, vis, target_mask),
            }

    def distance(a, b, mask=None):
        d = np.linalg.norm(arrays[a] - arrays[b], axis=-1)
        if mask is not None:
            d = d[:, mask]
        return {"all_mean_px": float(d.mean()), "final_mean_px": float(d[-1].mean())}

    paired_text = {}
    target_text_distances = []
    for seed in (42, 43, 44):
        correct = cases[f"correct_seed{seed}"]["target_roi"]
        opposite = cases[f"opposite_seed{seed}"]["target_roi"]
        target_distance = distance(f"correct_seed{seed}", f"opposite_seed{seed}", target_mask)
        target_text_distances.append(target_distance["all_mean_px"])
        paired_text[str(seed)] = {
            "target_correct_minus_opposite_ade_px": correct["ade_px"] - opposite["ade_px"],
            "target_correct_minus_opposite_fde_px": correct["fde_px"] - opposite["fde_px"],
            "target_prediction_distance": target_distance,
            "target_correct_minus_opposite_final_displacement_xy_px": (
                np.asarray(correct["mean_final_displacement_xy_px"]) -
                np.asarray(opposite["mean_final_displacement_xy_px"])
            ).tolist(),
        }

    random_distances = [
        distance("correct_seed42", "correct_seed43", target_mask)["all_mean_px"],
        distance("correct_seed42", "correct_seed44", target_mask)["all_mean_px"],
        distance("correct_seed43", "correct_seed44", target_mask)["all_mean_px"],
    ]

    result["videos"][stem] = {
        "target_roi_xyxy_frame0": TARGET_ROIS[stem],
        "cases": cases,
        "contrasts": {
            "correct_vs_opposite_seed42": distance("correct_seed42", "opposite_seed42"),
            "correct_vs_no_text_seed42": distance("correct_seed42", "no_text_seed42"),
            "correct_seed42_vs_seed43": distance("correct_seed42", "correct_seed43"),
            "correct_seed42_vs_seed44": distance("correct_seed42", "correct_seed44"),
        },
        "paired_text_contrasts_by_seed": paired_text,
        "mean_target_text_effect_px": float(np.mean(target_text_distances)),
        "mean_target_random_seed_effect_px": float(np.mean(random_distances)),
        "text_to_random_effect_ratio": float(np.mean(target_text_distances) / np.mean(random_distances)),
        "target_text_ade_delta_correct_minus_opposite_px": (
            cases["correct_seed42"]["target_roi"]["ade_px"] -
            cases["opposite_seed42"]["target_roi"]["ade_px"]
        ),
        "target_text_fde_delta_correct_minus_opposite_px": (
            cases["correct_seed42"]["target_roi"]["fde_px"] -
            cases["opposite_seed42"]["target_roi"]["fde_px"]
        ),
    }

out = REMOTE_ROOT / "analysis" / "event_text_probe_metrics.json"
out.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
