import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path("outputs/text_matrix")
IN = Path("inputs/tracks")
VIDEOS = ["10790317943435558913", "gear_v2_9-wrist_front_rgb_img-0"]
CONDITIONS = ["none", "consistent", "static", "conflict"]
SEEDS = [42, 43, 44]


def load_run(stem: str, condition: str, seed: int):
    d = ROOT / stem / condition / f"seed_{seed}" / stem
    if not (d / "pred_tracks.npy").exists():
        return None
    pred = np.load(d / "pred_tracks.npy").astype(np.float32)
    pv = np.load(d / "pred_visibility.npy").astype(bool)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    return d, pred, pv, meta


def sha(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


def pair_metrics(a, b, valid):
    dist = np.linalg.norm(a - b, axis=-1)
    ade = float(dist[valid].mean())
    last = valid[-1]
    fde = float(dist[-1, last].mean()) if np.any(last) else float("nan")
    return {"ade_px": ade, "fde_px": fde}


def main():
    all_rows = []
    summary = {}
    for stem in VIDEOS:
        gt_raw = np.load(IN / f"{stem}.npz")
        gt = gt_raw["tracks"].astype(np.float32)
        gv = gt_raw["vis"].astype(bool)
        # Input archives may store tracks as (points, time, xy); normalize
        # to the time-first layout used by predictions before selecting the
        # 81-frame future horizon.
        if gt.ndim == 3 and gt.shape[0] != 81 and gt.shape[1] in (81, 162):
            gt = np.transpose(gt, (1, 0, 2))
            gv = np.transpose(gv, (1, 0))
        if gt.shape[0] == 162:
            gt = gt[81:162]
            gv = gv[81:162]
        elif gt.shape[0] != 81:
            raise ValueError(f"Unexpected GT shape after normalization: {gt.shape}")
        runs = {}
        for condition in CONDITIONS:
            for seed in SEEDS:
                item = load_run(stem, condition, seed)
                if item is None:
                    continue
                d, pred, pv, meta = item
                valid = gv & np.isfinite(gt).all(-1) & np.isfinite(pred).all(-1)
                err = np.linalg.norm(pred - gt, axis=-1)
                path = np.linalg.norm(np.diff(pred, axis=0), axis=-1)
                gt_path = np.linalg.norm(np.diff(gt, axis=0), axis=-1)
                pairvalid = valid[1:] & valid[:-1]
                row = {
                    "video": stem,
                    "condition": condition,
                    "seed": seed,
                    "caption": meta.get("caption", ""),
                    "pred_shape": list(pred.shape),
                    "pred_sha256": sha(pred),
                    "pred_visibility_mean": float(pv.mean()),
                    "ade_px": float(err[valid].mean()),
                    "fde_px": float(err[-1, valid[-1]].mean()),
                    "pred_path_px_mean": float(path[pairvalid].mean()),
                    "gt_path_px_mean": float(gt_path[pairvalid].mean()),
                    "final_centroid": pred[-1].mean(axis=0).tolist(),
                    "last_history_centroid": pred[0].mean(axis=0).tolist(),
                }
                all_rows.append(row)
                runs[(condition, seed)] = (pred, valid, row)

        fixed = {c: runs[(c, 42)] for c in CONDITIONS if (c, 42) in runs}
        pairs = {}
        for i, ca in enumerate(CONDITIONS):
            if ca not in fixed:
                continue
            for cb in CONDITIONS[i + 1 :]:
                if cb not in fixed:
                    continue
                pa, va, _ = fixed[ca]
                pb, vb, _ = fixed[cb]
                pairs[f"{ca}__vs__{cb}"] = pair_metrics(pa, pb, va & vb)

        seed_pairs = {}
        if all(("consistent", s) in runs for s in SEEDS):
            for sa, sb in [(42, 43), (42, 44), (43, 44)]:
                pa, va, _ = runs[("consistent", sa)]
                pb, vb, _ = runs[("consistent", sb)]
                seed_pairs[f"{sa}__vs__{sb}"] = pair_metrics(pa, pb, va & vb)

        base = fixed.get("none")
        delta_from_none = {}
        if base is not None:
            p0, v0, _ = base
            for c, (pc, vc, _) in fixed.items():
                if c == "none":
                    continue
                delta_from_none[c] = pair_metrics(p0, pc, v0 & vc)
        summary[stem] = {
            "available_runs": [
                {"condition": c, "seed": s} for (c, s) in sorted(runs)
            ],
            "fixed_seed42_condition_pairs": pairs,
            "consistent_same_caption_seed_pairs": seed_pairs,
            "condition_vs_none_seed42": delta_from_none,
            "condition_path_px_seed42": {
                c: fixed[c][2]["pred_path_px_mean"] for c in fixed
            },
        }

    out = {
        "protocol": {
            "videos": VIDEOS,
            "conditions": CONDITIONS,
            "fixed_seed": 42,
            "repeated_consistent_caption_seeds": [43, 44],
            "steps": 20,
            "history_frames": 81,
            "future_frames": 81,
            "resolution": [832, 480],
            "text_encoder": "Wan UMT5-XXL pooled embeddings",
            "interpretation": "fixed-seed cross-condition differences test text sensitivity; repeated seeds test stochastic branch variation",
        },
        "rows": all_rows,
        "videos": summary,
    }
    Path("analysis").mkdir(exist_ok=True)
    Path("analysis/text_matrix_metrics.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
