"""Compare cached 7/22-frame 3D tracks, CPU only, without altering raw outputs.

Distances are straight-line lengths, not surface geodesics or edge vectors.
The 22-frame run is a comparison reference, not 3D ground truth. A single
t0 scalar removes context scale ambiguity; it is diagnostic, not label repair.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


VIDEOS = ("car-turn", "drift-turn", "boat", "soccerball")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    assert values.size and np.isfinite(values).all()
    return {"count": int(values.size), "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)), "max": float(values.max()),
            "mean": float(values.mean())}


def distances(tracks, pairs):
    return np.linalg.norm(tracks[:, pairs[:, 0]] - tracks[:, pairs[:, 1]], axis=-1)


def relative_difference(observed, full):
    # No short edges silently removed; zero-length reference would fail explicitly.
    assert np.isfinite(observed).all() and np.isfinite(full).all()
    assert (full > 0).all(), "Zero reference length; protocol needs explicit handling"
    return np.abs(observed - full) / full


def knn_pairs(uv, k=3):
    square = np.sum((uv[:, None] - uv[None]) ** 2, axis=-1)
    np.fill_diagonal(square, np.inf)
    nearest = np.argsort(square, axis=1, kind="stable")[:, :k]
    return np.array(sorted({tuple(sorted((i, int(j))))
                            for i, js in enumerate(nearest) for j in js}), dtype=int)


def scale_holdout(observed, full, pairs, t0):
    # Four disjoint query folds; training pairs never contain a held-out query.
    errors, scales = [], []
    for fold in range(4):
        held_out = np.arange(48) % 4 == fold
        train = ~held_out[pairs].any(axis=1)
        test = held_out[pairs].all(axis=1)
        scale = float(np.median(full[t0, train] / observed[t0, train]))
        errors.append(relative_difference(scale * observed[t0, test], full[t0, test]))
        scales.append(scale)
    return {"fold_scales": scales, "heldout_t0_relative_difference": summary(np.concatenate(errors)),
            "train_query_count_per_fold": 36, "test_query_count_per_fold": 12,
            "test_pair_count_per_fold": 66}


def self_test():
    rng = np.random.default_rng(173)
    x = rng.normal(size=(7, 48, 3))
    pairs = np.array(np.triu_indices(48, 1)).T
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    y = 2.5 * (x @ q) + np.array([4., -3., 1.])
    a, b = distances(x, pairs), distances(y, pairs)
    scale = np.median(b[6] / a[6])
    assert np.max(relative_difference(scale * a, b)) < 1e-12
    y[:, 0, 0] += 3
    changed = distances(y, pairs)
    assert np.median(relative_difference(scale * a[:, pairs[:, 0] == 0],
                                         changed[:, pairs[:, 0] == 0])) > 0.01
    permutation = rng.permutation(48)
    inverse = np.argsort(permutation)
    assert np.allclose(distances(x[:, permutation][:, inverse], pairs), a)
    return {"similarity_transform_invariance": True, "local_deformation_detected": True,
            "query_permutation_invariance": True}


def load_tracks(path, expected_t, manifest):
    with np.load(path, allow_pickle=False) as data:
        xyz = data["tracks_3d"].astype(np.float64)
        source = data["source_frame_indices"].copy()
        uv = data["query_uv_normalized_01"].copy()
        assert xyz.shape == (expected_t, 48, 3) and np.isfinite(xyz).all()
        assert np.array_equal(source, manifest["source_frame_indices"][:expected_t])
        assert np.allclose(uv, manifest["query_uv_normalized_01"], rtol=0, atol=1e-7)
        # Verify these are tracks from query frame 6, not self-pair point maps.
        pair_idx = data["pair_idx"]
        for t in range(expected_t):
            matching = np.flatnonzero((pair_idx == [6, t]).all(axis=1))
            assert len(matching) == 1
            assert np.array_equal(xyz[t], data["warp3d"][matching[0]].astype(np.float64))
    return xyz, source, uv


def analyze(base, out):
    out.mkdir(parents=True, exist_ok=True)
    packaging = json.loads((base / "visual-packaging-checks.json").read_text(encoding="utf-8-sig"))
    expected = {(v["video"], c["context_frames"]): c["raw_sha256"]
                for v in packaging["videos"] for c in v["contexts"]}
    result = {
        "verification_status": "ANALYZED", "cpu_only": True, "model_forward_called": False,
        "raw_outputs_modified": False, "ground_truth_available": False,
        "units": "ambiguous model units, not meters",
        "distance_definition": "Euclidean length of 3D differences between matched query IDs",
        "primary_relative_difference": "abs(s * d_observed - d_full) / d_full; s=1 for raw",
        "scale_definition": "one scalar s=median(d_full[t0]/d_observed[t0]) over all 1128 pairs; constant over all 7 frames",
        "secondary_temporal_difference": "abs(d_observed(t)/d_observed(t0) - d_full(t)/d_full(t0)); dimensionless",
        "confidence_filter": "none; all 48 approved queries retained, no outcome-based exclusion",
        "self_tests": self_test(), "videos": []}
    rows = []
    for video in VIDEOS:
        manifest_path = base / "inputs" / video / "queries.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        t0 = manifest["query_model_frame_index"]
        assert t0 == 6 and manifest["query_count"] == 48
        paths = {t: base / "outputs" / video / f"{video}_{t}frames_raw.npz" for t in (7, 22)}
        hashes = {str(t): sha256(path) for t, path in paths.items()}
        assert all(hashes[str(t)] == expected[(video, t)] for t in paths)
        obs, source, uv = load_tracks(paths[7], 7, manifest)
        full, full_source, full_uv = load_tracks(paths[22], 22, manifest)
        assert np.array_equal(source, full_source[:7]) and np.array_equal(uv, full_uv)
        # Both view-0 coordinate frames share the same source image, but scale is ambiguous.
        for t in (7, 22):
            meta = json.loads((paths[t].parent / f"{video}_{t}frames_metadata.json").read_text())
            assert meta["coordinate_frame"] == "context view0 camera"
            assert meta["query_source_model_index"] == 6 and not meta["t0_posthoc_alignment"]
        pairs = np.array(np.triu_indices(48, 1)).T
        d_obs, d_full = distances(obs, pairs), distances(full[:7], pairs)
        assert (d_obs > 0).all() and (d_full > 0).all()
        scale = float(np.median(d_full[t0] / d_obs[t0]))
        raw = relative_difference(d_obs, d_full)
        aligned = relative_difference(scale * d_obs, d_full)
        local = knn_pairs(np.array(manifest["query_uv_processed_pixels"]))
        local_mask = np.array([tuple(pair) in {tuple(p) for p in local} for pair in pairs])
        temporal = np.abs(d_obs / d_obs[t0] - d_full / d_full[t0])
        frames = []
        for t in range(7):
            frame = {"model_frame_index": t, "source_frame_index": int(source[t]),
                     "is_t0": t == t0, "raw": summary(raw[t]), "scale_adjusted": summary(aligned[t]),
                     "local_knn_scale_adjusted": summary(aligned[t, local_mask]),
                     "median_scale_ratio_full_over_observed": float(np.median(d_full[t] / d_obs[t]))}
            frames.append(frame)
            rows.append({"video": video, "model_frame": t, "source_frame": int(source[t]),
                         "scale_at_t0": scale, "raw_median_percent": 100 * frame["raw"]["median"],
                         "raw_p90_percent": 100 * frame["raw"]["p90"],
                         "adjusted_median_percent": 100 * frame["scale_adjusted"]["median"],
                         "adjusted_p90_percent": 100 * frame["scale_adjusted"]["p90"]})
        item = {"video": video, "query_count": 48, "pairs_per_frame": 1128,
                "overlap_source_frames": source.tolist(), "t0_source_frame": int(source[t0]),
                "raw_sha256": hashes, "manifest_sha256": sha256(manifest_path),
                "scale_at_t0": scale, "raw_all_frames": summary(raw),
                "scale_adjusted_all_frames": summary(aligned),
                "raw_t0": summary(raw[t0]), "scale_adjusted_t0": summary(aligned[t0]),
                "local_knn_pair_count": len(local), "local_knn_scale_adjusted_all_frames": summary(aligned[:, local_mask]),
                "t0_scale_holdout": scale_holdout(d_obs, d_full, pairs, t0),
                "temporal_length_change_disagreement_excluding_t0": summary(temporal[:t0]),
                "minimum_pair_length_observed": float(d_obs.min()),
                "minimum_pair_length_full": float(d_full.min()), "frames": frames}
        np.savez_compressed(out / f"{video}_pair_distances.npz", pairs=pairs,
                            source_frame_indices=source, observed_distances=d_obs,
                            full_distances=d_full, scale_at_t0=np.array(scale),
                            raw_relative_difference=raw, scale_adjusted_relative_difference=aligned,
                            temporal_relative_change_disagreement=temporal, local_knn_pair_mask=local_mask)
        assert all(sha256(path) == hashes[str(t)] for t, path in paths.items())
        result["videos"].append(item)
    (out / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "per_frame.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return result


def plot(result, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, video in zip(axes.ravel(), result["videos"]):
        frames = video["frames"]
        x = np.arange(7)
        for key, label, color in (("raw", "Raw median", "#999999"),
                                  ("scale_adjusted", "Single-scale median", "#2467ae")):
            ax.plot(x, [100 * f[key]["median"] for f in frames], marker="o", label=label, color=color)
        ax.plot(x, [100 * f["scale_adjusted"]["p90"] for f in frames],
                linestyle="--", label="Single-scale p90", color="#cc6535")
        ax.axvline(6, color="black", alpha=.3, linestyle=":")
        ax.set_title(video["video"])
        ax.set_ylabel("Absolute difference / full-run distance (%)")
        ax.set_xlabel("Shared frame index (6 = t0)")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("7-frame vs 22-frame contexts: same 48 queries, all 1128 pairs\n"
                 "One scale fitted at t0, fixed for every frame; neither run is ground truth")
    fig.tight_layout()
    fig.savefig(out / "pair_distance_comparison.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.base, args.out)
    plot(result, args.out)
    for v in result["videos"]:
        print(v["video"], json.dumps({k: v[k] for k in (
            "scale_at_t0", "raw_all_frames", "scale_adjusted_all_frames", "scale_adjusted_t0",
            "local_knn_scale_adjusted_all_frames", "t0_scale_holdout",
            "temporal_length_change_disagreement_excluding_t0")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
