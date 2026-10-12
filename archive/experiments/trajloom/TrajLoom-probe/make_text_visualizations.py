from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path("outputs/text_matrix")
VIDEO_DIR = Path("inputs/videos")
OUT = Path("analysis/text_matrix_visuals")
VIDEOS = ["10790317943435558913", "gear_v2_9-wrist_front_rgb_img-0"]
CONDITIONS = ["none", "consistent", "static", "conflict"]
LABELS = {
    "none": "no text",
    "consistent": "consistent text",
    "static": "static text",
    "conflict": "conflict text",
}
COLORS = {
    "none": (1.0, 0.55, 0.0),
    "consistent": (0.0, 0.72, 1.0),
    "static": (0.2, 0.85, 0.2),
    "conflict": (0.95, 0.15, 0.15),
}


def get_background(stem: str):
    frame_path = OUT / f"{stem}-frame80.png"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(VIDEO_DIR / f"{stem}.mp4"),
         "-vf", "select=eq(n\\,80)", "-vframes", "1", str(frame_path)],
        check=True,
    )
    return np.asarray(Image.open(frame_path).convert("RGB"))


def load(stem: str, condition: str):
    d = ROOT / stem / condition / "seed_42" / stem
    return (
        np.load(d / "pred_tracks.npy").astype(np.float32),
        np.load(d / "pred_visibility.npy").astype(bool),
    )


def draw_tracks(ax, tracks, visibility, color, indices):
    for qi in indices:
        pts = tracks[:, qi]
        valid = visibility[:, qi] & np.isfinite(pts).all(axis=1)
        pts = pts[valid]
        if len(pts) < 2:
            continue
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=0.8, alpha=0.75)
        ax.scatter(pts[0, 0], pts[0, 1], s=5, c=[color], marker="o")
        ax.scatter(pts[-1, 0], pts[-1, 1], s=8, c="black", marker="x")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for stem in VIDEOS:
        bg = get_background(stem)
        sample_indices = np.linspace(0, 389, 48, dtype=int)
        runs = {c: load(stem, c) for c in CONDITIONS}

        fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(20, 4.8), squeeze=False)
        for c in CONDITIONS:
            pred, vis = runs[c]
            ax = axes[0, CONDITIONS.index(c)]
            ax.imshow(bg)
            draw_tracks(ax, pred, vis, COLORS[c], sample_indices)
            ax.set_title(f"{LABELS[c]} | seed 42", fontsize=11)
            ax.set_axis_off()
        fig.suptitle(f"TrajLoom text-conditioned future trajectories | {stem}", fontsize=13)
        fig.tight_layout()
        fig.savefig(OUT / f"{stem}-text-conditions.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        frame_path = OUT / f"{stem}-frame80.png"
        if frame_path.exists():
            frame_path.unlink()
        print(f"wrote {stem}")


if __name__ == "__main__":
    main()
