from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path("/root/autodl-tmp/projects/TrajLoom-probe")
OUT = ROOT / "analysis" / "event_text_visuals"
STEMS = ["10790317943435558913", "gear_v2_9-wrist_front_rgb_img-0"]
ROIS = {
    "10790317943435558913": (320, 576, 128, 416),
    "gear_v2_9-wrist_front_rgb_img-0": (384, 480, 64, 256),
}
CASES = [
    ("no_text", 42, "no text", "#f59e0b"),
    ("correct", 42, "matching event", "#0891b2"),
    ("opposite", 42, "opposite event", "#dc2626"),
    ("correct", 43, "matching event, seed 43", "#16a34a"),
    ("correct", 44, "matching event, seed 44", "#7c3aed"),
]


def case_dir(stem, key, seed):
    if key == "no_text":
        return ROOT / "outputs" / "matrix" / stem / f"seed_{seed}" / stem
    return ROOT / "outputs" / "event_text_probe" / stem / f"{key}_seed{seed}" / stem


def frame80(stem):
    path = OUT / f"{stem}-frame80.png"
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-y", "-i",
        str(ROOT / "inputs" / "videos" / f"{stem}.mp4"),
        "-vf", "select=eq(n\\,80)", "-vframes", "1", str(path),
    ], check=True)
    return np.asarray(Image.open(path).convert("RGB")), path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for stem in STEMS:
        bg, tmp = frame80(stem)
        d0 = case_dir(stem, "no_text", 42)
        query = np.load(d0 / "query_xy.npy")
        x0, x1, y0, y1 = ROIS[stem]
        target = np.flatnonzero(
            (query[:, 0] >= x0) & (query[:, 0] <= x1) &
            (query[:, 1] >= y0) & (query[:, 1] <= y1)
        )

        fig, axes = plt.subplots(1, len(CASES), figsize=(22, 4.4), squeeze=False)
        for ax, (key, seed, label, color) in zip(axes[0], CASES):
            d = case_dir(stem, key, seed)
            pred = np.load(d / "pred_tracks.npy").astype(np.float32)
            vis = np.load(d / "pred_visibility.npy").astype(bool)
            ax.imshow(bg)
            for qi in target:
                valid = vis[:, qi] & np.isfinite(pred[:, qi]).all(axis=1)
                pts = pred[valid, qi]
                if len(pts) < 2:
                    continue
                ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=0.8, alpha=0.6)
                ax.scatter(pts[-1, 0], pts[-1, 1], s=7, c=color, marker="o")
            ax.set_title(label, fontsize=10)
            ax.set_xlim(0, bg.shape[1])
            ax.set_ylim(bg.shape[0], 0)
            ax.set_axis_off()
        fig.suptitle(f"Target-region forecast comparison | {stem}", fontsize=12)
        fig.tight_layout()
        fig.savefig(OUT / f"{stem}-event-text.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
