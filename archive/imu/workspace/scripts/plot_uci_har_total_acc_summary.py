from __future__ import annotations

import csv
import html
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "processed" / "uci_har"
OUT = PROJECT_ROOT / "reports" / "figures" / "uci_har" / "total_acc_summary.svg"
ACTIVITIES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]
COLORS = {"x": "#2f6fbb", "y": "#c4494d", "z": "#4c8f4f"}


def x_scale(value: float, left: float, width: float, vmin: float, vmax: float) -> float:
    return left + ((value - vmin) / (vmax - vmin)) * width


def main() -> int:
    X = np.fromfile(DATA_ROOT / "X_train_raw_float32.bin", dtype="<f4").reshape(7352, 128, 9)
    with (DATA_ROOT / "train_metadata.csv").open(encoding="utf-8", newline="") as f:
        metadata = list(csv.DictReader(f))

    rows = []
    for activity in ACTIVITIES:
        idx = np.array([i for i, row in enumerate(metadata) if row["activity"] == activity], dtype=int)
        data = X[idx, :, 0:3]
        flat = data.reshape(-1, 3)
        means = flat.mean(axis=0)
        avg_window_std = data.std(axis=1).mean()
        rows.append((activity, means, avg_window_std))

    width = 1120
    height = 650
    left = 230
    axis_width = 560
    bar_h = 13
    row_gap = 78
    top = 92
    vmin, vmax = -0.9, 1.1
    zero_x = x_scale(0, left, axis_width, vmin, vmax)
    std_left = 870
    std_width = 170
    std_max = 0.25

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbf8"/>',
        '<text x="28" y="36" font-size="24" font-family="Arial" font-weight="700">total_acc class summary</text>',
        '<text x="28" y="60" font-size="13" font-family="Arial" fill="#555">Left: class-level mean of total_acc x/y/z. Right: average within-window temporal std, a rough motion-strength measure.</text>',
        f'<line x1="{zero_x:.1f}" y1="78" x2="{zero_x:.1f}" y2="{height - 50}" stroke="#777" stroke-dasharray="4 4"/>',
        f'<text x="{left}" y="82" font-size="12" font-family="Arial" fill="#555">axis mean in g (-0.9 to 1.1)</text>',
        f'<text x="{std_left}" y="82" font-size="12" font-family="Arial" fill="#555">avg window std</text>',
    ]

    for i, (activity, means, avg_window_std) in enumerate(rows):
        y = top + i * row_gap
        parts.append(f'<text x="28" y="{y + 22}" font-size="14" font-family="Arial" font-weight="700">{html.escape(activity)}</text>')
        for axis_i, axis in enumerate(["x", "y", "z"]):
            value = float(means[axis_i])
            x0 = x_scale(min(0, value), left, axis_width, vmin, vmax)
            x1 = x_scale(max(0, value), left, axis_width, vmin, vmax)
            by = y + axis_i * 18
            parts.append(f'<rect x="{x0:.1f}" y="{by}" width="{max(x1 - x0, 1):.1f}" height="{bar_h}" fill="{COLORS[axis]}"/>')
            label_x = x1 + 5 if value >= 0 else x0 - 42
            parts.append(f'<text x="{label_x:.1f}" y="{by + 11}" font-size="11" font-family="Arial" fill="#333">{axis}:{value:.2f}</text>')

        std_w = min(avg_window_std / std_max, 1.0) * std_width
        parts.append(f'<rect x="{std_left}" y="{y + 11}" width="{std_w:.1f}" height="20" fill="#7768ae"/>')
        parts.append(f'<text x="{std_left + std_w + 6:.1f}" y="{y + 26}" font-size="12" font-family="Arial" fill="#333">{avg_window_std:.3f}</text>')

    legend_y = height - 24
    lx = 28
    for j, axis in enumerate(["x", "y", "z"]):
        x = lx + j * 70
        parts.append(f'<rect x="{x}" y="{legend_y - 10}" width="24" height="10" fill="{COLORS[axis]}"/>')
        parts.append(f'<text x="{x + 31}" y="{legend_y}" font-size="12" font-family="Arial">{axis}</text>')
    parts.append("</svg>")

    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
