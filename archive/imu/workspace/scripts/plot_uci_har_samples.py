from __future__ import annotations

import html
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.uci_har import ACTIVITY_ORDER, CHANNELS, dataset_root, load_metadata, load_sample  # noqa: E402


COLORS = {
    "x": "#2f6fbb",
    "y": "#c4494d",
    "z": "#4c8f4f",
}


def choose_samples(metadata: list[dict[str, int | str]], activity: str, count: int = 4) -> list[int]:
    rows = [row for row in metadata if row["activity"] == activity]
    selected: list[int] = []
    seen_subjects: set[int] = set()
    for row in rows:
        subject_id = int(row["subject_id"])
        if subject_id in seen_subjects:
            continue
        selected.append(int(row["sample_index"]))
        seen_subjects.add(subject_id)
        if len(selected) == count:
            return selected
    for row in rows:
        sample_index = int(row["sample_index"])
        if sample_index not in selected:
            selected.append(sample_index)
        if len(selected) == count:
            break
    return selected


def scale_points(values: list[float], x: float, y: float, width: float, height: float, vmin: float, vmax: float) -> str:
    if vmax == vmin:
        vmax = vmin + 1.0
    points = []
    for i, value in enumerate(values):
        px = x + (i / (len(values) - 1)) * width
        py = y + height - ((value - vmin) / (vmax - vmin)) * height
        points.append(f"{px:.1f},{py:.1f}")
    return " ".join(points)


def channel_values(sample: list[list[float]], channel_name: str) -> list[float]:
    channel_idx = CHANNELS.index(channel_name)
    return [time_step[channel_idx] for time_step in sample]


def subplot_svg(sample: list[list[float]], x: int, y: int, width: int, height: int, names: list[str]) -> str:
    series = [(name[-1], channel_values(sample, name)) for name in names]
    all_values = [value for _, values in series for value in values]
    vmin = min(all_values)
    vmax = max(all_values)
    pad = max((vmax - vmin) * 0.08, 0.05)
    vmin -= pad
    vmax += pad

    parts = [
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="#ffffff" stroke="#c8c8c8"/>'
    ]
    for frac in [0.25, 0.5, 0.75]:
        gy = y + frac * height
        parts.append(f'<line x1="{x}" y1="{gy:.1f}" x2="{x + width}" y2="{gy:.1f}" stroke="#ececec"/>')
    for axis, values in series:
        points = scale_points(values, x, y, width, height, vmin, vmax)
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{COLORS[axis]}" stroke-width="1.4"/>'
        )
    parts.append(f'<text x="{x + 4}" y="{y + 14}" font-size="11" fill="#555">{vmin:.2f} to {vmax:.2f}</text>')
    return "\n".join(parts)


def render_body_motion_svg(
    root: Path,
    metadata: list[dict[str, int | str]],
    activity: str,
    sample_indices: list[int],
    output_path: Path,
) -> None:
    width = 1240
    row_height = 220
    top = 72
    left_label_width = 190
    plot_width = 470
    plot_height = 155
    col_gap = 42
    height = top + len(sample_indices) * row_height + 40

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbf8"/>',
        f'<text x="28" y="34" font-size="24" font-family="Arial" font-weight="700">{html.escape(activity)}</text>',
        '<text x="28" y="56" font-size="13" font-family="Arial" fill="#555">Each row is one 2.56s training window. X-axis is time, colored lines are sensor x/y/z axes.</text>',
        f'<text x="{left_label_width}" y="66" font-size="15" font-family="Arial" font-weight="700">body_acc x/y/z (g)</text>',
        f'<text x="{left_label_width + plot_width + col_gap}" y="66" font-size="15" font-family="Arial" font-weight="700">body_gyro x/y/z (rad/s)</text>',
    ]

    meta_by_index = {int(row["sample_index"]): row for row in metadata}
    for row_idx, sample_index in enumerate(sample_indices):
        row_y = top + row_idx * row_height
        sample = load_sample(root, "train", sample_index)
        meta = meta_by_index[sample_index]
        label = f"sample {sample_index}\\nsubject {meta['subject_id']}"
        parts.append(
            f'<text x="28" y="{row_y + 48}" font-size="14" font-family="Arial" fill="#222">'
            f'{html.escape(label)}</text>'
        )
        parts.append(
            subplot_svg(
                sample,
                left_label_width,
                row_y,
                plot_width,
                plot_height,
                ["body_acc_x", "body_acc_y", "body_acc_z"],
            )
        )
        parts.append(
            subplot_svg(
                sample,
                left_label_width + plot_width + col_gap,
                row_y,
                plot_width,
                plot_height,
                ["body_gyro_x", "body_gyro_y", "body_gyro_z"],
            )
        )
        parts.append(f'<text x="{left_label_width}" y="{row_y + plot_height + 20}" font-size="11" font-family="Arial" fill="#666">0s</text>')
        parts.append(f'<text x="{left_label_width + plot_width - 24}" y="{row_y + plot_height + 20}" font-size="11" font-family="Arial" fill="#666">2.56s</text>')

    legend_y = height - 18
    legend_x = 28
    for idx, axis in enumerate(["x", "y", "z"]):
        x = legend_x + idx * 70
        parts.append(f'<line x1="{x}" y1="{legend_y}" x2="{x + 28}" y2="{legend_y}" stroke="{COLORS[axis]}" stroke-width="2"/>')
        parts.append(f'<text x="{x + 34}" y="{legend_y + 4}" font-size="12" font-family="Arial">{axis}</text>')
    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def render_total_acc_svg(
    root: Path,
    metadata: list[dict[str, int | str]],
    activity: str,
    sample_indices: list[int],
    output_path: Path,
) -> None:
    width = 820
    row_height = 220
    top = 72
    left_label_width = 190
    plot_width = 540
    plot_height = 155
    height = top + len(sample_indices) * row_height + 40

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbf8"/>',
        f'<text x="28" y="34" font-size="24" font-family="Arial" font-weight="700">{html.escape(activity)}</text>',
        '<text x="28" y="56" font-size="13" font-family="Arial" fill="#555">total_acc keeps gravity. Each row is one 2.56s training window.</text>',
        f'<text x="{left_label_width}" y="66" font-size="15" font-family="Arial" font-weight="700">total_acc x/y/z (g)</text>',
    ]

    meta_by_index = {int(row["sample_index"]): row for row in metadata}
    for row_idx, sample_index in enumerate(sample_indices):
        row_y = top + row_idx * row_height
        sample = load_sample(root, "train", sample_index)
        meta = meta_by_index[sample_index]
        label = f"sample {sample_index}\\nsubject {meta['subject_id']}"
        parts.append(
            f'<text x="28" y="{row_y + 48}" font-size="14" font-family="Arial" fill="#222">'
            f'{html.escape(label)}</text>'
        )
        parts.append(
            subplot_svg(
                sample,
                left_label_width,
                row_y,
                plot_width,
                plot_height,
                ["total_acc_x", "total_acc_y", "total_acc_z"],
            )
        )
        parts.append(f'<text x="{left_label_width}" y="{row_y + plot_height + 20}" font-size="11" font-family="Arial" fill="#666">0s</text>')
        parts.append(f'<text x="{left_label_width + plot_width - 24}" y="{row_y + plot_height + 20}" font-size="11" font-family="Arial" fill="#666">2.56s</text>')

    legend_y = height - 18
    legend_x = 28
    for idx, axis in enumerate(["x", "y", "z"]):
        x = legend_x + idx * 70
        parts.append(f'<line x1="{x}" y1="{legend_y}" x2="{x + 28}" y2="{legend_y}" stroke="{COLORS[axis]}" stroke-width="2"/>')
        parts.append(f'<text x="{x + 34}" y="{legend_y + 4}" font-size="12" font-family="Arial">{axis}</text>')
    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def write_gallery(output_dir: Path, generated: list[tuple[str, Path, Path, list[int]]]) -> None:
    cards = []
    for activity, body_path, total_path, sample_indices in generated:
        body_rel = body_path.name
        total_rel = total_path.name
        cards.append(
            f"<section><h2>{html.escape(activity)}</h2>"
            f"<p>Samples: {', '.join(str(i) for i in sample_indices)}</p>"
            f"<h3>body_acc + body_gyro</h3>"
            f"<img src=\"{html.escape(body_rel)}\" alt=\"{html.escape(activity)} body acceleration and gyroscope samples\"/>"
            f"<h3>total_acc</h3>"
            f"<img src=\"{html.escape(total_rel)}\" alt=\"{html.escape(activity)} total acceleration samples\"/></section>"
        )
    html_text = (
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>UCI HAR IMU Samples</title>"
        "<style>body{font-family:Arial,sans-serif;margin:28px;background:#fbfbf8;color:#222}"
        "section{margin:0 0 44px}img{max-width:100%;border:1px solid #ddd;background:white}"
        "p{color:#555}h3{margin:18px 0 8px;font-size:16px}</style></head><body>"
        "<h1>UCI HAR IMU Sample Gallery</h1>"
        "<p>Each figure shows four training windows from one activity. "
        "The body view emphasizes motion after gravity separation; the total_acc view keeps gravity and is useful for static posture differences.</p>"
        "<section><h2>total_acc summary</h2>"
        "<p>Class-level total_acc axis means and within-window variation. This is easier for cross-class comparison than per-sample curves.</p>"
        "<img src=\"total_acc_summary.svg\" alt=\"total acceleration class summary\"/></section>"
        + "\n".join(cards)
        + "</body></html>"
    )
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")


def main() -> int:
    root = dataset_root(PROJECT_ROOT)
    metadata = load_metadata(root, "train")
    output_dir = PROJECT_ROOT / "reports" / "figures" / "uci_har"
    output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[tuple[str, Path, Path, list[int]]] = []
    for activity in ACTIVITY_ORDER:
        sample_indices = choose_samples(metadata, activity, count=4)
        body_output_path = output_dir / f"{activity.lower()}_samples.svg"
        total_output_path = output_dir / f"{activity.lower()}_total_acc_samples.svg"
        render_body_motion_svg(root, metadata, activity, sample_indices, body_output_path)
        render_total_acc_svg(root, metadata, activity, sample_indices, total_output_path)
        generated.append((activity, body_output_path, total_output_path, sample_indices))
        print(f"{activity}: {sample_indices} -> {body_output_path}")
        print(f"{activity} total_acc: {sample_indices} -> {total_output_path}")

    write_gallery(output_dir, generated)
    print(f"gallery: {output_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
