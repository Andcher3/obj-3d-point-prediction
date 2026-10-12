from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


CELL_W = 192
CELL_H = 208
ATLAS_W = 1536
ATLAS_H = 2288
JUMPING_ROW = 4
JUMPING_FRAMES = 5
ANIMATION_ROWS = {
    "idle": (0, 6),
    "running-right": (1, 8),
    "running-left": (2, 8),
    "waving": (3, 4),
    "jumping": (4, 5),
    "failed": (5, 8),
    "waiting": (6, 6),
    "running": (7, 6),
    "review": (8, 6),
}


def rgba_bytes(image: Image.Image) -> bytes:
    return image.convert("RGBA").tobytes()


def normalize_transparent_rgb(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    pixels = bytearray(rgba.tobytes())
    for offset in range(0, len(pixels), 4):
        if pixels[offset + 3] == 0:
            pixels[offset] = 0
            pixels[offset + 1] = 0
            pixels[offset + 2] = 0
    return Image.frombytes("RGBA", rgba.size, bytes(pixels))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-atlas", required=True, type=Path)
    parser.add_argument("--frames-dir", required=True, type=Path)
    parser.add_argument("--png-output", required=True, type=Path)
    parser.add_argument("--webp-output", required=True, type=Path)
    parser.add_argument("--preservation-json", required=True, type=Path)
    parser.add_argument("--metrics-json", required=True, type=Path)
    parser.add_argument("--qa-frames-root", type=Path)
    args = parser.parse_args()

    base = Image.open(args.base_atlas).convert("RGBA")
    if base.size != (ATLAS_W, ATLAS_H):
        raise SystemExit(f"Unexpected atlas size: {base.size}")

    atlas = base.copy()
    clear = Image.new("RGBA", (ATLAS_W, CELL_H), (0, 0, 0, 0))
    atlas.paste(clear, (0, JUMPING_ROW * CELL_H))

    frame_metrics = []
    for index in range(JUMPING_FRAMES):
        path = args.frames_dir / f"{index:02d}.png"
        frame = Image.open(path).convert("RGBA")
        if frame.size != (CELL_W, CELL_H):
            raise SystemExit(f"Unexpected frame size for {path}: {frame.size}")
        frame = normalize_transparent_rgb(frame)
        atlas.alpha_composite(frame, (index * CELL_W, JUMPING_ROW * CELL_H))
        alpha = frame.getchannel("A")
        bbox = alpha.getbbox()
        frame_metrics.append(
            {
                "index": index,
                "bbox": list(bbox) if bbox else None,
                "nontransparent_pixels": sum(1 for value in alpha.getdata() if value),
            }
        )

    atlas = normalize_transparent_rgb(atlas)
    args.png_output.parent.mkdir(parents=True, exist_ok=True)
    args.webp_output.parent.mkdir(parents=True, exist_ok=True)
    atlas.save(args.png_output, format="PNG", optimize=True)
    atlas.save(args.webp_output, format="WEBP", lossless=True, method=6, exact=True)

    if args.qa_frames_root:
        for state, (row, frame_count) in ANIMATION_ROWS.items():
            state_dir = args.qa_frames_root / state
            state_dir.mkdir(parents=True, exist_ok=True)
            for index in range(frame_count):
                box = (
                    index * CELL_W,
                    row * CELL_H,
                    (index + 1) * CELL_W,
                    (row + 1) * CELL_H,
                )
                normalize_transparent_rgb(atlas.crop(box)).save(
                    state_dir / f"{index:02d}.png", format="PNG", optimize=True
                )

    changed_rows = []
    unchanged_rows = []
    for row in range(11):
        box = (0, row * CELL_H, ATLAS_W, (row + 1) * CELL_H)
        if rgba_bytes(base.crop(box)) == rgba_bytes(atlas.crop(box)):
            unchanged_rows.append(row)
        else:
            changed_rows.append(row)

    preservation = {
        "ok": changed_rows == [JUMPING_ROW],
        "source_atlas": str(args.base_atlas),
        "changed_rows": changed_rows,
        "unchanged_rows": unchanged_rows,
        "method": "Decoded RGBA row comparison over each complete 1536x208 atlas row",
        "note": "Only the jumping row may differ; all other rows must remain pixel-identical.",
        "source_sha256": hashlib.sha256(args.base_atlas.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(args.webp_output.read_bytes()).hexdigest(),
    }
    args.preservation_json.write_text(
        json.dumps(preservation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.metrics_json.write_text(
        json.dumps(
            {
                "frame_count": JUMPING_FRAMES,
                "atlas_unused_columns": [5, 6, 7],
                "frames": frame_metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if not preservation["ok"]:
        raise SystemExit(f"Unexpected changed rows: {changed_rows}")


if __name__ == "__main__":
    main()
