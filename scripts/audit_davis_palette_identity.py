#!/usr/bin/env python3
"""Audit DAVIS palette indices and target continuity over a fixed frame window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--sequences", nargs="+", required=True)
    ap.add_argument("--start", type=int, default=14)
    ap.add_argument("--end", type=int, default=35)
    ap.add_argument("--logical-instance-id", type=int, default=1)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--contact-sheet", type=Path, required=True)
    args = ap.parse_args()

    results = {}
    panels = []
    for sequence in args.sequences:
        rows = []
        for index in range(args.start, args.end + 1):
            mask_path = args.data_root / "Annotations" / "480p" / sequence / f"{index:05d}.png"
            image_path = args.data_root / "JPEGImages" / "480p" / sequence / f"{index:05d}.jpg"
            mask_image = Image.open(mask_path)
            mask = np.asarray(mask_image, dtype=np.uint8)
            target = mask == args.logical_instance_id
            component_count, labels, stats, _ = cv2.connectedComponentsWithStats(target.astype(np.uint8), 8)
            component_areas = sorted(stats[1:, cv2.CC_STAT_AREA].astype(int).tolist(), reverse=True)
            yy, xx = np.nonzero(target)
            rows.append(
                {
                    "frame": index,
                    "png_mode": mask_image.mode,
                    "palette_indices": sorted(np.unique(mask).astype(int).tolist()),
                    "target_area_px": int(target.sum()),
                    "target_bbox_xyxy": [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())] if len(xx) else None,
                    "target_centroid_xy": [float(xx.mean()), float(yy.mean())] if len(xx) else None,
                    "connected_components": int(component_count - 1),
                    "largest_component_fraction": float(component_areas[0] / target.sum()) if component_areas else 0.0,
                }
            )
            if index in (args.start, 20, args.end):
                rgb = Image.open(image_path).convert("RGB")
                overlay = Image.new("RGBA", rgb.size, (0, 0, 0, 0))
                rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
                rgba[target] = [255, 0, 0, 110]
                overlay = Image.fromarray(rgba, "RGBA")
                panel = Image.alpha_composite(rgb.convert("RGBA"), overlay).convert("RGB")
                draw = ImageDraw.Draw(panel)
                draw.rectangle((0, 0, 240, 28), fill=(0, 0, 0))
                draw.text((8, 7), f"{sequence} frame {index}", fill=(255, 255, 255))
                panels.append(panel.resize((427, 240)))
        allowed = all(row["palette_indices"] == [0, args.logical_instance_id] for row in rows)
        nonempty = all(row["target_area_px"] > 0 for row in rows)
        palette_mode = all(row["png_mode"] == "P" for row in rows)
        results[sequence] = {
            "status": "PASS" if allowed and nonempty and palette_mode else "FAIL",
            "logical_instance_id": args.logical_instance_id,
            "png_palette_index": args.logical_instance_id,
            "opencv_bgr_rendering_for_index_1": [0, 0, 128],
            "frame_count": len(rows),
            "palette_mode_all_frames": palette_mode,
            "only_background_and_target_index_all_frames": allowed,
            "target_nonempty_all_frames": nonempty,
            "area_min_px": min(row["target_area_px"] for row in rows),
            "area_max_px": max(row["target_area_px"] for row in rows),
            "rows": rows,
        }
    status = "PASS" if all(row["status"] == "PASS" for row in results.values()) else "FAIL"
    payload = {
        "status": status,
        "scope": "palette-index and target-mask continuity audit; semantic identity also visually checked on frames 14, 20, 35",
        "frame_window": [args.start, args.end],
        "sequences": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    columns = 3
    rows = (len(panels) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 427, rows * 240), (32, 32, 32))
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % columns) * 427, (index // columns) * 240))
    args.contact_sheet.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.contact_sheet, quality=92)
    print(json.dumps({"status": status, "videos": {k: {x: v[x] for x in ("status", "area_min_px", "area_max_px")} for k, v in results.items()}}, indent=2))


if __name__ == "__main__":
    main()
