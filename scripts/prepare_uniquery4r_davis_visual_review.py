"""CPU-only RGB/mask bundle and deterministic t0 query previews; no model inference."""
import argparse
import colorsys
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

VIDEOS = ("car-turn", "drift-turn", "boat", "soccerball")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sample(mask, count):
    ys, xs = np.nonzero(mask)
    pts = np.stack((xs, ys), axis=1).astype(np.float64)
    if len(pts) < count:
        raise ValueError("Not enough interior pixels")
    first = int(np.argmin(((pts - pts.mean(axis=0)) ** 2).sum(axis=1)))
    chosen = [first]
    minimum = ((pts - pts[first]) ** 2).sum(axis=1)
    while len(chosen) < count:
        minimum[chosen] = -1
        idx = int(np.argmax(minimum))
        chosen.append(idx)
        minimum = np.minimum(minimum, ((pts - pts[idx]) ** 2).sum(axis=1))
    return pts[chosen]


def color(idx):
    return tuple(round(c * 255) for c in colorsys.hsv_to_rgb(idx / 48, 0.8, 1.0))


def draw_points(im, uv, scale=1.0, offset=(0, 0), numbers=False):
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 15)
    for i, (u, v) in enumerate(uv):
        x, y = (u - offset[0]) * scale, (v - offset[1]) * scale
        radius = 3 if not numbers else 4
        draw.ellipse((x-radius-1, y-radius-1, x+radius+1, y+radius+1), fill="black")
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color(i))
        if numbers:
            draw.text((x+5, y-12), str(i), fill=color(i), font=font,
                      stroke_width=2, stroke_fill="black")


def render(image, mask, queries, title, output):
    rgb = np.asarray(image).copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, contours, -1, (255, 230, 0), 1)
    full = Image.fromarray(rgb)
    draw_points(full, queries)
    ys, xs = np.nonzero(mask)
    margin = max(12, round(max(xs.max()-xs.min(), ys.max()-ys.min()) * 0.12))
    x0, y0 = max(0, int(xs.min())-margin), max(0, int(ys.min())-margin)
    x1, y1 = min(image.width, int(xs.max())+margin+1), min(image.height, int(ys.max())+margin+1)
    crop = Image.fromarray(rgb).crop((x0, y0, x1, y1))
    zoom = min(600 / crop.width, 470 / crop.height)
    crop = crop.resize((round(crop.width*zoom), round(crop.height*zoom)), Image.Resampling.LANCZOS)
    draw_points(crop, queries, scale=zoom, offset=(x0, y0), numbers=True)
    canvas = Image.new("RGB", (image.width + 620, 540), "#20242b")
    canvas.paste(full, (0, 50))
    canvas.paste(crop, (image.width+10, 50))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 19)
    draw.text((10, 12), title, fill="white", font=font)
    draw.text((image.width+10, 518), "Zoom: query IDs 0-47; yellow = target mask", fill="white", font=font)
    canvas.save(output)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--davis-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--soccerball-t0", type=int, default=14)
    ap.add_argument("--approval-recorded", action="store_true",
                    help="Use only after the human has approved these exact query previews")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    previews = args.output / "previews"
    previews.mkdir(exist_ok=True)
    inventory, images, entries = {}, [], []
    for video in VIDEOS:
        t0 = args.soccerball_t0 if video == "soccerball" else 20
        frames = list(range(t0-6, t0+16))
        dest = args.output / "inputs" / video
        for kind in ("rgb", "mask"):
            (dest / kind).mkdir(parents=True, exist_ok=True)
        for frame in frames:
            for src_kind, dst_kind, suffix in (("JPEGImages", "rgb", "jpg"),
                                               ("Annotations", "mask", "png")):
                src = args.davis_root / src_kind / "480p" / video / f"{frame:05d}.{suffix}"
                dst = dest / dst_kind / src.name
                shutil.copy2(src, dst)
                inventory[str(dst.relative_to(args.output)).replace("\\", "/")] = sha256(dst)
        original = Image.open(dest / "rgb" / f"{t0:05d}.jpg").convert("RGB")
        palette = np.asarray(Image.open(dest / "mask" / f"{t0:05d}.png"))
        if palette.ndim != 2:
            raise ValueError("Expected original indexed DAVIS mask")
        target = palette == 1
        w = 504
        h = round(original.height * w / original.width / 14) * 14
        proc = cv2.resize(target.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        interior = cv2.erode(proc, np.ones((5, 5), np.uint8)).astype(bool)
        candidates = sample(interior, 48)
        # PIL resize pixel-center convention, also used by UniQuery4R preprocessing.
        original_uv = (candidates + 0.5) * np.array([original.width/w, original.height/h]) - 0.5
        source_px = np.rint(original_uv).astype(int)
        if not target[source_px[:, 1], source_px[:, 0]].all():
            raise ValueError("Source-image roundtrip query lies outside target")
        manifest = {
            "video_id": video, "status": "USER_APPROVED_AWAITING_GPU" if args.approval_recorded else "AWAITING_USER_QUERY_APPROVAL",
            "target_palette_id": 1, "query_count": 48,
            "source_frame_indices": frames,
            "observed_source_frame_indices": frames[:7],
            "query_source_frame_index": t0, "query_model_frame_index": 6,
            "source_resolution_hw": [original.height, original.width],
            "processed_resolution_hw": [h, w],
            "query_uv_source_pixels": original_uv.tolist(),
            "query_uv_processed_pixels": candidates.tolist(),
            "query_uv_normalized_01": (candidates / np.array([w-1, h-1])).tolist(),
            "selection": "deterministic pixel-space farthest-point sampling on t0 target mask eroded 2 processed pixels",
            "source_image_sha256": sha256(dest / "rgb" / f"{t0:05d}.jpg"),
            "source_mask_sha256": sha256(dest / "mask" / f"{t0:05d}.png"),
            "input_placement_check": "48/48 in t0 source target mask; no prediction metric computed",
            "half_pixel_mapping": "source=(processed+0.5)*source_size/processed_size-0.5",
        }
        mp = dest / "queries.json"
        mp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        inventory[str(mp.relative_to(args.output)).replace("\\", "/")] = sha256(mp)
        pp = previews / f"{video}_query_t0.png"
        images.append(render(original, target, original_uv,
                             f"{video} | t0 = source frame {t0} / model frame 6 | 48 points", pp))
        entries.append({"video_id": video, "manifest": str(mp), "preview": str(pp)})
    sheet = Image.new("RGB", (images[0].width * 2, images[0].height * 2), "#20242b")
    for i, im in enumerate(images):
        sheet.paste(im, ((i % 2)*im.width, (i//2)*im.height))
    sheet.save(previews / "four_videos_query_t0.png")
    (args.output / "input_sha256.json").write_text(json.dumps(inventory, indent=2) + "\n")
    (args.output / "preparation_status.json").write_text(json.dumps({
        "status": "QUERY_APPROVED_AWAITING_GPU_ENABLE" if args.approval_recorded else "AWAITING_USER_QUERY_APPROVAL_AND_GPU_ENABLE",
        "query_approved": args.approval_recorded, "inference_started": False,
        "numerical_quality_evaluation_started": False,
        "videos": entries}, indent=2) + "\n")
    with tarfile.open(args.output / "input_bundle.tar.gz", "w:gz") as tar:
        # Only upload manifest-listed frames; earlier preview windows remain local.
        for rel in inventory:
            tar.add(args.output / rel, arcname=rel)
        tar.add(args.output / "input_sha256.json", arcname="input_sha256.json")
        tar.add(args.output / "preparation_status.json", arcname="preparation_status.json")
    print(json.dumps({"status": "QUERY_APPROVED_AWAITING_GPU_ENABLE" if args.approval_recorded
                     else "PREVIEWS_READY_NOT_APPROVED", "videos": entries}, indent=2))


if __name__ == "__main__":
    main()
