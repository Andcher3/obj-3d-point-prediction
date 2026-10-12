from pathlib import Path

import cv2
import numpy as np


ROOT = Path("/root/autodl-tmp/projects/TrajLoom-probe")
ROIS = {
    "10790317943435558913": (320, 576, 128, 416),
    "gear_v2_9-wrist_front_rgb_img-0": (384, 480, 64, 256),
}

out_dir = ROOT / "analysis" / "target_roi_overlays"
out_dir.mkdir(parents=True, exist_ok=True)
for stem, (x0, x1, y0, y1) in ROIS.items():
    cap = cv2.VideoCapture(str(ROOT / "inputs" / "videos" / f"{stem}.mp4"))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read {stem}")
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 255), 3)
    for y in range(y0, y1 + 1, 32):
        for x in range(x0, x1 + 1, 32):
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
    cv2.imwrite(str(out_dir / f"{stem}.jpg"), frame)
