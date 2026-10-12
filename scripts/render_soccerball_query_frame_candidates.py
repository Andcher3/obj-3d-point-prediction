"""Render source frames for choosing a visible ball anchor, no tracking model."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("--davis-root", type=Path, required=True)
ap.add_argument("--output", type=Path, required=True)
args = ap.parse_args()
panels = []
font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 22)
for index in range(8, 21):
    im = Image.open(args.davis_root / "JPEGImages/480p/soccerball" / f"{index:05d}.jpg").convert("RGB")
    mask = np.asarray(Image.open(args.davis_root / "Annotations/480p/soccerball" / f"{index:05d}.png")) == 1
    ys, xs = np.nonzero(mask)
    pad = 30
    bounds = (max(0, int(xs.min())-pad), max(0, int(ys.min())-pad),
              min(im.width, int(xs.max())+pad+1), min(im.height, int(ys.max())+pad+1))
    crop = im.crop(bounds)
    crop.thumbnail((340, 250), Image.Resampling.LANCZOS)
    # Enlarge the crop so the object and occluder can be inspected.
    scale = min(340/crop.width, 250/crop.height)
    crop = crop.resize((round(crop.width*scale), round(crop.height*scale)), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (350, 290), "#20242b")
    panel.paste(crop, (5, 35))
    ImageDraw.Draw(panel).text((8, 5), f"Source frame {index}", fill="white", font=font)
    panels.append(panel)
sheet = Image.new("RGB", (350*4, 290*4), "#20242b")
for i, panel in enumerate(panels):
    sheet.paste(panel, ((i%4)*350, (i//4)*290))
args.output.parent.mkdir(parents=True, exist_ok=True)
sheet.save(args.output)
print(args.output)
