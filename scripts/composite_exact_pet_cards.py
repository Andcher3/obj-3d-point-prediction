#!/usr/bin/env python3
"""Composite user-supplied card art unchanged over generated pet backplates."""

from __future__ import annotations

from collections import deque
from pathlib import Path
import shutil

from PIL import Image


ROOT = Path(r"D:\Pycharm\post_graduate_project")
OUTPUT_DIR = ROOT / "reports" / "pet_runs" / "2026-07-19-chibi-defect-v3" / "state-bases"
GENERATED = Path(r"C:\Users\Andcher\.codex\generated_images\019f7570-d67d-7281-b69f-73cd17220e92")
TEMP = Path(r"C:\Users\Andcher\AppData\Local\Temp")


def remove_edge_black(image: Image.Image, threshold: int = 42) -> Image.Image:
    """Make only edge-connected near-black source background transparent."""

    rgba = image.convert("RGBA")
    pixels = rgba.load()
    width, height = rgba.size
    seen: set[tuple[int, int]] = set()
    queue: deque[tuple[int, int]] = deque()

    for x in range(width):
        queue.append((x, 0))
        queue.append((x, height - 1))
    for y in range(height):
        queue.append((0, y))
        queue.append((width - 1, y))

    while queue:
        x, y = queue.popleft()
        if (x, y) in seen:
            continue
        seen.add((x, y))
        red, green, blue, alpha = pixels[x, y]
        if alpha == 0 or max(red, green, blue) > threshold:
            continue
        pixels[x, y] = (red, green, blue, 0)
        for next_x, next_y in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= next_x < width and 0 <= next_y < height:
                queue.append((next_x, next_y))

    return rgba


def place_exact_card(
    backplate_path: Path,
    card_path: Path,
    crop_box: tuple[int, int, int, int],
    output_name: str,
    target_height: int,
    left: int,
    top: int,
) -> None:
    backplate = Image.open(backplate_path).convert("RGBA")
    # Preserve the supplied card pixels and remove only edge-connected black
    # screenshot padding. Card text, artwork, frame, and rules remain untouched.
    card = Image.open(card_path).convert("RGBA").crop(crop_box)
    card = remove_edge_black(card)
    target_width = round(card.width * target_height / card.height)
    card = card.resize((target_width, target_height), Image.Resampling.LANCZOS)
    backplate.alpha_composite(card, (left, top))
    backplate.save(OUTPUT_DIR / output_name)
    print(output_name, "backplate", backplate.size, "card", card.size, "at", (left, top))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    place_exact_card(
        GENERATED / "exec-caee44ee-fc49-4576-a4b7-81cc8def179a.png",
        TEMP / "codex-clipboard-3bcc0d73-c531-48c0-8368-7321092381bc.png",
        (6, 12, 317, 442),
        "waiting.png",
        target_height=1210,
        left=15,
        top=120,
    )
    place_exact_card(
        GENERATED / "exec-07c35e55-8bdc-4297-8a1a-6dd84786b711.png",
        TEMP / "codex-clipboard-fef9f8f0-ac56-43b9-8cda-57ae28db2f5d.png",
        (7, 12, 235, 343),
        "review.png",
        target_height=1210,
        left=10,
        top=120,
    )
    place_exact_card(
        GENERATED / "exec-cd3a1178-6b4b-44fb-a59b-904af8b4e31c.png",
        TEMP / "codex-clipboard-35a6260f-50d3-4586-9a96-c3e3570fa4cf.png",
        (0, 0, 180, 230),
        "running.png",
        target_height=1120,
        left=200,
        top=140,
    )

    shutil.copy2(
        GENERATED / "exec-733a5cde-121c-47cb-aa72-019dc995f0f7.png",
        OUTPUT_DIR / "failed.png",
    )

    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
