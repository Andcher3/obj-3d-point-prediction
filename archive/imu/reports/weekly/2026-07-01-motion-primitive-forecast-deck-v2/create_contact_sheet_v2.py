from pathlib import Path

from PIL import Image, ImageDraw


deck_dir = Path(
    r"D:\Pycharm\post_graduate_project\reports\weekly\2026-07-01-motion-primitive-forecast-deck-v2"
)
slides = [deck_dir / "slides" / f"slide_{i:02d}.png" for i in range(1, 11)]

thumb_w, thumb_h = 400, 225
sheet = Image.new("RGB", (thumb_w * 2, thumb_h * 5), (245, 247, 250))
draw = ImageDraw.Draw(sheet)

for idx, slide_path in enumerate(slides):
    image = Image.open(slide_path).convert("RGB")
    image = image.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
    x = (idx % 2) * thumb_w
    y = (idx // 2) * thumb_h
    sheet.paste(image, (x, y))
    draw.rectangle((x, y, x + thumb_w - 1, y + thumb_h - 1), outline=(180, 190, 200), width=2)
    draw.text((x + 8, y + 8), str(idx + 1), fill=(10, 40, 80))

sheet.save(deck_dir / "contact_sheet_v2.png")
print(deck_dir / "contact_sheet_v2.png")
