from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"D:\Pycharm\post_graduate_project")
OUT_DIR = ROOT / "reports" / "weekly" / "sensorlm_slides"
W, H = 1600, 900


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


BLUE = (27, 70, 140)
LIGHT_BLUE = (231, 239, 251)
MID_BLUE = (82, 132, 199)
TEXT = (20, 28, 38)
MUTED = (88, 98, 112)
RED = (178, 38, 38)
GREEN = (70, 140, 90)
ORANGE = (214, 130, 44)
GRAY = (235, 238, 242)
PALE = (248, 250, 252)


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    lines: list[str] = []
    for para in text.split("\n"):
        if not para:
            lines.append("")
            continue
        cur = ""
        for ch in para:
            trial = cur + ch
            if draw.textlength(trial, font=fnt) <= max_width:
                cur = trial
            else:
                if cur:
                    lines.append(cur)
                cur = ch
        if cur:
            lines.append(cur)
    return lines


def draw_lines(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    lines: Iterable[str],
    fnt: ImageFont.FreeTypeFont,
    fill=TEXT,
    line_gap: int = 12,
) -> int:
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=fnt, fill=fill)
        bbox = draw.textbbox((x, y), line, font=fnt)
        y += bbox[3] - bbox[1] + line_gap
    return y


def header(draw: ImageDraw.ImageDraw, title: str, kicker: str | None = None) -> None:
    draw.text((70, 38), title, font=font(42, True), fill=TEXT)
    if kicker:
        draw.text((72, 92), kicker, font=font(20), fill=MUTED)
    draw.rectangle((0, 126, W, 134), fill=BLUE)


def footer(draw: ImageDraw.ImageDraw) -> None:
    draw.rectangle((0, 855, W, 900), fill=(28, 32, 38))
    draw.rectangle((805, 855, W, 900), fill=(21, 132, 202))
    draw.text((70, 867), "SensorLM | wearable sensor-language representation", font=font(18), fill=(230, 232, 235))
    draw.text((1165, 867), "方向一：IMU 行为语义理解", font=font(18, True), fill=(255, 255, 255))


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill, outline=None, radius=18, width=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], fill=BLUE, width=4):
    draw.line((start, end), fill=fill, width=width)
    ex, ey = end
    sx, sy = start
    if ex >= sx:
        pts = [(ex, ey), (ex - 14, ey - 8), (ex - 14, ey + 8)]
    else:
        pts = [(ex, ey), (ex + 14, ey - 8), (ex + 14, ey + 8)]
    draw.polygon(pts, fill=fill)


def slide1() -> Image.Image:
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    header(d, "SensorLM：让可穿戴传感器进入语言语义空间", "Learning the Language of Wearable Sensors, 2025")

    d.text((88, 176), "传统 HAR", font=font(26, True), fill=BLUE)
    rounded(d, (88, 224, 625, 300), fill=PALE, outline=GRAY, radius=14)
    d.text((120, 244), "sensor time series  →  activity label", font=font(27, True), fill=TEXT)

    d.text((88, 372), "这篇论文的问题意识", font=font(26, True), fill=BLUE)
    lines = wrap(d, "传感器信号不是天然语言，通用 LLM 也不会自动理解表格化的连续时间序列。SensorLM 希望先建立 sensor-language alignment，让传感器表示进入人能理解的语义空间。", font(27), 670)
    draw_lines(d, (90, 420), lines, font(27), fill=TEXT, line_gap=13)

    rounded(d, (820, 190, 1460, 650), fill=LIGHT_BLUE, outline=(204, 220, 244), radius=22)
    d.text((860, 230), "数据与目标", font=font(30, True), fill=BLUE)
    items = [
        ("输入", "26 个分钟级特征 × 1440 分钟"),
        ("规模", "59.7M+ 小时，103K+ 用户"),
        ("训练", "sensor ↔ text 对齐 + caption 生成"),
        ("能力", "zero-shot / few-shot / 检索 / 生成描述"),
    ]
    y = 300
    for k, v in items:
        d.text((870, y), k, font=font(24, True), fill=BLUE)
        d.text((970, y), v, font=font(24), fill=TEXT)
        y += 72

    rounded(d, (88, 690, 1460, 810), fill=(255, 249, 235), outline=(236, 205, 141), radius=16)
    d.text((122, 718), "可以用于我们方向一的视角", font=font(25, True), fill=ORANGE)
    d.text((122, 760), "手机 IMU 不只作为分类输入，也可以学习与人体动作、手机运镜和场景语义对齐的表示。", font=font(25, True), fill=TEXT)
    footer(d)
    return img


def slide2() -> Image.Image:
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    header(d, "方法：先自动生成 caption，再做跨模态预训练")

    # Caption generation side
    d.text((80, 174), "三层 caption", font=font(28, True), fill=BLUE)
    boxes = [
        ((80, 230, 420, 330), "Statistical", "mean / max / min / std", BLUE),
        ((80, 360, 420, 460), "Structural", "trend / spike / drop", ORANGE),
        ((80, 490, 420, 590), "Semantic", "Walk / Sleep / Outdoor Bike", GREEN),
    ]
    for box, title, sub, color in boxes:
        rounded(d, box, fill=PALE, outline=color, radius=14, width=3)
        d.text((box[0] + 24, box[1] + 20), title, font=font(26, True), fill=color)
        d.text((box[0] + 24, box[1] + 60), sub, font=font(21), fill=TEXT)

    arrow(d, (430, 410), (545, 410), fill=BLUE)

    # Joint model
    rounded(d, (560, 190, 1450, 740), fill=(250, 252, 255), outline=(210, 222, 240), radius=22)
    d.text((600, 222), "SensorLM 预训练框架", font=font(30, True), fill=BLUE)

    rounded(d, (610, 305, 900, 385), fill=(239, 233, 255), outline=(178, 153, 222), radius=14)
    rounded(d, (610, 475, 900, 555), fill=(239, 233, 255), outline=(178, 153, 222), radius=14)
    d.text((655, 328), "Sensor Encoder", font=font(25, True), fill=TEXT)
    d.text((665, 498), "Text Encoder", font=font(25, True), fill=TEXT)

    rounded(d, (1080, 388, 1390, 478), fill=(255, 238, 218), outline=(224, 168, 100), radius=14)
    d.text((1116, 416), "Multimodal Text Decoder", font=font(23, True), fill=TEXT)

    d.text((620, 280), "26 × 1440 sensor matrix", font=font(20), fill=MUTED)
    d.text((620, 450), "caption tokens", font=font(20), fill=MUTED)
    arrow(d, (900, 345), (1060, 425), fill=MID_BLUE)
    arrow(d, (900, 515), (1060, 445), fill=MID_BLUE)

    rounded(d, (1035, 570, 1390, 660), fill=(235, 250, 240), outline=(160, 210, 175), radius=14)
    d.text((1078, 594), "Contrastive loss", font=font(23, True), fill=GREEN)
    d.text((1078, 627), "匹配 pair 更近，不匹配更远", font=font(20), fill=TEXT)

    rounded(d, (610, 620, 960, 710), fill=(255, 247, 235), outline=(230, 200, 150), radius=14)
    d.text((650, 644), "Captioning loss", font=font(23, True), fill=ORANGE)
    d.text((650, 677), "逐 token 预测原 caption", font=font(20), fill=TEXT)

    d.text((600, 770), "核心：CLIP-like 负责对齐，Cap-like 负责生成，CoCa-like 同时做两件事。", font=font(24, True), fill=TEXT)
    footer(d)
    return img


def draw_bar_chart(d: ImageDraw.ImageDraw, origin: tuple[int, int], size: tuple[int, int], labels, vals, colors, title: str, maxv=1.0):
    x, y = origin
    w, h = size
    d.text((x, y - 45), title, font=font(25, True), fill=BLUE)
    d.line((x, y + h, x + w, y + h), fill=(120, 130, 145), width=2)
    bw = w // (len(vals) * 2)
    gap = bw
    for i, (lab, val, col) in enumerate(zip(labels, vals, colors)):
        bx = x + i * (bw + gap) + 18
        bh = int(h * val / maxv)
        d.rounded_rectangle((bx, y + h - bh, bx + bw, y + h), radius=6, fill=col)
        d.text((bx - 2, y + h - bh - 34), f"{val:.2f}", font=font(20, True), fill=TEXT)
        lines = wrap(d, lab, font(17), bw + 28)
        draw_lines(d, (bx - 8, y + h + 12), lines, font(17), fill=MUTED, line_gap=3)


def slide3() -> Image.Image:
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    header(d, "结果：优势主要体现在零样本、少标签和检索")

    draw_bar_chart(
        d,
        (95, 230),
        (520, 260),
        ["Gemma", "Gemini", "SensorLM"],
        [0.50, 0.51, 0.84],
        [(170, 178, 190), (150, 158, 170), BLUE],
        "Zero-shot 20-class Activity AUROC",
    )

    draw_bar_chart(
        d,
        (780, 230),
        (600, 260),
        ["DINO", "MSN", "SimCLR", "SensorLM"],
        [0.54, 0.60, 0.60, 0.88],
        [(170, 178, 190), (150, 158, 170), (130, 142, 155), BLUE],
        "Few-shot Activity AUROC (50 labels/class)",
    )

    rounded(d, (94, 610, 700, 780), fill=PALE, outline=GRAY, radius=18)
    d.text((124, 638), "Caption 类型消融", font=font(26, True), fill=BLUE)
    d.text((124, 690), "statistical: 0.51", font=font(22), fill=MUTED)
    d.text((380, 690), "structural: 0.50", font=font(22), fill=MUTED)
    d.text((124, 730), "structural + semantic: 0.84", font=font(25, True), fill=GREEN)

    rounded(d, (780, 610, 1385, 780), fill=PALE, outline=GRAY, radius=18)
    d.text((810, 638), "CLIP-like 与 CoCa-like", font=font(26, True), fill=BLUE)
    d.text((810, 690), "zero-shot AUROC: 0.83 vs 0.84", font=font(23), fill=TEXT)
    d.text((810, 730), "分类主要靠 contrastive alignment；CoCa 额外支持生成。", font=font(22), fill=MUTED)

    footer(d)
    return img


def slide4() -> Image.Image:
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    header(d, "方向一可以怎样接上")

    y = 205
    items = [
        ("从固定标签到语义表示", "手机 IMU / 运镜信号可以先进入可解释的行为语义空间，再接具体任务头。"),
        ("SensorLM 的位置", "它验证了大规模弱语义文本能提升少标签活动识别，但本身偏长期手表数据。"),
        ("下一步阅读", "SensorLLM 更直接面向 motion sensor HAR；Video-IMU MAE 更贴近相机视角与 IMU 融合。"),
    ]
    for title, desc in items:
        rounded(d, (110, y, 1460, y + 120), fill=PALE, outline=(218, 225, 235), radius=18)
        d.text((145, y + 24), title, font=font(28, True), fill=BLUE)
        lines = wrap(d, desc, font(24), 1000)
        draw_lines(d, (520, y + 26), lines, font(24), fill=TEXT, line_gap=8)
        y += 150

    rounded(d, (260, 675, 1300, 790), fill=(255, 249, 235), outline=(238, 207, 145), radius=18)
    d.text((300, 704), "本周汇报主线", font=font(26, True), fill=ORANGE)
    d.text((300, 748), "从传统 HAR 分类，转向 sensor-language alignment 对方向一的启发。", font=font(27, True), fill=TEXT)

    footer(d)
    return img


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slides = [slide1(), slide2(), slide3(), slide4()]
    for i, img in enumerate(slides, 1):
        img.save(OUT_DIR / f"sensorlm_slide_{i:02d}.png")
    print(f"saved {len(slides)} slides to {OUT_DIR}")


if __name__ == "__main__":
    main()
