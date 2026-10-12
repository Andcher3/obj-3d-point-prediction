from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DECK_DIR = Path(
    r"D:\Pycharm\post_graduate_project\reports\weekly\2026-07-01-motion-primitive-forecast-deck-v2"
)
SLIDES_DIR = DECK_DIR / "slides"

W, H = 1600, 900

NAVY = "#12386b"
NAVY2 = "#0f2f59"
TEAL = "#0f766e"
CYAN = "#0ea5b7"
BLUE = "#2563eb"
TEXT = "#1f2937"
MUTED = "#52616f"
GRAY = "#d9e2ec"
PALE_BLUE = "#eef6fb"
WHITE = "#ffffff"

FONT_REG = r"C:\Windows\Fonts\msyh.ttc"
FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size=size)


def blank_slide() -> Image.Image:
    return Image.new("RGBA", (W, H), (255, 255, 255, 255))


def draw_text(draw: ImageDraw.ImageDraw, xy, text: str, size: int, fill=TEXT, bold=False, anchor=None):
    draw.text(xy, text, font=font(size, bold), fill=fill, anchor=anchor)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, max_width: int, size: int, bold=False):
    f = font(size, bold)
    lines: list[str] = []
    current = ""
    for char in text:
        test = current + char
        if draw.textlength(test, font=f) <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = char
    if current:
        lines.append(current)
    return lines


def bullet_lines(draw: ImageDraw.ImageDraw, x: int, y: int, items: list[str], width: int, *, size=23, gap=44, color=TEXT, bullet_color=CYAN):
    yy = y
    for item in items:
        wrapped = wrap_text(draw, item, width - 34, size)
        draw.ellipse((x, yy - 15, x + 10, yy - 5), fill=bullet_color)
        for i, line in enumerate(wrapped):
            draw_text(draw, (x + 28, yy + i * (size + 7)), line, size, color)
        yy += max(gap, len(wrapped) * (size + 7) + 14)


def panel(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, title: str, accent=NAVY):
    draw.rounded_rectangle((x, y, x + w, y + h), radius=8, fill=(255, 255, 255, 255), outline=GRAY, width=2)
    draw.rounded_rectangle((x, y, x + 8, y + h), radius=4, fill=accent)
    draw_text(draw, (x + 28, y + 45), title, 29, accent, bold=True)


def header(draw: ImageDraw.ImageDraw, title: str, subtitle: str):
    draw.line((82, 72, 1518, 72), fill=NAVY, width=4)
    draw.line((82, 84, 1518, 84), fill=CYAN, width=2)
    draw_text(draw, (92, 132), title, 42, NAVY2, bold=True)
    draw_text(draw, (94, 174), subtitle, 23, MUTED)


def flow(draw: ImageDraw.ImageDraw, x: int, y: int, labels: list[str], accent=TEAL):
    box_w, box_h, gap = 122, 44, 24
    for i, label in enumerate(labels):
        xx = x + i * (box_w + gap)
        draw.rounded_rectangle((xx, y, xx + box_w, y + box_h), radius=10, fill=PALE_BLUE, outline=accent, width=2)
        draw_text(draw, (xx + box_w // 2, y + 29), label, 19, NAVY, bold=True, anchor="mm")
        if i < len(labels) - 1:
            x1 = xx + box_w + 6
            x2 = xx + box_w + gap - 8
            yy = y + box_h // 2
            draw.line((x1, yy, x2, yy), fill=accent, width=3)
            draw.polygon([(x2, yy), (x2 - 10, yy - 6), (x2 - 10, yy + 6)], fill=accent)


def table(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, rows: list[list[str]], col_fracs: list[float]):
    header_h, row_h = 44, 46
    h = header_h + (len(rows) - 1) * row_h
    draw.rounded_rectangle((x, y, x + w, y + h), radius=10, fill=(255, 255, 255, 240), outline=GRAY, width=2)
    draw.rounded_rectangle((x, y, x + w, y + header_h), radius=10, fill=NAVY2)
    col_w = [int(w * frac) for frac in col_fracs]
    col_w[-1] = w - sum(col_w[:-1])
    cx = x
    for i, head in enumerate(rows[0]):
        draw_text(draw, (cx + 16, y + 28), head, 19, WHITE, bold=True)
        cx += col_w[i]
    for r, row in enumerate(rows[1:]):
        ry = y + header_h + r * row_h
        draw.line((x, ry, x + w, ry), fill=GRAY, width=1)
        cx = x
        for c, cell in enumerate(row):
            draw_text(draw, (cx + 16, ry + 29), cell, 17, NAVY if c == 0 else TEXT, bold=(c == 0))
            cx += col_w[c]
    bx = x
    for cw in col_w[:-1]:
        bx += cw
        draw.line((bx, y, bx, y + h), fill=GRAY, width=1)


def slide_1():
    img = blank_slide()
    d = ImageDraw.Draw(img)
    header(d, "SensorLM 与 SensorLLM：传感器-语言对齐", "关注：数据形态、模型结构、训练目标，以及它们对 HAR 的实际价值")
    panel(d, 86, 220, 690, 390, "SensorLM: wearable sensor-language model", NAVY)
    bullet_lines(
        d,
        124,
        294,
        [
            "输入：Fitbit / Pixel Watch 的多模态 wearable 特征",
            "样本：26 个分钟级特征 × 1440 min，表示一天",
            "文本：statistical / structural / semantic 三层 caption",
            "结构：sensor encoder + text encoder + multimodal decoder",
            "目标：sensor-text 对比学习 + sensor-to-text 生成",
        ],
        615,
    )
    flow(d, 178, 608, ["sensor", "caption", "align", "generate"], CYAN)
    panel(d, 824, 220, 690, 390, "SensorLLM: LLM + motion sensors", TEAL)
    bullet_lines(
        d,
        862,
        294,
        [
            "输入：短窗口 motion sensor / IMU 序列",
            "思路：训练 sensor encoder，再对齐到 LLM 表示空间",
            "任务：HAR 分类、趋势理解、文本式解释",
            "价值：通用 LLM 不能直接理解传感器，需要 alignment",
            "限制：主要仍服务于识别和解释，不直接输出未来序列",
        ],
        615,
        bullet_color=TEAL,
    )
    flow(d, 916, 608, ["IMU", "encoder", "LLM space", "HAR"], TEAL)
    table(
        d,
        86,
        672,
        1428,
        [
            ["论文", "解决什么", "可借鉴点", "主要局限"],
            ["SensorLM", "wearable 数据与自然语言对齐", "caption 构造；zero / few-shot", "日级统计特征，不是手机短窗 IMU"],
            ["SensorLLM", "让 LLM 理解 motion sensor", "sensor-LLM alignment", "偏语义解释，未来预测能力有限"],
        ],
        [0.15, 0.32, 0.25, 0.28],
    )
    return img


def slide_2():
    img = blank_slide()
    d = ImageDraw.Draw(img)
    header(d, "EVI-MAE 与 P2LHAP：自监督重构与 Patch-to-Label", "关注：无标签预训练、连续传感器流、标签序列输出和 smoothing")
    panel(d, 86, 220, 690, 405, "EVI-MAE: video-IMU masked autoencoder", NAVY)
    bullet_lines(
        d,
        124,
        294,
        [
            "输入：第一视角视频 + body-worn IMU",
            "机制：随机遮盖 video token 和 IMU token",
            "训练：重构被遮盖的多模态 token",
            "用途：为 egocentric action recognition 学表示",
            "启发：无标签数据可用于动作表征预训练",
        ],
        615,
        gap=48,
    )
    flow(d, 154, 624, ["video/IMU", "mask", "encoder", "reconstruct"], CYAN)
    panel(d, 824, 220, 690, 405, "P2LHAP: Patch-to-Label Seq2Seq", TEAL)
    bullet_lines(
        d,
        862,
        294,
        [
            "输入：连续 wearable sensor stream",
            "切分：按 patch 建模，不把单个窗口当完整活动",
            "结构：channel-independent Transformer Seq2Seq",
            "输出：patch-level activity label sequence",
            "任务：HAR / HAS / HAF 统一为标签序列预测",
        ],
        615,
        gap=48,
        bullet_color=TEAL,
    )
    flow(d, 884, 624, ["stream", "patches", "Seq2Seq", "labels"], TEAL)
    d.rounded_rectangle((86, 724, 1514, 834), radius=12, fill=(255, 255, 255, 240), outline=GRAY, width=2)
    draw_text(d, (116, 766), "三个关键收获", 26, NAVY, bold=True)
    cols = [(332, "forecast 可以直接预测未来标签序列", "不一定先预测未来 IMU 再分类"),
            (806, "smoothing 抑制 over-segmentation", "减少孤立错误 patch 造成的碎片边界"),
            (1190, "边界由 label sequence 派生", "首次标签变化即可读出切换时机")]
    for x, title, sub in cols:
        draw_text(d, (x, 766), title, 22, TEXT, bold=True)
        draw_text(d, (x, 808), sub, 19, MUTED)
    return img


def slide_3():
    img = blank_slide()
    d = ImageDraw.Draw(img)
    header(d, "进一步参考：motion primitives、future queries 与边界建模", "这些论文提供任务形式和建模工具，具体结构仍需在手机 IMU 双标签任务上验证")
    panel(d, 86, 224, 452, 460, "MoPFormer：运动原语 token", NAVY)
    bullet_lines(
        d,
        126,
        302,
        [
            "单通道片段先做 normalization",
            "VQ codebook 给出离散局部波形原型",
            "motion primitive 表示短时运动形状",
            "适合把 IMU 片段转成 token",
        ],
        382,
        size=22,
        gap=58,
    )
    flow(d, 132, 640, ["window", "VQ", "token"], CYAN)
    panel(d, 574, 224, 452, 460, "FUTR / ActFusion：未来序列", TEAL)
    bullet_lines(
        d,
        614,
        302,
        [
            "观察到的序列进入 encoder",
            "future queries 并行预测多个未来位置",
            "比逐步自回归更少误差累积",
            "适合固定长度未来标签序列",
        ],
        382,
        size=22,
        gap=58,
        bullet_color=TEAL,
    )
    flow(d, 622, 640, ["history", "queries", "future"], TEAL)
    panel(d, 1062, 224, 452, 460, "Boundary-aware：切换时机", BLUE)
    bullet_lines(
        d,
        1102,
        302,
        [
            "先预测 patch-level label",
            "再平滑并合并相邻同类 patch",
            "切换来自首次 label 变化",
            "避免一开始设计复杂边界回归头",
        ],
        382,
        size=22,
        gap=58,
        bullet_color=BLUE,
    )
    flow(d, 1110, 640, ["labels", "smooth", "switch"], CYAN)
    d.rounded_rectangle((86, 742, 1514, 826), radius=12, fill=(255, 255, 255, 240), outline=GRAY, width=2)
    draw_text(d, (116, 793), "阅读定位", 25, NAVY, bold=True)
    draw_text(d, (250, 793), "不照搬单篇论文结构，提取三个可验证组件：token 化、并行未来预测、平滑后的边界读取。", 24, TEXT)
    return img


def main():
    SLIDES_DIR.mkdir(parents=True, exist_ok=True)
    for idx, image in enumerate([slide_1(), slide_2(), slide_3()], start=1):
        out = SLIDES_DIR / f"slide_{idx:02d}.png"
        image.convert("RGB").save(out, "PNG")
        print(out)


if __name__ == "__main__":
    main()
