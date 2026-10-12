# SensorLM 原生图像生成提示词记录

日期：2026-06-24

## 能力测试图

```text
Use case: photorealistic-natural
Asset type: quick capability test image
Primary request: A small golden retriever puppy running joyfully across a green grassy field.
Scene/backdrop: open sunny meadow with soft natural daylight, shallow depth of field, a few blurred wildflowers in the foreground.
Subject: one puppy in mid-run, ears bouncing, happy expression, realistic fur texture.
Style/medium: photorealistic outdoor photography.
Composition/framing: landscape composition, puppy centered slightly left, visible motion and clean background.
Lighting/mood: bright, warm, cheerful, natural sunlight.
Constraints: no text, no watermark, no logos.
```

## Slide 1

```text
Use case: productivity-visual
Asset type: 16:9 academic presentation slide image
Primary request: Generate slide 1 for a concise Chinese academic group-meeting presentation about the paper SensorLM: Learning the Language of Wearable Sensors.
Canvas: 16:9 landscape slide, white background, clean academic style similar to a university paper-reading PPT.
Typography: Microsoft YaHei style, bold black title, dark blue section accents, no page number.
Exact title text: "SensorLM：让可穿戴传感器进入语言语义空间"
Subtitle text: "Learning the Language of Wearable Sensors, 2025"
Main visual layout: left side shows a simple flow "sensor time series → activity label" labeled "传统 HAR"; below it a short problem statement box: "传感器信号需要先对齐到人能理解的语义空间". Right side shows a compact data card with four rows: "输入 26×1440", "规模 59.7M+ hours / 103K+ users", "训练 对齐 + 生成", "能力 zero-shot / few-shot / 检索".
Bottom emphasis band text: "方向一视角：手机 IMU 不只做分类输入，也可以学习动作、运镜和场景语义的对齐表示。"
Style constraints: enough information density but not crowded; white background; blue divider line near top; dark blue and black text; no decorative blobs; no cartoon icons; use only a small clean vector arrow and card shapes; no page number; no watermark; do not write labels like "真实场景示例" or "学术风格插图".
```

## Slide 2

```text
Use case: productivity-visual
Asset type: 16:9 academic presentation slide image
Primary request: Generate slide 2 for a concise Chinese academic group-meeting presentation about SensorLM.
Canvas: 16:9 landscape slide, white background, clean university paper-reading PPT style, Microsoft YaHei style typography, no page number.
Exact title text: "从自动 caption 到 sensor-language 预训练"
Layout: left column shows three stacked caption cards with colored borders:
1) "Statistical" and smaller text "mean / max / min / std"
2) "Structural" and smaller text "trend / spike / drop"
3) "Semantic" and smaller text "Walk / Sleep / Outdoor Bike"
Middle-right large clean architecture diagram: "26×1440 sensor matrix" enters "Sensor Encoder"; "caption tokens" enter "Text Encoder"; both connect to "Multimodal Text Decoder". Add two small loss boxes: "Contrastive loss：匹配 pair 更近" and "Captioning loss：逐 token 生成 caption".
Bottom line text: "CLIP-like 负责对齐；Cap-like 负责生成；CoCa-like 同时优化两类目标。"
Style constraints: white background, dark blue accents, thin top blue divider, simple rounded rectangles and arrows, no decorative illustrations, no page number, no watermark. Keep text crisp and large; avoid tiny paragraphs.
```

## Slide 3

```text
Use case: productivity-visual
Asset type: 16:9 academic presentation slide image
Primary request: Generate slide 3 for a concise Chinese academic group-meeting presentation about SensorLM results.
Canvas: 16:9 landscape slide, white background, clean academic paper-reading PPT style, Microsoft YaHei style typography, no page number.
Exact title text: "零样本、少标签和检索上的表现"
Top half: two clean bar charts side by side.
Left chart title: "Zero-shot 20-class Activity AUROC". Bars: Gemma 0.50, Gemini 0.51, SensorLM 0.84. SensorLM bar dark blue, baselines gray.
Right chart title: "Few-shot Activity AUROC (50 labels/class)". Bars: DINO 0.54, MSN 0.60, SimCLR 0.60, SensorLM 0.88. SensorLM bar dark blue, baselines gray.
Bottom left compact box title: "caption 消融" with text: "statistical 0.51 / structural 0.50 / structural+semantic 0.84".
Bottom right compact box title: "CLIP-like vs CoCa-like" with text: "zero-shot AUROC 0.83 vs 0.84；分类主要受益于对比对齐，CoCa 额外支持生成。"
Style constraints: charts should be readable, no overly small numbers, no page number, no watermark, no decorative blobs, white background, dark blue top divider, dense but uncluttered.
```

## Slide 4

```text
Use case: productivity-visual
Asset type: 16:9 academic presentation slide image
Primary request: Generate slide 4 for a concise Chinese academic group-meeting presentation about how SensorLM informs research direction 1.
Canvas: 16:9 landscape slide, white background, clean academic paper-reading PPT style, Microsoft YaHei style typography, no page number.
Exact title text: "对方向一的启发"
Main layout: three horizontal insight rows with left dark-blue short labels and right concise explanations.
Row 1 label: "表示学习". Row 1 text: "手机 IMU / 运镜信号可以先进入可解释的行为语义空间，再接具体任务头。"
Row 2 label: "少标签识别". Row 2 text: "大规模弱语义文本能提升活动识别的 zero-shot 与 few-shot 表现。"
Row 3 label: "后续阅读". Row 3 text: "SensorLLM 更直接面向 motion sensor HAR；Video-IMU MAE 更贴近相机视角与 IMU 融合。"
Bottom emphasis box text: "本周汇报主线：从传统 HAR 分类，转向 sensor-language alignment 对方向一的启发。"
Style constraints: white background, dark blue top divider, no page number, no watermark, no decorative blobs, restrained academic look, enough information density, do not label any area as conclusion or method flow.
```
