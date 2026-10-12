# SensorLM 论文介绍幻灯片 v2 提示词记录

日期：2026-06-24

输出目录：

```text
reports/weekly/native_sensorlm_slides_v2/
```

## Slide 1

```text
Use case: productivity-visual
Asset type: 16:9 academic paper-reading presentation slide image
Primary request: Generate revised slide 1 for a detailed Chinese academic PPT introducing the paper SensorLM: Learning the Language of Wearable Sensors. This slide explains the motivation and target of the paper, not any personal research direction.
Canvas: 16:9 landscape, white background, information-dense but readable, similar density to a university paper-reading PPT template, Microsoft YaHei style typography, strong black title, dark blue accents, thin blue divider near top, no page number.
Exact title text: "为什么需要 SensorLM"
Content layout: three vertical content blocks across the middle, each with a dark-blue heading and 3 compact bullets.
Block 1 heading: "已有基础：大规模 wearable 数据". Bullets: "Fitbit / Pixel Watch"; "59.7M+ hours, 103K+ users"; "2,489,570 person-days, 127 countries".
Block 2 heading: "核心瓶颈：传感器不是语言". Bullets: "连续数值信号，缺少自然语义接口"; "一天级序列很长，通用 LLM 难直接理解"; "高质量 sensor-text pair 稀缺".
Block 3 heading: "论文目标：sensor-language foundation model". Bullets: "zero-shot activity recognition"; "few-shot adaptation"; "sensor-text retrieval"; "sensor caption generation".
Bottom full-width flow band: "sensor time series → language-understandable representation → recognition / retrieval / captioning".
Visual details: use compact cards, thin borders, small simple arrows, dark blue and black text with occasional muted gray; no decorative blobs; no cartoon icons; no watermark; no footer page number; no labels such as conclusion or method flow.
```

## Slide 2

```text
Use case: productivity-visual
Asset type: 16:9 academic paper-reading presentation slide image
Primary request: Generate revised slide 2 for a detailed Chinese academic PPT explaining SensorLM data and input representation.
Canvas: 16:9 landscape, white background, dense academic PPT style, Microsoft YaHei-like typography, black title, dark blue accents, thin blue divider at top, no page number.
Exact title text: "训练样本：一天级 sensor 矩阵 + 自动文本"
Layout: left half is a data source and feature table; right half is a large matrix diagram.
Left top card heading: "数据来源". Include bullets: "Fitbit 与 Pixel Watch"; "wrist-worn wearable sensors"; "59.7M+ hours / 103K+ users".
Left middle table heading: "传感器与分钟级特征". Table columns: "modality" and "features". Rows: "PPG: heart rate, HRV"; "accelerometer: accel, jerk, steps"; "EDA: electrodermal activity"; "temperature: mean, slope"; "altimeter: pressure / elevation".
Right large visual: a 26 by 1440 matrix labeled vertically "26 features" and horizontally "1440 minutes = 1 day". Use a heatmap-like grid but keep it clean. Add annotation: "one person-day = one training sample".
Bottom data-pair band: "X_i = 26×1440 sensor matrix" plus arrow to "T_i = generated caption text" plus note "paired sample: (X_i, T_i)".
Small side note: "目标不是直接把长表格喂给 LLM，而是先训练 sensor encoder。"
Style constraints: high information density like example.pdf, clean tables and diagrams, no decorative blobs, no cartoon scenes, no page number, no watermark, no irrelevant research-direction content.
```

## Slide 3

```text
Use case: productivity-visual
Asset type: 16:9 academic paper-reading presentation slide image
Primary request: Generate revised slide 3 for a detailed Chinese academic PPT explaining how SensorLM constructs captions from wearable sensor data.
Canvas: 16:9 landscape, white background, high information density, academic paper-reading PPT style, Microsoft YaHei-like typography, dark blue headings, thin blue top divider, no page number.
Exact title text: "Hierarchical Sensor Caption：把数值信号写成文本"
Layout: three stacked horizontal rows or columns, each row has: source signal pattern -> generated caption example -> role in training.
Section 1 heading: "Statistical caption". Source text: "mean / max / min / std". Example text: "heart rate mean is ...; step count max is ...". Role text: "覆盖全局数值摘要".
Section 2 heading: "Structural caption". Source text: "trend / spike / drop / stable". Example text: "heart rate increases between minute 640 and 840". Role text: "描述时间结构与局部变化".
Section 3 heading: "Semantic caption". Source text: "recognized activity / sleep / mood logs". Example text: "Outdoor Bike was recorded between minute 1121 and 1133". Role text: "提供活动语义，是 zero-shot activity 的关键".
Right-side compact mini diagram: "26×1440 sensor matrix" -> "rule/template mining" -> "caption text T_i" -> "paired sample (X_i, T_i)".
Bottom note band: "主实验默认更重视 structural + semantic：既保留信号变化，也引入活动语义。"
Style constraints: dense but readable, use small line charts or heatmap snippets only as supporting visuals, no decorative blobs, no cartoon characters, no page number, no watermark, no irrelevant personal research direction content.
```

## Slide 4

```text
Use case: productivity-visual
Asset type: 16:9 academic paper-reading presentation slide image
Primary request: Generate revised slide 4 for a detailed Chinese academic PPT explaining SensorLM architecture, data flow, and training objectives.
Canvas: 16:9 landscape, white background, dense academic PPT style, Microsoft YaHei-like typography, dark blue headings, thin top divider, no page number.
Exact title text: "一个统一架构：对齐表示 + 条件文本生成"
Main architecture diagram: left side shows two inputs: "X_i: 26×1440 sensor matrix" and "T_i: caption tokens". X_i goes to "Sensor Encoder (ViT-2D patches: 2 features × 10 min)". T_i goes to "Text Encoder". From both encoders draw two flows.
Flow A heading: "contrastive flow". Show: "pooling + projection" from both encoders -> "normalized embeddings s_i, v_i" -> "L_con: matched pairs closer, unmatched pairs farther".
Flow B heading: "captioning flow". Show: "sensor tokens + text representations" -> "Multimodal Text Decoder (causal, cross-attention)" -> "predict y_t from y_<t and x" -> "L_cap: next-token cross entropy".
Right side compact variant table with columns "variant", "modules/loss", "what it supports". Rows: "CLIP-like | L_con only | zero-shot / retrieval"; "Cap-like | L_cap only | caption generation"; "CoCa-like | L_con + L_cap | alignment + generation".
Bottom formula band: "L_SensorLM = λ_con L_con + λ_cap L_cap".
Style constraints: emulate paper Figure 3 but do not copy it exactly; use academic block diagrams, arrows, compact table, enough detail, no decorative blobs, no cartoon icons, no page number, no watermark, no personal research direction content.
```

## Slide 5

```text
Use case: productivity-visual
Asset type: 16:9 academic paper-reading presentation slide image
Primary request: Generate revised slide 5 for a detailed Chinese academic PPT summarizing SensorLM experimental results and ablations.
Canvas: 16:9 landscape, white background, dense academic PPT style, Microsoft YaHei-like typography, dark blue title and accents, thin top divider, no page number.
Exact title text: "实验结果：语义对齐带来 zero-shot 与 few-shot 优势"
Layout: four panels in a 2x2 grid plus a bottom takeaway band.
Panel 1 title: "Zero-shot activity recognition". Include small bar chart: Gemma 0.50, Gemini 0.51, SensorLM 0.84, metric label AUROC. Add note: "通用 LLM 直接处理 sensor table 接近随机".
Panel 2 title: "Few-shot activity recognition". Include compact table for AUROC by labels/class: "5: SimCLR 0.54 vs SensorLM 0.73"; "10: 0.55 vs 0.79"; "20: 0.58 vs 0.83"; "50: 0.60 vs 0.88". Note: "label efficiency 更高".
Panel 3 title: "caption 类型消融". Include compact comparison: "statistical 0.51"; "structural 0.50"; "semantic 0.71"; "structural + semantic 0.84". Emphasize: "semantic caption 是活动识别关键".
Panel 4 title: "架构消融". Include table: "CLIP-like: zero-shot AUROC 0.83, linear F1 0.53"; "Cap-like: 0.55, 0.32"; "CoCa-like: 0.84, 0.57". Note: "对比对齐支撑识别，生成目标改善表示并支持 captioning".
Bottom band text: "核心证据：SensorLM 的优势来自大规模弱语义 caption + sensor-text contrastive alignment，而不是单纯更大的通用 LLM。"
Style constraints: dense but readable, exact numeric values should be visible, use dark blue for SensorLM bars and gray for baselines, no decorative blobs, no cartoon icons, no page number, no watermark, no personal research direction content.
```
