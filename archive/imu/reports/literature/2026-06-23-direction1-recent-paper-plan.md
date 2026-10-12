# 方向一近年论文阅读计划：IMU HAR 到传感器语义理解

日期：2026-06-23

## 组会汇报主线

本周组会建议不把重点放在“已经了解 UCI HAR 数据集”或“读了一篇综述”上，而是表述为：

```text
围绕方向一，调研了 sensor-based HAR 的最新趋势：
从传统 IMU 活动分类，转向自监督表示学习、传感器-语言对齐、
传感器-视频融合和可解释的活动语义理解。
```

这个主线和方向一的关系是：

```text
人体动作识别
-> 手机/相机运动理解
-> 复杂活动语义
-> 多模态/LLM 辅助的上下文推理
```

## 推荐优先阅读的论文

### 1. SensorLM: Learning the Language of Wearable Sensors

链接：<https://arxiv.org/abs/2506.09108>

年份：2025

为什么重要：

- 最新 sensor-language foundation model 方向代表。
- 使用超过 59.7 million hours、103,000+ 人的 wearable 数据。
- 构造 hierarchical caption：statistical、structural、semantic。
- 将传感器数据和自然语言对齐，支持 zero-shot、few-shot、retrieval、captioning。

和方向一的关系：

```text
把 IMU/可穿戴信号从“分类标签”提升到“可语言描述的行为语义”。
```

组会可讲：

- 传感器语义缺失是 HAR 走向真实复杂应用的关键瓶颈。
- 未来方向一可以考虑用文本/视觉语义补足 IMU 对活动语境的理解。

### 2. SensorLLM: Aligning Large Language Models with Motion Sensors for HAR

链接：<https://arxiv.org/abs/2410.10624>

年份：2024，EMNLP 2025

为什么重要：

- 明确把 LLM 引入 motion sensor HAR。
- 两阶段框架：Sensor-Language Alignment + Task-Aware Tuning。
- 使用 trend descriptions 和 channel special tokens，让 LLM 理解不同通道、不同长度的 sensor time series。
- 目标仍以 HAR 分类为主，但强调跨数据集泛化和 sensor-language alignment。

和方向一的关系：

```text
为“手机 IMU -> 活动语义/动作解释”提供可参考框架。
```

组会可讲：

- 不是简单把 IMU 数字丢给 LLM，而是先做 sensor-language alignment。
- 对人体动作和手机运镜联合建模有启发：可以把不同运动通道转成更高层语义。

### 3. LLaSA: A Sensor-Aware LLM for Natural Language Reasoning of Human Activity from IMU Data

链接：<https://arxiv.org/html/2406.14498v3>

年份：2024/2025

为什么重要：

- 从“分类”进一步走向 sensor-aware QA 和解释。
- 不只回答“是什么动作”，还尝试回答“为什么这样判断”“动作意味着什么”。
- 构造 SensorCaps、OpenSQA、Tune-OpenSQA 等 sensor-language 数据。
- 用 sensor encoder + projection + LLM 生成自然语言解释。

和方向一的关系：

```text
复杂活动、手机运镜和场景语义之间的组合关系，需要解释和推理，而不只是分类。
```

组会可讲：

- 方向一若只输出类别，价值有限；若能输出可解释的组合行为状态，更接近真实应用。
- LLM 适合作为语义层，不适合作为端侧实时核心。

### 4. Masked Video and Body-worn IMU Autoencoder for Egocentric Action Recognition

链接：<https://arxiv.org/html/2407.06628v1>

年份：ECCV 2024 / 2025 proceedings

为什么重要：

- 直接结合 egocentric video 和 body-worn IMU。
- 使用 multimodal masked autoencoder，联合重构 video 和 IMU。
- 解决标签稀缺、多 IMU 关系建模、视频质量退化、IMU 缺失等真实问题。
- 与我们之前讨论的 masked reconstruction 高度一致。

和方向一的关系：

```text
方向一不是纯 IMU，也不是纯视频，而是手机/相机运动与人体动作的联合理解。
```

组会可讲：

- Video 提供场景和物体语义，IMU 提供稳定运动信号。
- MAE 可以在无标签 video-IMU 数据上预训练，缓解标注困难。

### 5. EgoDistill: Egocentric Head Motion Distillation for Efficient Video Understanding

链接：<https://proceedings.neurips.cc/paper_files/paper/2023/hash/6a412f0037b0df295a39a198666ea6a6-Abstract-Conference.html>

年份：NeurIPS 2023

为什么仍值得读：

- 虽然不是 2025/2026，但非常贴近“相机/头部运动 + 视频理解”。
- 使用轻量 IMU/head motion 辅助重构重型 egocentric video feature。
- 目标是提升视频理解效率，报告 200x fewer GFLOPs。

和方向一的关系：

```text
手机/相机运动信号可以作为视频理解的低成本辅助信息。
```

组会可讲：

- 手机 IMU 不一定只用于 HAR 分类，还可以辅助视频语义理解和轻量化。

## 建议阅读顺序

考虑明天下午组会，建议只精读 3 篇，另外 2 篇作为补充提及：

```text
精读：
1. SensorLM
2. SensorLLM
3. Masked Video and Body-worn IMU Autoencoder

补充：
4. LLaSA
5. EgoDistill
```

这样汇报结构最清晰：

```text
传感器-语言 foundation model
-> LLM 如何对齐 IMU
-> video-IMU masked autoencoder 如何落地到动作理解
```

## 明天组会可汇报的三点结论

### 结论一：方向一不应停留在传统 HAR 分类

传统 HAR 是：

```text
IMU window -> activity label
```

而方向一更像：

```text
IMU + 手机/相机运动 + 场景语义 -> 组合行为理解
```

### 结论二：最新趋势是传感器语义化

SensorLM、SensorLLM、LLaSA 都说明一个趋势：

```text
把 sensor time series 映射到 human-understandable semantic space。
```

这能缓解纯 IMU 信号语义不足的问题。

### 结论三：短期可落地实验应从 MAE / self-supervised 表示学习开始

EgoVideoIMU-MAE 说明，masked autoencoder 不只是想法，而是已经用于 video-IMU action recognition。

当前项目可先从 UCI HAR 做小原型：

```text
IMU masked reconstruction
-> HAR 分类头
-> 后续扩展到 video/phone motion 或 sensor-language alignment
```

## 明天前的工作安排

今天下午/晚上：

```text
精读 SensorLM 和 SensorLLM，整理 sensor-language 路线。
```

明天上午：

```text
精读 EgoVideoIMU-MAE，整理 video-IMU 自监督路线。
```

明天中午前：

```text
形成 1 页组会汇报稿：
研究背景 -> 最新趋势 -> 读到的 3 篇论文 -> 下一步计划。
```
