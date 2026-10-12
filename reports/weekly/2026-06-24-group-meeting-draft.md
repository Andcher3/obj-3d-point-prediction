# 2026-06-24 组会汇报草稿

## 本周主要进展

本周围绕方向一“手机 IMU 场景下的人体行为与手机/相机运动理解”，从传统 HAR 入门转向近年相关论文调研。

当前形成的判断是：

```text
方向一不应只被理解为 IMU 活动分类，
而应被理解为 IMU、手机/相机运动、场景语义之间的组合行为建模问题。
```

## 已确认的领域趋势

近期 sensor-based / wearable HAR 的趋势正在从：

```text
IMU window -> activity label
```

转向：

```text
sensor time series -> semantic representation / language alignment / multimodal action understanding
```

具体表现为：

- 使用自监督学习缓解标注不足。
- 使用 masked reconstruction / MAE 学习 IMU 表示。
- 将 IMU 与语言、视频、场景语义对齐。
- 利用 LLM 或多模态大模型补足传感器信号缺乏语义上下文的问题。
- 面向真实应用时，还需要考虑跨人、跨设备、传感器缺失、视频质量退化和端侧部署。

## 已重点阅读的论文

### 1. SensorLM: Learning the Language of Wearable Sensors

论文定位：

```text
sensor-language foundation model
```

核心方法：

```text
可穿戴传感器数据
-> 自动生成 statistical / structural / semantic 三层 caption
-> 用 contrastive + generative 目标对齐 sensor 和 language
-> 支持 zero-shot / few-shot / cross-modal retrieval / sensor captioning
```

对方向一的启发：

```text
手机 IMU 不应只被看作分类输入，
而可以被转成可语言描述、可检索、可迁移的行为语义表示。
```

局限：

- 数据来自 Fitbit / Pixel Watch 的长期 wearable 数据，不是手机短窗口 IMU。
- 数据规模极大，当前阶段不能直接复现。
- 更偏长期行为和健康场景，不是手机运镜实时理解。

### 2. 后续计划：SensorLLM

```text
继续看 SensorLLM，因为它更直接讨论 motion sensor 与 LLM 的对齐，
可能比 SensorLM 更贴近短窗口 IMU HAR。
```

### 3. 后续计划：Video-IMU MAE

```text
再看 Masked Video and Body-worn IMU Autoencoder，
用于连接手机/相机视角、IMU 和自监督表示学习。
```

## 当前形成的技术路线

短期不把 1D CNN 分类准确率作为主线，而是将其作为 sanity check。

更适合当前研究准备阶段的路线是：

```text
UCI HAR 上做 IMU masked reconstruction 原型
-> 接 HAR 分类头验证表示有效性
-> 扩展到 video/phone motion 或 sensor-language alignment
```

如果后续进入真实方向一数据，可进一步发展为：

```text
IMU encoder + phone/camera motion encoder + video/text semantic prior
-> 人体动作 + 手机运镜 + 场景语义的组合状态建模
```

## 5 分钟汇报取舍

5 分钟内不适合展开 5 篇论文。当前建议只讲 2 篇，最多补充 1 篇：

```text
主讲：SensorLM
主讲：SensorLLM
补充：Masked Video and Body-worn IMU Autoencoder
```

其中 `SensorLM` 负责说明“为什么要把传感器转成语言/语义空间”；`SensorLLM` 负责说明“LLM 如何和短窗口 motion sensor 对齐”；`Video-IMU MAE` 作为后续把 IMU 和手机/相机画面结合的方向。

## 明天前计划

1. 继续精读 `SensorLLM`，整理 LLM 如何接收和理解 motion sensor time series。
2. 视时间补充 `Masked Video and Body-worn IMU Autoencoder`，连接 video-IMU 自监督路线。
3. 将汇报控制在 5 分钟内，重点讲清 `SensorLM` 一篇论文带来的方向判断。

## 需要老师进一步确认的问题

- 方向一是否更偏向纯 IMU 行为识别，还是允许引入视频/语言语义辅助？
- 手机“运镜行为”的标签定义是否已有初步标准，例如平移、旋转、启停、变焦、抖动等？
- 后续是否有真实手机 IMU + 视频同步数据，或需要先用公开 egocentric video-IMU 数据做方法验证？
