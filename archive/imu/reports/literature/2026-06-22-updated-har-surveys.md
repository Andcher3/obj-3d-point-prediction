# 更新近年的 HAR / IMU 时序综述候选

日期：2026-06-22

## 为什么需要换一组综述

此前阅读的 `Deep Learning in Human Activity Recognition with Wearable Sensors` 适合作为历史入口，但它主要反映 2022 年前后的深度学习分类框架。

目前 sensor-based / wearable HAR 的研究重心已经明显转向：

- 自监督学习和少标签学习
- masked reconstruction / masked autoencoding
- 跨人、跨设备、跨位置、跨数据集泛化
- foundation model
- LLM / sensor-language / multimodal alignment
- 端侧低功耗、低延迟、可量化部署

因此，后续建议不要再只围绕 `CNN / RNN / DBN / GAN / DRL` 这类旧式模型分类来理解领域，而是按“现实问题”来读：

```text
泛化能力 -> 少标签/无标签 -> 自监督预训练 -> foundation model -> 端侧部署
```

## 最推荐优先阅读的综述

### 1. Past, Present, and Future of Sensor-Based HAR Using Wearables

链接：<https://arxiv.org/abs/2411.14452>

定位：

- 2024 投稿、2025 修订。
- 适合作为新的主综述。
- 从过去、现在、未来三个视角重新梳理 wearable sensor-based HAR。
- 特别强调：HAR 在很多 benchmark 上进展开始停滞，社区正在转向 foundation model 和更真实应用场景。
- 还带 tutorial 色彩，适合边读边形成实践路线。

建议用途：

```text
替代 2022 综述，作为当前领域主地图。
```

### 2. Towards Generalizable Human Activity Recognition: A Survey

链接：<https://arxiv.org/html/2508.12213v1>

定位：

- 2025 综述，直接围绕 IMU-based generalizable HAR。
- 覆盖 229 篇论文和 25 个公开数据集。
- 强调跨人、跨设备、跨位置、跨活动、跨数据集等真实部署问题。
- 方法组织包括 supervised、weakly supervised、unsupervised、self-supervised、LLM-based learning、multi-modal fusion、cross-modal learning、data augmentation。

建议用途：

```text
理解为什么 UCI HAR 准确率不是最终目标，真正困难在泛化。
```

### 3. Self-supervised Learning for Accelerometer-based HAR: A Survey

链接：<https://dl.acm.org/doi/10.1145/3699767>

定位：

- 2024 IMWUT 综述。
- 专门讨论 accelerometer-based HAR 中的 self-supervised learning。
- 检查 52 种 SSL 方法，并按预训练目标进行分类。
- 对当前想走 AutoEncoder / masked reconstruction 路线非常重要。

建议用途：

```text
作为自监督 IMU 表示学习主入口。
```

### 4. Foundation Models Defining A New Era In Sensor-based HAR

链接：<https://arxiv.org/abs/2604.02711>

定位：

- 2026 最新 survey and outlook。
- 重点讨论 sensor-based HAR 中 foundation model 的输入设计、预训练、适配和使用。
- 将前沿方向分为三条：HAR 专用 foundation model、通用时间序列/多模态 foundation model 适配、LLM 用于推理/标注/人机交互。
- 适合作为了解最新趋势的前沿材料。

建议用途：

```text
理解 2026 年该领域正在往哪里走。
```

## 重要的辅助阅读

### 5. Comparing Self-Supervised Learning Techniques for Wearable HAR

链接：<https://arxiv.org/abs/2404.15331>

定位：

- 不是纯综述，而是系统比较文。
- 对比 contrastive、generative、predictive 三类 SSL 方法。
- 结果显示 MAE 路线表现很强，支持后续先做 masked autoencoder。

建议用途：

```text
为“为什么从 MAE 做起”提供实验依据。
```

### 6. Transformer-Based Approaches for Sensor-Based HAR: Opportunities and Challenges

链接：<https://arxiv.org/abs/2410.13605>

定位：

- 不是传统综述，更像针对 Transformer 的系统性评估与提醒。
- 结论偏谨慎：Transformer 在 sensor-based HAR 中往往需要大量数据和计算，量化后可能退化明显。
- 对避免盲目追新很有价值。

建议用途：

```text
帮助判断 Transformer 该怎么用，而不是直接迷信。
```

### 7. A Survey on Multimodal Wearable Sensor-based Human Action Recognition

链接：<https://arxiv.org/html/2404.15349v1>

定位：

- 2024 多模态 wearable HAR 综述。
- 讨论视觉和非视觉模态、inertial sensor、multimodal fusion、cross-modal learning、model deployment。
- 对课题一中“人体行为 + 手机运镜/视觉场景”有启发。

建议用途：

```text
后续从纯 IMU 扩展到 IMU + video / sensor-language 时阅读。
```

### 8. Large Language Models for Wearable Sensor-Based HAR

链接：<https://www.mdpi.com/1424-8220/24/15/5045>

定位：

- 2024 LLM + wearable sensor 综述。
- 更偏趋势观察，而不是马上落地。
- 讨论 LLM 如何辅助 wearable sensor data 的建模、理解、解释和行为建模。

建议用途：

```text
了解 sensor-language / LLM for HAR 的早期形态。
```

## 和课题二相关的时间序列综述

课题二不是纯 HAR，而是短时未来预测、姿态回归和冲击事件预测，因此还应补读时间序列 foundation model / forecasting 的综述。

### 9. Empowering Time Series Analysis with Foundation Models

链接：<https://arxiv.org/abs/2405.02358>

定位：

- 2024 投稿、2025 更新。
- 从 time series、language、vision 三类预训练来源看时间序列 foundation model。
- 对“能否用通用时间序列预训练模型处理 IMU”很重要。

### 10. Deep Learning for Time Series Forecasting: A Survey

链接：<https://arxiv.org/abs/2503.10198>

定位：

- 2025 时间序列预测综述。
- 适合为 OIS 姿态预测、未来 gyro/acc 预测提供模型背景。

## 建议阅读顺序

第一轮只读 4 篇：

```text
1. Past, Present, and Future of Sensor-Based HAR Using Wearables
2. Towards Generalizable Human Activity Recognition
3. Self-supervised Learning for Accelerometer-based HAR
4. Foundation Models Defining A New Era In Sensor-based HAR
```

第二轮再补：

```text
5. Comparing SSL Techniques for Wearable HAR
6. Transformer-Based Approaches for Sensor-Based HAR
9. Empowering Time Series Analysis with Foundation Models
```

当前最值得精读的是第 1 篇和第 3 篇：

- 第 1 篇帮助建立领域全局地图。
- 第 3 篇直接服务于 masked AutoEncoder / self-supervised IMU representation learning。
