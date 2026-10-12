# HAR 现代方法定位笔记

日期：2026-06-22

## 背景

第一篇综述 `Deep Learning in Human Activity Recognition with Wearable Sensors` 对深度学习方法的分类比较传统，适合建立历史脉络，但不足以代表 2024-2026 年的研究前沿。

更近的文献显示，wearable / sensor-based HAR 的重点正在从单一监督分类模型，转向以下问题：

- 少标签或无标签数据利用
- 跨人、跨设备、跨数据集泛化
- 自监督预训练和迁移学习
- masked reconstruction / masked autoencoding
- contrastive learning
- sensor foundation model
- 多模态传感器与语言、视觉语义对齐
- 端侧低延迟和轻量化部署

参考线索：

- Past, Present, and Future of Sensor-Based Human Activity Recognition Using Wearables: <https://arxiv.org/abs/2411.14452>
- Comparing Self-Supervised Learning Techniques for Wearable Human Activity Recognition: <https://arxiv.org/abs/2404.15331>
- Self-supervised learning for HAR using 700,000 person-days of wearable data: <https://www.nature.com/articles/s41746-024-01062-3>
- Transformer-Based Approaches for Sensor-Based HAR: Opportunities and Challenges: <https://arxiv.org/abs/2410.13605>
- Foundation Models Defining a New Era in Sensor-based HAR: <https://arxiv.org/html/2604.02711v2>

## DBN 和 DRL 是否需要重点学习

### DBN

DBN 是 Deep Belief Network，通常由多层受限玻尔兹曼机或类似生成式结构堆叠而成。它在深度学习早期常用于无监督预训练和表示学习。

对当前 HAR 学习来说，DBN 只需要知道历史地位即可：

- 它代表早期深度表示学习路线。
- 它曾经用于缓解深层网络训练困难和标签不足。
- 现在不再是 sensor-based HAR 的主流优先选择。
- 当前更值得学的是 autoencoder、masked autoencoder、contrastive learning、Transformer/TCN、foundation model。

### DRL

DRL 是 Deep Reinforcement Learning，核心是智能体在状态中选择动作，通过奖励学习策略。

它不适合作为当前 UCI HAR 分类入门模型，因为 HAR 的基本任务是：

```text
IMU 窗口 -> 活动类别
```

这不是天然的强化学习问题。

但 DRL 在远期可能和以下问题有关：

- 动态选择窗口长度
- 自适应采样率
- 端侧计算资源调度
- 主动学习或交互式标注策略
- OIS 控制策略优化

因此，DRL 目前不作为主线，只保留概念认识。

## 当前更合理的主线

如果目标是研究生阶段做出真实有意义的工作，主线不应停留在简单的 1D CNN 分类准确率。更合理的路线是：

```text
UCI HAR 上建立 IMU 输入/输出直觉
-> 做一个极薄的监督 sanity check
-> 转向自监督/重构式表示学习
-> 再扩展到 Transformer/Attention 或多任务预测
```

这里的 sanity check 不应理解成研究目标，而是用于确认：

- 数据读取是否正确
- 标签是否对齐
- 模型训练流程是否可用
- 后续复杂模型是否有可靠对照

## AutoEncoder 路线

AutoEncoder 更适合作为从 UCI HAR 过渡到现代 HAR 的第一条主线，尤其是 masked autoencoder。

推荐问题形式：

```text
输入：128 x 9 IMU 窗口
处理：随机 mask 一部分时间片或通道
目标：重构被遮蔽的 IMU 片段
用途：学习可迁移的运动表示，再接分类头或预测头
```

优点：

- 不完全依赖标签。
- 和当前前沿的 masked modeling / foundation model 方向一致。
- 可自然扩展到异常检测、冲击预警、短时预测。
- 和课题二的未来轨迹预测有较强关联。

风险：

- UCI HAR 数据量较小，预训练收益可能有限。
- 重构好不等于分类好。
- 需要设计合理 mask 策略和评价方式。

## Attention / Transformer 路线

Transformer 适合处理长依赖、多通道融合、关键时间片选择和多任务建模，但在 HAR 中不能盲目迷信。

近年有研究指出，在 wearable HAR 场景中，Transformer 往往需要更多数据和计算；在小数据、端侧量化和低功耗条件下，未必稳定优于 CNN/TCN/RNN。

因此更稳妥的做法不是直接上大 Transformer，而是：

```text
轻量 Transformer encoder
或 TCN/Conv backbone + attention pooling
或 masked autoencoder + Transformer blocks
```

## 和两个课题的关系

课题一：人体行为与手机运镜联合识别

- 需要识别组合行为和状态切换。
- Attention 可帮助定位关键时间片。
- 自监督预训练可缓解组合标签不足。
- 多任务头可同时预测人体动作、手机运动和状态转移。

课题二：OIS 姿态预测与冲击预警

- masked reconstruction 和未来片段预测非常贴近短时预测。
- AutoEncoder/Transformer encoder 可作为时序表示骨干。
- 可扩展为多任务学习：未来 gyro/acc 回归 + 冲击分类 + 到达时间回归。
- 端侧部署约束要求模型必须轻量化，因此需要关注 tiny Transformer、TCN、蒸馏、量化。

## 当前建议

后续落地模型主线建议定为：

```text
Masked AutoEncoder / self-supervised IMU representation learning
```

同时保留一个极薄的监督分类 baseline，但不把它作为研究目标。

第一版可以做成：

```text
输入：UCI HAR 128 x 9
预训练任务：mask 时间片/通道后重构
下游任务：6 类 HAR 分类
评价：与从头训练的同结构 encoder 分类模型比较
```

如果这条线跑通，再向两个方向扩展：

1. 加入 attention / Transformer blocks。
2. 把下游任务从分类改为未来 IMU 预测和事件预警。
