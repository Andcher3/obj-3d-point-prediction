# 阅读笔记：Past, Present, and Future of Sensor-Based HAR Using Wearables

日期：2026-06-23

论文：

```text
Past, Present, and Future of Sensor-Based Human Activity Recognition Using Wearables:
A Surveying Tutorial on a Still Challenging Task
```

链接：

- <https://arxiv.org/abs/2411.14452>
- <https://ar5iv.labs.arxiv.org/html/2411.14452>

## 1. 这篇论文的核心判断

作者认为 sensor-based / wearable HAR 已经发展约 30 年，但仍然不是一个“已经解决”的问题。简单活动如计步、睡眠、粗粒度运动识别已经比较成熟，但更真实、更细粒度、更复杂的活动识别仍然困难。

论文最重要的判断是：

```text
HAR 的主线正在从手工特征 -> 监督深度学习 -> 自监督表示学习 -> foundation model / 多模态知识。
```

这比 2022 年那篇综述更适合用来理解当前领域。

## 2. 作者如何定义 HAR

作者把 sensor-based HAR 定义为：

```text
把传感器数据自动分类为感兴趣活动或空类。
```

典型输入来自 body-worn movement sensors，尤其是 IMU。

这和 UCI HAR 的关系很直接：

```text
手机 IMU 窗口 -> 活动类别
```

区别在于，作者强调真实世界 HAR 往往比 UCI HAR 更难，因为真实世界有更多受试者、设备位置、设备类型、噪声、活动组合和标注困难。

## 3. 传统 HAR 的 Activity Recognition Chain

经典流程仍然是理解 HAR 的基础：

```text
data collection
-> pre-processing
-> segmentation
-> feature extraction
-> classification
```

对应到 UCI HAR：

```text
IMU 采集
-> 去噪/归一化/重力分离
-> 2.56s 滑动窗口
-> 561 维手工特征或 128 x 9 深度输入
-> 6 类动作分类
```

作者提醒：传统滑动窗口方法有两个问题：

- 相邻重叠窗口不是独立同分布。
- 固定窗口同时承担特征提取和活动建模，不一定适合所有活动。

## 4. 第一范式变化：从手工特征到监督深度学习

早期 HAR 依赖人工特征，比如：

- 均值
- 方差
- 能量
- 频域熵
- 轴间相关性
- ECDF / 分布特征

这些特征在粗粒度活动上有效，但泛化到复杂场景时不够。

深度学习带来的变化是：

```text
把 feature extraction 和 classification 合并成 end-to-end 表示学习。
```

典型模型包括：

- CNN
- LSTM / GRU
- DeepConvLSTM
- attention
- lightweight attention models, such as TinyHAR
- 小型 Transformer

但作者也强调：传感器数据不是图像。IMU 数据没有图像那种清晰的局部空间层级结构，因此 CNN 革命在 HAR 上没有像视觉领域那样彻底。

## 5. HAR 仍然困难的原因

作者把瓶颈讲得很清楚：

### 标注数据不足

真实 wearable 数据容易收集，但标签很难获得。IMU 波形不像图片那样容易众包标注；即使专家也不一定能直接从波形判断动作。

### 泛化能力弱

模型在新用户、新设备、新佩戴位置、新场景上容易退化。作者认为这是 HAR 长期停滞的重要原因。

### 传感器信号语义模糊

同一段 IMU 信号可能对应不同语义，不同动作也可能产生相似波形。复杂活动常常需要上下文、物体、位置或视觉信息才能解释。

### benchmark 进展停滞

许多常用 benchmark 上已经有不少方法，但真实复杂任务的表现并没有达到“足够可靠”的程度。

## 6. 第二范式变化：自监督表示学习

这是本篇对我们最重要的部分。

作者认为，HAR 社区已经从“研究更深的监督网络”转向“利用大量无标签 wearable 数据学习通用表示”。

标准流程是：

```text
无标签数据预训练 encoder
-> 少量标注数据 fine-tune 或训练分类头
-> 用于 HAR 或其他下游任务
```

作者将其称为：

```text
pretrain-then-finetune
```

这条线适合我们的原因：

- UCI HAR 可以先作为小规模原型。
- 后续真实课题数据可能标注昂贵。
- 课题一和课题二都需要可迁移的 IMU 表示。
- MAE / masked reconstruction 与短时预测、冲击预警天然相近。

## 7. 自监督方法类别

### AutoEncoder

输入 IMU 窗口，encoder 压缩成低维表示，decoder 重构原始窗口。

作用：

- 学习压缩表示
- 去噪
- 建立时序窗口的 latent representation

局限：

- 重构好不等于分类好。
- 如果任务太简单，encoder 可能只学到低级信号统计。

### Masked Reconstruction

随机遮蔽一部分时间点或通道，让模型根据上下文重构缺失部分。

典型形式：

```text
输入：128 x 9 IMU 窗口
mask：随机时间片 / 通道 / patch
目标：只在被 mask 部分计算 MSE
```

作者提到 wearable HAR 中已有 transformer encoder + masked reconstruction 的做法，并且认为它可以看作 denoising autoencoder。

对我们来说，这是最值得落地的第一条路线。

### Multi-task Self-supervised Learning

通过判断某些变换是否被施加来学习表示，例如：

- 加噪声
- scaling
- rotation
- negation
- channel flipping
- segment permutation
- time warping
- channel shuffling

关键直觉是：这些变换对应传感器噪声、方向变化、佩戴位置变化和运动速度变化。

### Contrastive Learning

把相似样本拉近，不相似样本推远。

常见方法：

- SimCLR
- CPC
- BYOL
- SimSiam

作者强调，传感器数据不能直接照搬图像增强；增强方式本身定义了模型要学什么不变性。比如 3D rotation、time warping、Gaussian noise 在 IMU 中有物理意义。

### Multi-device / Multi-modal Contrastive Learning

比人工增强更自然的思路是利用同步多设备或多模态数据：

```text
同一时间、同一人的不同设备信号 = 正样本
不同时间或不同活动信号 = 负样本
```

代表思路包括：

- ColloSSL：多设备同步 IMU
- COCOA：跨模态同步传感器
- wearable movement sensors + pose data

这对课题一很有启发，因为人体行为和手机运镜本质上也是多视角/多语义因素叠加。

## 8. 第三范式变化：foundation model 和世界知识

作者认为 field 可能正在进入第三次范式变化：用 foundation model 的世界知识补上传感器信号的语义不足。

有三条方向：

### 生成虚拟 IMU 数据

用文本描述生成 3D 人体动作，再通过虚拟 IMU 工具生成加速度和角速度。

这能缓解真实 IMU 数据难采集、难标注的问题，但存在 domain gap。

### 用 LLM 提供语义/上下文推理

复杂活动不仅依赖 IMU 波形，还依赖地点、物体、时间、前后动作。LLM 可以提供语义先验或逻辑约束。

但对于纯 IMU，难点在于：

```text
如何把 IMU 窗口翻译成 LLM 能理解的语义概念。
```

### Sensor-language / sensor-vision-language alignment

类似 CLIP，把传感器、图像/视频、文本放到共同 embedding space。

代表思路：

- IMU2CLIP
- Ts2Act
- ImageBind
- SensorLLM
- LLaSA

作者也提醒：当前 plug-and-play sensor-language model 还远未解决，主要原因是传感器异质性和缺少丰富动作文本描述。

## 9. 和时间序列 foundation model 的关系

作者讨论了 TimesFM、Chronos、MOMENT 等时间序列 foundation model。

重点不是说这些模型已经能直接解决 HAR，而是提示：

```text
用适当规模的模型 + 精心选择的数据集，也许能为移动感知任务训练实用的 foundation model。
```

其中 MOMENT 使用 masked reconstruction 作为目标，这再次支持我们从 masked reconstruction / MAE 做起。

## 10. 对我们当前路线的启发

### 不应该把 UCI HAR 准确率当最终目标

UCI HAR 适合作为学习和原型验证，但领域真正难点在：

- 泛化到新用户
- 泛化到新设备和新位置
- 少标签学习
- 复杂活动和组合行为
- 真实端侧部署

### 1D CNN 可以保留为 sanity check，但不是研究主线

作者承认 DeepConvLSTM、CNN、attention 是重要基线，但也说明监督深度模型受标注数据限制明显。

### 当前主线应转向表示学习

最适合我们的下一步是：

```text
Masked AutoEncoder / masked reconstruction for IMU
```

第一版实验：

```text
输入：UCI HAR 128 x 9
预训练：mask 时间片或通道
目标：重构被 mask 的 IMU 值
下游：6 类 HAR 分类
对照：同结构 encoder 从零监督训练
```

### 与课题一的联系

课题一需要同时理解人体动作和手机运镜行为。对应论文中的关键词：

- multi-modal representation
- contrastive learning
- semantic knowledge
- sensor-language / sensor-vision alignment
- complex activity decomposition

### 与课题二的联系

课题二需要短时姿态预测和冲击预警。对应论文中的关键词：

- predictive representation learning
- masked reconstruction
- time-series foundation model
- change point detection
- unlabeled sensor pretraining
- lightweight downstream adaptation

## 11. 建议下一步

先不要急着读完所有 foundation model 和 LLM 方向。下一步应围绕实验可落地性，精读自监督 HAR：

```text
Self-supervised Learning for Accelerometer-based HAR: A Survey
Comparing Self-Supervised Learning Techniques for Wearable HAR
```

并同步设计 UCI HAR 上的第一版 masked reconstruction 实验。
