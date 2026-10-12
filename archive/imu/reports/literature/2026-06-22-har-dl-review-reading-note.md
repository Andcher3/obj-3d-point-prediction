# 阅读笔记：Deep Learning in Human Activity Recognition with Wearable Sensors

论文：

```text
Deep Learning in Human Activity Recognition with Wearable Sensors: A Review on Advances
```

链接：

- <https://pmc.ncbi.nlm.nih.gov/articles/PMC8879042/>
- <https://arxiv.org/abs/2111.00418>

## 为什么先读它

这篇综述最贴近当前阶段：

- 我们正在使用 UCI HAR。
- 数据是手机/可穿戴传感器 IMU。
- 任务是人体行为识别。
- 下一步要进入深度学习模型。

因此它适合作为第一篇主线综述。

## 第一轮阅读目标

第一轮不追求读懂所有细节，只解决 5 个问题：

1. HAR 的标准流程是什么？
2. 传统机器学习和深度学习在 HAR 中的区别是什么？
3. CNN、RNN/LSTM、CNN-RNN 分别为什么适合 IMU？
4. 当前 wearable HAR 的主要挑战是什么？
5. 哪些挑战和我们的两个课题直接相关？

## 预期重点

读正文时优先找这些内容：

- HAR pipeline
- sensor modality
- sliding window
- feature learning
- CNN
- RNN / LSTM / GRU
- hybrid model
- attention
- transfer learning
- self-supervised learning
- real-time / mobile deployment
- generalization

## 和当前 UCI HAR 的连接

当前我们已经完成：

```text
UCI HAR 数据结构理解
IMU 曲线可视化
body_acc / total_acc 差异理解
初步波形观察
```

阅读这篇综述时，应重点把这些概念和综述中的 pipeline 对上：

```text
原始传感器数据 -> 窗口切分 -> 特征/表示学习 -> 分类器 -> 活动标签
```

其中深度学习方法主要替代或弱化了手工特征工程：

```text
传统方法：手工提取均值/方差/频率等特征 -> 分类器
深度方法：输入原始或较原始的时序窗口 -> 模型自动学习特征 -> 分类
```

## 初始阅读提示

读 Abstract 和 Introduction 时，先标出：

- 作者如何定义 HAR。
- 作者为什么强调 wearable/mobile sensors。
- 作者认为 deep learning 带来了什么变化。
- 作者将现有方法分成哪些类别。

读方法分类部分时，建议先画一张自己的表：

```text
模型类别 | 输入形式 | 擅长什么 | 局限是什么 | 和 UCI HAR 的关系
```

第一轮读完后，我们再一起把这张表填完整。

## 第一轮阅读结论

### 1. 作者如何定义 HAR

作者把 HAR 放在移动/可穿戴设备场景中理解：利用手机、手表等设备中的低功耗传感器识别人类活动，从而支持活动追踪、健康监测、人机交互等应用。

本文重点不是视觉 HAR，而是：

```text
wearable/mobile sensor-based HAR
```

也就是基于身体佩戴或手机内置传感器的数据来识别人的物理活动。

作者明确聚焦于 physical activities，包括：

- locomotion
- activities of daily living, ADL
- exercise
- factory work

### 2. 作者为什么强调 wearable/mobile sensors

原因主要有四点：

1. 手机、手表等设备已经非常普及。
2. 它们内置低功耗传感器，适合长期、日常、低侵入式采集。
3. 传感器能覆盖身体运动、生理信号和交互行为。
4. 这些设备能把 HAR 应用落到真实生活中，而不只是实验室。

其中 IMU 是最核心的物理传感器之一。作者说明 IMU 通常包括：

```text
accelerometer + gyroscope + sometimes magnetometer
```

加速度计测三轴线性运动和重力，陀螺仪测 roll/yaw/pitch 方向的旋转率。

### 3. 作者认为 deep learning 带来了什么变化

作者认为 deep learning 对 HAR 的关键变化是：

```text
从手工特征工程转向从原始或低加工传感器信号中自动学习表示。
```

传统 ML 往往需要人工设计时域/频域特征；DL 能直接从原始信号中学习更鲁棒、更任务相关的特征。

作者还指出，DL 发展受益于三点：

- 数据集越来越多
- GPU/FPGA 等硬件加速
- 优化和训练技术进步

### 4. 作者将现有方法分成哪些类别

论文在 taxonomy 中从四个维度组织 HAR：

- sensor
- application
- deep learning approach
- challenge

其中深度学习方法主要分成六类：

```text
Autoencoder, AE
Deep Belief Network, DBN
Convolutional Neural Network, CNN
Recurrent Neural Network, RNN, including LSTM/GRU
Generative Adversarial Network, GAN
Deep Reinforcement Learning, DRL
```

此外，正文还讨论了 CNN-LSTM、multi-branch CNN、attention、federated learning、semi-supervised learning 等变体和趋势。

### 5. HAR 的标准流程

结合本文和当前 UCI HAR，可概括为：

```text
佩戴/手机传感器采集
-> 数据清洗/去噪/同步
-> 滑动窗口切分
-> 特征工程或深度表示学习
-> 分类/识别模型
-> 活动标签或状态输出
-> 部署到手机/可穿戴/边缘设备
```

对 UCI HAR 来说，对应关系是：

```text
IMU 采集 -> 2.56s 窗口 -> 128 x 9 时序 -> 模型 -> 6 类动作
```

### 6. 传统机器学习和深度学习的区别

传统路线：

```text
原始信号 -> 人工设计特征 -> SVM/KNN/RF 等分类器
```

深度学习路线：

```text
原始或低加工时序窗口 -> 神经网络自动学习特征 -> 分类器输出
```

区别不只是“分类器换了”，而是特征来源变了。

传统方法的优势：

- 小数据上稳
- 可解释性较强
- 训练和部署成本低

深度学习的优势：

- 自动特征学习
- 能学习跨通道、跨时间的复杂关系
- 大数据和复杂活动上潜力更高

### 7. CNN、RNN/LSTM、CNN-RNN 为什么适合 IMU

CNN 适合 IMU：

- IMU 是多通道时间序列。
- 局部时间片段中存在步态、冲击、摆动等局部模式。
- 1D 卷积适合提取局部时间依赖。
- 多层 CNN 能把低层局部模式组合成更高层语义。

RNN/LSTM/GRU 适合 IMU：

- IMU 是按时间展开的序列。
- RNN 类模型能建模前后时间依赖。
- LSTM/GRU 通过门控缓解普通 RNN 的长依赖问题。
- 对连续动作、手势、状态切换更自然。

CNN-RNN / DeepConvLSTM 适合 IMU：

- CNN 负责提局部运动特征。
- LSTM/GRU 负责建模这些局部特征的时间演化。
- 对相近活动、复杂活动、多阶段活动更有优势。

### 8. 当前 wearable HAR 的主要挑战

作者在挑战部分按机器学习系统流程归纳为四类：

1. data acquisition
2. label acquisition
3. modeling
4. model deployment

更具体地说：

- 数据采集困难，需要更多高质量、多样化数据。
- 真实场景数据噪声、缺失、采样率变化明显。
- 隐私问题突出。
- 标签获取困难，IMU 曲线本身很难人工直接标注，常需要视频/音频辅助。
- 实地数据集比实验室数据更难获取和同步。
- 固定滑动窗口可能不适合所有活动。
- 复杂活动识别仍然困难。
- 跨人、跨设备、跨场景泛化仍然困难。
- 多传感器鲁棒性和缺失传感器适应仍有问题。
- 移动端部署需要考虑延迟、功耗、模型大小和计算划分。

### 9. 和我们两个课题直接相关的挑战

课题一：人体行为与运镜行为联合建模

直接相关挑战：

- 数据采集与标注：人体行为和运镜行为组合标注更复杂。
- 复杂活动识别：人体动作 + 手机运动是组合行为。
- 动态分割：行为切换不是固定窗口天然能解决的。
- 泛化：不同人、不同拍摄习惯、不同手机握持姿态会导致 IMU 分布变化。
- 部署：移动端实时推理和轻量化。

课题二：OIS 姿态预测与冲击预警

直接相关挑战：

- 数据质量：IMU 噪声、缺失、采样不同步会影响短时预测。
- 标注：冲击事件、到达时间、强度标签不易获得。
- 建模：不只是分类，还要做未来序列回归和事件预测。
- 部署：低延迟、低功耗、流式推理是核心约束。
- 鲁棒性：真实拍摄场景中的手抖、启停、冲击分布复杂。

## 模型类别表

| 模型类别 | 输入形式 | 擅长什么 | 局限是什么 | 和 UCI HAR 的关系 |
|---|---|---|---|---|
| 传统 ML | 手工统计/频域特征，如 `X_train` 的 561 维特征 | 小数据、快速 baseline、可解释性较强 | 依赖特征工程，难自动发现复杂时序模式 | UCI HAR 已提供 `X_train.txt`，可直接跑 SVM/RF |
| Autoencoder | 原始或低加工传感器序列/特征 | 降维、去噪、表示学习、隐私保护 | 本身通常不是最终分类器，需要配合分类头 | 可用于学习 128 x 9 的压缩表示或去噪 |
| DBN | 加速度、频谱或特征向量 | 早期深度表示学习 | 现在不是主流首选，训练和实践生态较旧 | 可了解历史，不建议第一优先实现 |
| 1D CNN | `128 x 9` 原始 IMU 窗口 | 局部时序模式、步态周期、跨通道局部特征 | 长时间依赖较弱 | 最适合作为第一个深度 baseline |
| RNN/LSTM/GRU | 按时间输入的 `128 x 9` 序列 | 时间依赖、动作演化、状态切换 | 训练较慢，可能有梯度/效率问题 | 适合和 CNN baseline 比较 |
| CNN-RNN / DeepConvLSTM | 原始多通道 IMU 序列 | CNN 提局部特征，LSTM 建模时间演化 | 模型更复杂，端侧部署成本更高 | 很适合作为第二阶段 HAR 经典模型 |
| Multi-branch CNN | 多传感器、多位置 IMU | 不同传感器分支独立学习再融合 | 结构复杂，需要多传感器数据 | UCI HAR 只有手机腰部单位置，暂时不必复杂化 |
| Attention / Transformer | 序列 token 或 patch | 长依赖、关键时间片选择、多任务表示 | 小数据上未必划算，计算成本较高 | 可作为后续前沿拓展 |
| GAN / 半监督 | 少标签/无标签传感器序列 | 数据增强、缓解标签不足 | 训练不稳定，评估复杂 | 当前不是第一步，未来可用于扩充 IMU 数据 |
| DRL | 状态、动作、奖励 | 动态分割、策略优化、交互决策 | 不适合普通分类入门，问题定义复杂 | 可作为行为切换/窗口选择的远期方向 |
