# 阅读笔记：SensorLLM

日期：2026-06-24

论文：

```text
SensorLLM: Aligning Large Language Models with Motion Sensors for Human Activity Recognition
Li et al., EMNLP 2025
```

链接：

- https://aclanthology.org/2025.emnlp-main.19/
- https://arxiv.org/abs/2410.10624
- https://github.com/cruiseresearchgroup/SensorLLM

## 1. 一句话理解

SensorLLM 的核心不是“直接把 IMU 数字表格喂给 LLM”，而是：

```text
motion sensor time series
-> 自动生成趋势/摘要 QA 文本
-> 学习 sensor-language alignment
-> 再用对齐后的传感器表示做 HAR 分类
```

它比 SensorLM 更贴近我们当前的 UCI HAR 场景，因为它直接面向短窗口 motion sensor HAR，并且实验数据集包含 UCI-HAR。

## 2. 它想解决什么问题

传统 HAR 通常是：

```text
IMU window -> activity label
```

这类模型可以分类，但通常是 task-specific 的：换数据集、换传感器位置、换动作集合，泛化就会变弱。

LLM 有很强的语义理解和泛化能力，但 motion sensor 对 LLM 来说并不天然友好，主要问题是：

- 连续数值很难被文本 tokenizer 合理编码。
- 长时间序列容易超过 LLM context length。
- 多通道 IMU 存在通道间关系，直接展开成文本会丢结构。
- prompt engineering 很难稳定地让 LLM 理解趋势、周期、波动和活动类别。

所以作者提出：先把传感器序列映射到 LLM 能处理的 embedding 空间，并用“人能理解的趋势文本”作为桥梁，让 LLM 学会传感器信号和语言描述之间的对应关系。

## 3. 和 SensorLM 的区别

SensorLM 更像“大规模 wearable sensor-language foundation model”：

```text
一天级别 wearable features + captions -> sensor-language representation
```

SensorLLM 更像“短窗口 motion sensor + LLM 的 HAR 框架”：

```text
短窗口 accelerometer/gyroscope -> trend QA alignment -> HAR classification
```

所以 SensorLLM 对 UCI HAR、手机 IMU、方向一的人体动作识别更直接。

## 4. 数据集形式

论文使用 5 个公开 HAR 数据集：

- USC-HAD
- UCI-HAR
- PAMAP2
- MHealth
- CAPTURE-24

其中 UCI-HAR 在论文中的设置是：

- 30 名志愿者。
- 手机佩戴在腰部。
- 采样率 50 Hz。
- 加速度计 + 陀螺仪，共 6 个通道。
- 6 类活动：Standing、Sitting、Laying、Walking、Walking downstairs、Walking upstairs。
- Stage 1 alignment 使用可变窗口 `w in [5, 200]`。
- Stage 2 HAR 使用窗口 `w = 128`，stride = 64。
- Stage 2 样本数为 train 7352、test 2947，和 UCI HAR 原始划分一致。

注意：我们本地 UCI HAR 的 `Inertial Signals` 有 9 个文件：

```text
total_acc_x/y/z
body_acc_x/y/z
body_gyro_x/y/z
```

而论文描述中把 UCI-HAR 概括为 6 个 accelerometer/gyroscope 通道。后续如果复现，需要先确认作者代码具体使用的是哪 6 个通道，以及是否丢弃了 `total_acc` 或 `body_acc` 中的一组。

## 5. 整体模型结构

SensorLLM 有三个核心模块：

```text
pretrained TS embedder: Chronos
alignment module: MLP
pretrained LLM: LLaMA3-8B
```

训练时：

- Chronos 负责把传感器片段编码为 time-series embedding。
- MLP 负责把 time-series embedding 投影到 LLM 的 hidden dimension。
- LLaMA3-8B 作为语言模型主体。
- Chronos 和 LLM 主体冻结。
- Stage 1 主要训练 alignment MLP。
- Stage 2 主要训练 alignment/classification 相关的小模块。

作者强调这个设计的好处是计算量相对可控：不是全量微调一个 8B LLM，而是只训练较小比例的参数。

## 6. Stage 1：Sensor-Language Alignment

Stage 1 的目标是让模型学会：

```text
单通道传感器片段 -> 趋势描述 / 摘要回答
```

### 6.1 输入形式

原始传感器数据记为：

```text
X: C x T
```

其中：

- `C` 是通道数。
- `T` 是时间长度。

Stage 1 不把所有通道直接拼在一起，而是先按单通道处理：

```text
X_c -> variable-length segment x_s
```

这样做的直觉是：先让模型理解每个通道自己的趋势，再在 Stage 2 组合多通道信息。

### 6.2 趋势 QA 怎么生成

作者不依赖人工标注趋势文本，而是自动生成 QA pair。大体过程是：

```text
传感器片段
-> 统计趋势变化，如 rising / falling / stable
-> 用模板转成问题和回答
```

两类主要 QA：

```text
Trend Analysis QA:
描述这个时间窗口内信号如何变化。

Trend Summary QA:
用更短的自然语言总结整体趋势。
```

例如输入可能是某个 x-axis accelerometer 片段，输出是类似：

```text
0.00-0.13 seconds: downward;
0.13-0.14 seconds: stable;
0.14-0.34 seconds: upward;
...
Overall, the trend remains downward.
```

这个文本不是活动标签，而是“传感器形状的语言化描述”。

### 6.3 数据流

Stage 1 数据流可以写成：

```text
single-channel sensor segment
-> instance normalization
-> Chronos encoder
-> TS embedding
-> MLP alignment projection
-> aligned sensor embedding
-> add channel special tokens, e.g. <x_acc_start>, <x_acc_end>
-> concatenate with instruction/question text embeddings
-> LLaMA causal LM
-> generate trend answer
```

这里的 special tokens 很关键。它们相当于告诉 LLM：

```text
这段 embedding 是 x 轴加速度
那段 embedding 是 z 轴角速度
```

如果没有这些 token，LLM 看到的只是一串连续 embedding，更难知道不同通道的边界和含义。

### 6.4 损失函数

Stage 1 用的是 causal language modeling loss：

```text
给定 sensor embedding + question
预测 answer 的下一个 token
```

也就是类似普通语言模型的 next-token prediction，只是输入里混入了对齐后的传感器 embedding。

## 7. Stage 2：Task-Aware Tuning

Stage 2 的目标是：

```text
多通道传感器窗口 -> HAR activity label
```

### 7.1 输入形式

传感器数据被切成重叠窗口：

```text
X_S: S x C x L
```

其中：

- `S` 是窗口数量。
- `C` 是通道数。
- `L` 是窗口长度。

对 UCI-HAR，论文中是：

```text
L = 128
stride = 64
```

这和我们本地 UCI HAR 的 128 点窗口是一致的。

### 7.2 数据流

Stage 2 数据流可以写成：

```text
multi-channel IMU window
-> each channel through Chronos + alignment MLP
-> aligned channel embeddings
-> concatenate all channel embeddings
-> concatenate statistical text/prompt, such as mean and variance
-> LLaMA forward
-> take final hidden state
-> classification head
-> softmax activity probabilities
```

换句话说，Stage 2 不是让 LLM 生成一句“这个动作是 walking”，而是把 LLM 当作一个 frozen representation processor，最后接分类头做标准监督分类。

### 7.3 损失函数

Stage 2 用标准分类交叉熵：

```text
predicted class distribution vs. ground-truth activity label
```

这和我们后续做 UCI HAR 六分类 baseline 的监督目标是同一类目标。

## 8. 训练流程

完整训练流程：

```text
Step 1: 为传感器片段自动生成 trend/summarization QA pairs
Step 2: 固定 Chronos 和 LLaMA，训练 MLP，把 sensor embedding 对齐到 LLM 空间
Step 3: 用对齐后的多通道窗口做 HAR，训练分类相关模块
Step 4: 在测试集上评估 F1-macro 和 accuracy
```

两个阶段的差别：

| 阶段 | 输入 | 输出 | 目标 | 是否需要活动标签 |
|---|---|---|---|---|
| Stage 1 Sensor-Language Alignment | 单通道可变长度片段 | 趋势/摘要文本 | 让 LLM 理解 sensor trend | 不需要人工活动标签 |
| Stage 2 Task-Aware Tuning | 多通道固定窗口 | 活动类别 | HAR 分类 | 需要活动标签 |

## 9. 推理流程

如果做趋势解释：

```text
sensor segment
-> Chronos + MLP
-> LLM receives sensor embedding + question
-> generate trend description
```

如果做 HAR 分类：

```text
multi-channel sensor window
-> Chronos + MLP
-> concatenate channel embeddings + statistics prompt
-> LLM forward
-> final hidden state
-> classifier
-> activity label
```

## 10. 主要实验结果

### 10.1 Sensor Data Understanding

作者比较了 SensorLLM 和 GPT-4o 对传感器趋势描述的能力。

结论：SensorLLM 在 BLEU、ROUGE、METEOR、SBERT、SimCSE、GPT-4o 打分、人类专家打分上都优于直接让 GPT-4o 读传感器趋势。

对 UCI-HAR 来说：

- BLEU-1：GPT-4o 37.97，SensorLLM 56.78。
- Human score：GPT-4o 1.94，SensorLLM 4.04。

这说明一个重要点：

```text
通用 LLM 并不会天然理解传感器数值；
必须先做 sensor-language alignment。
```

### 10.2 HAR 分类结果

作者在五个数据集上比较了 SensorLLM 和多种时间序列/HAR baseline：

- Transformer
- Informer
- NS-Transformer
- PatchTST
- TimesNet
- iTransformer
- DeepConvLSTM
- DeepConvLSTMAttn
- Attend
- Chronos+MLP
- GPT4TS

SensorLLM 在 5 个数据集中有 4 个取得最好 F1-macro：

- USC-HAD
- PAMAP2
- MHealth
- CAPTURE-24

在 UCI-HAR 上：

```text
Attend:    F1-macro 93.2, accuracy 92.8
SensorLLM: F1-macro 91.2, accuracy 90.8
```

所以要注意：SensorLLM 在 UCI-HAR 上不是第一，而是第二。它的价值不只是单一数据集精度，而是跨数据集、多通道、可变长度、传感器-语言对齐带来的泛化性。

### 10.3 消融实验

几个关键结论：

1. 去掉 Stage 1 alignment 会明显变差。

UCI-HAR 上：

```text
Task-only + prompts: 82.0 F1
SensorLLM + prompts: 91.2 F1
```

这说明 Chronos embedding 本身不够，必须通过趋势 QA 对齐到 LLM 空间。

2. textual prompts 有帮助。

UCI-HAR 上：

```text
SensorLLM w/o prompts: 89.2 F1
SensorLLM w/ prompts:  91.2 F1
```

这里的 prompt 不是普通聊天 prompt，而是包含每个 sensor channel 的统计信息等辅助文本。

3. special tokens 有帮助。

在 PAMAP2、MHealth 这类多通道数据上，加入 `<channel_start>` / `<channel_end>` 之类 token 能提升表现。通道越多，这类结构提示越重要。

4. alignment 可以跨数据集迁移。

作者让 Stage 1 在 USC-HAD 上学 alignment，再在 UCI-HAR 上做 Stage 2，结果接近同数据集训练：

```text
UCI-HAR stage1 + UCI-HAR stage2: 91.2 F1
USC-HAD stage1 + UCI-HAR stage2: 91.0 F1
```

这很重要：它暗示学到的不是某个数据集私有模式，而是某种较通用的 sensor-language 表示。

## 11. 对方向一的启发

方向一不是单纯做：

```text
IMU -> walking / sitting / standing
```

而更可能需要：

```text
手机 IMU
+ 人体动作
+ 手机/相机运动
+ 场景语义
-> 组合行为理解
```

SensorLLM 给我们的启发是：

1. LLM 不适合直接生吞 IMU 数字。

真正有效的是先通过 encoder + alignment module 把传感器序列变成 LLM 可用的 embedding。

2. 语言可以作为“中间语义层”。

trend description、statistical prompt、activity question 这些文本可以让模型把低层波形和高层活动语义接起来。

3. 对手机运镜行为也可以借鉴这个思路。

例如未来不只让模型输出：

```text
walking
```

而是输出或理解更复杂的状态：

```text
用户边走路边平移手机拍摄
手机存在明显上下抖动
相机运动和人体步态强相关
```

4. 对当前阶段来说，不建议直接复现完整 SensorLLM。

原因是它依赖 Chronos、LLaMA、大显存和复杂数据生成流程。更现实的路线是先把其中的思想压缩到小实验：

```text
UCI HAR 9-channel/6-channel IMU
-> 自动提取趋势/统计文本
-> 小型 sensor encoder
-> HAR 分类或 masked reconstruction
```

## 12. 局限

1. 仍然是封闭集分类。

虽然用了 LLM，但最终 HAR 部分还是 classification head，并没有完全发挥 LLM 的开放词表、生成式推理和 open-set recognition 能力。

2. trend description 还不够完整。

作者自己也指出，目前 alignment 主要关注趋势变化，没有充分覆盖频域、周期性、高阶运动模式。这对 HAR 很关键，因为走路、上下楼这类动作的周期和频率信息非常重要。

3. 计算资源较重。

实验使用 A100-80G。即使冻结 LLM，完整训练/复现对当前本地环境仍然不轻。

4. 和手机视频场景仍有距离。

它处理 motion sensor HAR，但没有处理手机视频、相机运动、镜头语义和场景语义之间的组合关系。

5. UCI-HAR 上不是最强。

它在 UCI-HAR 上略低于 Attend，说明在小型规范数据集上，专门设计的 HAR 模型仍然很有竞争力。

## 13. 组会中可以怎么讲

可以用 50-70 秒讲这篇：

```text
第二篇我看的是 EMNLP 2025 的 SensorLLM。它比 SensorLM 更贴近传统 HAR，因为它直接处理 accelerometer 和 gyroscope 这类短窗口 motion sensor，并在 UCI-HAR 等五个数据集上做实验。它的核心不是直接把 IMU 数字喂给 LLM，而是分两阶段：第一阶段自动生成传感器趋势 QA，比如某个轴在某段时间 rising、falling 或 stable，用这些文本训练一个 sensor-language alignment module；第二阶段把多通道传感器窗口经过这个对齐模块送入冻结的 LLM，再接分类头做 HAR。实验里，去掉 alignment 后 UCI-HAR F1 从 91.2 降到 82 左右，说明通用 LLM 并不会天然理解 IMU，关键是先建立传感器到语言空间的桥。对我的方向一启发是，后续手机 IMU 不应只做六分类，而可以学习可语言描述的行为表示，用来连接人体动作、手机运镜和场景语义。
```

## 14. 下一步

建议第三篇读：

```text
Masked Video and Body-worn IMU Autoencoder for Egocentric Action Recognition
```

原因：

- SensorLM 讲 sensor-language foundation model。
- SensorLLM 讲 motion sensor 和 LLM 对齐。
- EgoVideoIMU-MAE 直接进入 video + IMU masked autoencoder，更贴近方向一的“手机视频 + IMU”组合建模。
