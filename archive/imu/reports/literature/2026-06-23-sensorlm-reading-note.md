# 阅读笔记：SensorLM: Learning the Language of Wearable Sensors

日期：2026-06-23

论文：

```text
SensorLM: Learning the Language of Wearable Sensors
```

链接：

- <https://arxiv.org/abs/2506.09108>
- <https://ar5iv.labs.arxiv.org/html/2506.09108v1>

## 1. 这篇论文解决什么问题

传统 sensor-based HAR 通常做的是：

```text
sensor time series -> activity label
```

SensorLM 想解决的是更高一层的问题：

```text
sensor time series -> language-understandable semantic representation
```

也就是说，它不满足于判断“跑步 / 走路 / 睡眠”，而是希望把传感器数据和自然语言对齐，让模型能够：

- zero-shot 识别活动
- few-shot 适配新任务
- sensor-to-text / text-to-sensor 检索
- 根据传感器数据生成自然语言描述
- 泛化到未见过的活动概念

## 2. 为什么这件事重要

作者指出，真实世界 wearable sensor 数据虽然很多，但缺少高质量的传感器-文本配对数据。

这导致一个核心瓶颈：

```text
传感器数据本身是连续数值信号，缺少可供大模型直接理解的语义接口。
```

直接把一天的 sensor table 输入给 LLM 并不现实，因为：

- 序列太长，token 数爆炸。
- LLM 对连续时间序列的细粒度变化并不擅长。
- 直接表格化输入会丢掉传感器事件和语言描述之间的对齐关系。

这与我们前面讨论的“用 foundation model 的世界知识补 sensor 语义不足”正好对应。

## 3. 数据是什么

SensorLM 使用的是大规模 wrist-worn wearable 数据，来源包括 Fitbit 和 Pixel Watch。

传感器包括：

- PPG
- 3-axis accelerometer
- altimeter
- skin temperature
- electrodermal activity

作者从这些信号中提取 26 个分钟级特征，例如：

- heart rate / HRV
- acceleration-related features
- jerk
- steps
- temperature mean / slope
- EDA
- altimeter pressure

模型输入不是 UCI HAR 那种短窗口原始 IMU，而是：

```text
26 features x 1440 minutes
```

也就是一天级别的多模态 wearable sensor 表示。

数据规模很大：

```text
约 59.7 million hours
103,000+ people
2,489,570 person-days
127 countries
```

## 4. 核心方法：Hierarchical Sensor Caption

SensorLM 的关键不是只做模型，而是先构造 sensor-language 数据。

作者生成三层 caption：

### Statistical caption

描述基本统计量，例如：

```text
mean / max / min / std
```

作用是让语言描述覆盖数值摘要。

### Structural caption

描述时间序列结构，例如：

```text
increase / decrease / stable trend / spike / drop
```

作用是让语言描述覆盖局部趋势和动态变化。

### Semantic caption

描述高层语义，例如：

```text
某段时间出现 Outdoor Bike activity
某段时间用户记录 mood 为 Frustrated
```

作用是让模型把传感器模式和真实活动/状态联系起来。

作者的消融结果说明：

```text
semantic caption 对 zero-shot activity recognition 很关键；
semantic + structural caption 整体效果最好。
```

这点对我们很重要：纯统计描述不够，方向一真正需要的是运动模式和语义标签/场景之间的桥。

## 5. 模型结构

SensorLM 包含三个部分：

```text
sensor encoder
text encoder
multimodal text decoder
```

### Sensor encoder

使用类似 ViT 的方式处理传感器序列：

```text
sensor sequence -> patch -> linear embedding -> transformer blocks
```

目标是提取局部和长程时间依赖。

### Text encoder

把文本 caption 编码成语言表示。

### Multimodal text decoder

使用 causally masked transformer，通过 cross-attention 结合 sensor features 和 text features，生成或建模文本。

## 6. 训练目标

SensorLM 把两类目标统一起来：

### Contrastive objective

类似 CLIP：

```text
匹配的 sensor-text pair 拉近
不匹配的 sensor-text pair 拉远
```

适合 zero-shot classification 和 cross-modal retrieval。

### Generative objective

类似 captioning：

```text
给定 sensor，生成 paired text
```

适合 sensor captioning 和细粒度解释。

作者将不同组合统一为：

```text
CLIP-like
Cap-like
CoCa-like
```

结果显示，结合 contrastive 和 generative 的 CoCa-like 变体效果更好。

## 7. 实验结果重点

### Zero-shot activity recognition

SensorLM 在 20 类 activity recognition 上明显优于直接用 Gemini / Gemma 处理表格化 sensor 输入。

这说明：

```text
通用 LLM 本身不会天然理解传感器；
必须先做 sensor-language alignment。
```

### Cross-modal retrieval

SensorLM 可以做：

```text
sensor -> text
text -> sensor
```

也就是用传感器找描述，或用描述找传感器片段。

这对方向一很有想象空间：未来可以查询类似“人正在边走边移动手机镜头”这样的组合状态。

### Few-shot learning

在少量标签下，SensorLM 学到的 sensor representation 比 SimCLR、DINO、MSN 等 SSL baseline 更有优势。

这说明语言监督能提高 label efficiency。

### Unseen class generalization

作者展示模型可以对未见过的活动产生一定 zero-shot 泛化，例如根据语义相近性理解新活动。

这说明模型学到的不只是固定类别，而是某种活动概念空间。

## 8. 对方向一的启发

方向一如果只做：

```text
IMU -> human activity label
```

会比较像传统 HAR。

SensorLM 给我们的启发是，未来更有价值的方向可能是：

```text
IMU / phone motion -> semantic representation
```

进一步可以连接：

```text
人体动作
手机/相机运动
场景语义
自然语言描述
```

这对“人体行为 + 手机运镜行为联合建模”很有启发。因为运镜行为本身就不仅是物理运动，还带有拍摄意图和场景语义。

## 9. 对当前项目的局限

这篇论文不能直接照搬。

主要原因：

- 数据是 wrist-worn Fitbit / Pixel Watch，不是手机 IMU。
- 输入是一天级 `26 x 1440` 特征，不是 `128 x 9` 短窗口原始 IMU。
- 任务偏长期活动和健康分析，不是手机运镜实时理解。
- 数据规模巨大，当前阶段不可能复现。
- 论文依赖自动生成 caption，而这些 caption 的质量会影响模型学到的语义。

所以它更适合作为“方向判断”，不是当前可直接复现实验。

## 10. 5 分钟组会里怎么讲

可以用 40-60 秒讲这篇：

```text
我重点看了 2025 年的 SensorLM。它不是传统 HAR 分类论文，而是尝试把 wearable sensor 数据和自然语言对齐。作者构造了 statistical、structural、semantic 三层 caption，用大规模 Fitbit/Pixel Watch 数据训练 sensor-language foundation model。实验显示，直接把传感器表格喂给 LLM 效果不好，必须先做 sensor-language alignment；语义 caption 对 zero-shot 活动识别尤其关键。对我的方向一启发是：手机 IMU 不应只作为分类输入，而可以进一步学习能和人体动作、手机运镜、场景语义对齐的表示。
```

## 11. 下一步

下一篇建议读 `SensorLLM`。

原因：

- SensorLM 偏长期 wearable health/activity。
- SensorLLM 更直接面向 motion sensor 和 HAR。
- 它可能更接近 UCI HAR 这种短窗口 IMU 场景。

## 12. 核心方法详细解释

### 12.1 先分清两件事

SensorLM 里有两种“组合”，容易混淆：

第一种是 caption 内容组合：

```text
statistical caption
structural caption
semantic caption
```

论文会比较只用其中一种、两种组合、三种全用的效果。主实验默认使用：

```text
structural + semantic
```

第二种是训练目标/模型形态组合：

```text
CLIP-like
Cap-like
CoCa-like
```

它们不是三种 caption，而是三种使用 loss 和模块的方式。

### 12.2 一个训练样本长什么样

一个样本可以理解为：

```text
X_i: 26 features x 1440 minutes 的一天传感器矩阵
T_i: 由这个 X_i 自动生成的一段或多段文本 caption
```

例如 `T_i` 可能是多句话拼成的文本：

```text
Step count shows a spike at minute 720.
Heart rate exhibits an increasing trend from minute 680 to 960.
Outdoor Bike was recorded between minute 1121 and 1133.
```

这些文本不是人工逐条写的，而是由规则/模板从传感器统计、结构模式、活动/睡眠/情绪日志中生成的。

### 12.3 caption 最后生成的文本代表什么

生成的文本不是“模型随便写的解释”，而是对某一天传感器数据的自然语言描述。

根据训练用的 caption 类型不同，它可以描述：

- 统计信息：某个特征的均值、最大值、最小值、标准差。
- 结构变化：某段时间上升、下降、稳定、出现 spike/drop。
- 高层语义：某段时间发生 Walk、Outdoor Bike、Sleep，或用户记录某种 mood。

在 caption generation 推理时，模型输入 sensor，输出类似下面的文本：

```text
Walk was recorded between minute 785 and 822.
Sleep occurred from minute 1346 to 1440.
```

也就是说，它试图把传感器信号翻译成人能读懂的事件描述。

### 12.4 数据流：不是一条流，而是两条训练流

用户原先的理解是：

```text
sensor -> sensor encoder
caption -> text encoder
两个特征 -> multimodal decoder
```

这只说对了一部分。更准确地说，SensorLM 同时有两条训练流。

#### A. Contrastive flow

这一条像 CLIP。

```text
sensor X_i -> sensor encoder -> average pooling -> projection -> normalized sensor embedding s_i
text T_i   -> text encoder   -> average pooling -> projection -> normalized text embedding v_i
```

然后在一个 batch 里计算所有 sensor embedding 和 text embedding 的相似度。

目标是：

```text
匹配的 (X_i, T_i) 相似度高
不匹配的 (X_i, T_j) 相似度低
```

这条流训练的主要是：

- sensor encoder
- text encoder
- 两侧 projection heads

multimodal text decoder 不参与这条 contrastive loss。

#### B. Generative / captioning flow

这一条像 image captioning / sequence-to-sequence。

```text
sensor X_i -> sensor encoder -> sensor tokens
caption T_i 的文本 tokens -> text encoder -> text-side representations
text-side representations + sensor tokens -> multimodal text decoder
decoder 对 sensor tokens 做 cross-attention，并结合 text features
decoder 自回归预测下一个 text token
```

训练时采用 teacher forcing：

```text
输入：<START>, t1, t2, ..., t_{k-1}
目标：t1, t2, ..., t_k
```

loss 是标准的 next-token cross entropy。

这条流训练的主要是：

- sensor encoder
- text encoder / text-side representations
- multimodal text decoder
- decoder 侧的文本 token embedding / language modeling 部分

Figure 3 左图明确显示 caption 文本先进入 `Text Encoder`，再向下进入 `Multimodal Text Decoder`。因此，不能说 captioning 分支绕过 text encoder。更准确地说：text encoder 在 contrastive 分支中提供 pooled text embedding，在 captioning 分支中提供文本侧 token/representation；multimodal decoder 再结合 sensor encoder 输出进行自回归文本预测。

论文公式写成：

```text
L_cap = - sum_t log P(y_t | y_<t, x)
```

这说明生成第 t 个 token 时，模型条件于 sensor 输入 `x` 和过去文本前缀 `y_<t`，而不是直接看完整答案。

### 12.5 Generative objective 具体怎么生成 paired text

训练时，模型已经知道这一天 sensor 对应的 caption `T_i`。

它不是一次性输出整句，而是逐 token 学：

```text
给 sensor + <START>，预测第 1 个词
给 sensor + <START> + 第 1 个真词，预测第 2 个词
给 sensor + <START> + 前 2 个真词，预测第 3 个词
...
```

这就是 autoregressive generation。

推理时没有真词可喂，所以从 `[START]` 开始：

```text
sensor + [START] -> 生成第 1 个 token
sensor + [START] + token1 -> 生成第 2 个 token
...
直到 [END] 或长度上限
```

论文附录还说明，为了 caption generation，作者训练了一个 decoder 更深的 SensorLM-B 变体：caption generation 用 12 层 decoder，而默认 decoder 是 3 层。

### 12.6 CLIP-like / Cap-like / CoCa-like 是什么

这三者是 vision-language pretraining 里的经典范式，SensorLM 把它们迁移到 sensor-language。

| 名称 | 使用哪些模块 | 使用什么 loss | 擅长什么 | 局限 |
|---|---|---|---|---|
| CLIP-like | sensor encoder + text encoder | contrastive loss | zero-shot 分类、sensor-text 检索 | 不会生成文本 |
| Cap-like | sensor encoder + multimodal text decoder | captioning/generative loss | 根据 sensor 生成自然语言描述 | 跨模态 embedding 对齐较弱 |
| CoCa-like | sensor encoder + text encoder + multimodal text decoder | contrastive + generative | 同时支持检索、zero-shot、captioning | 模型更复杂 |

论文主张 CoCa-like 更强，因为它同时学两件事：

```text
1. sensor 和 text 在同一 embedding space 里对齐
2. sensor 能条件生成对应的 text
```

### 12.7 具体模型结构

#### Sensor encoder

输入：

```text
26 features x 1440 minutes
```

它被当成一个二维矩阵，类似一张很窄的“传感器图像”。

论文使用 ViT-2D：

```text
patch size = (2 features, 10 minutes)
```

所以：

```text
26 / 2 = 13
1440 / 10 = 144
13 x 144 = 1872 sensor tokens
```

这些 tokens 经过 linear embedding 和 transformer blocks，得到 sensor token features。

这个设计有一个需要批判性理解的地方：图像 patch 的局部像素天然相关，但 sensor feature 维度没有同等强的空间局部性。SensorLM 的 26 个特征在论文表格中按传感器模态分组，例如 PPG 特征、accelerometer 特征、EDA、temperature、altimeter，因此相邻 feature 往往同属或接近同一类传感器。但 `(2 features, 10 minutes)` 仍然依赖特征排列顺序。

如果 feature 顺序被随意打乱，初始 patch 内混合的两个 feature 就会改变，模型归纳偏置也会改变。Transformer 后续可以让所有 token 全局交互，但第一步 patch projection 已经把局部 feature 组合固定下来了。

因此，这里的 2D patch 更像是：

```text
为了降低 token 数、引入局部时间-特征模式的工程设计
```

而不是像图像 patch 那样有非常自然的空间语义。若迁移到 UCI HAR 或手机 IMU，通道顺序和 patch 方式需要重新设计，例如优先按 `acc/gyro`、`x/y/z` 或时间 patch 来组织。

#### Text encoder

输入 tokenized caption。

输出文本 token features，再通过 average pooling、projection、normalization 得到一个全局 text embedding，用于 contrastive loss。

#### Multimodal text decoder

这是 causally masked transformer decoder。

它做两件事：

- 使用 text encoder 输出的文本侧表示作为语言上下文。
- 对 sensor encoder 输出的 sensor tokens 做 cross-attention。

因此每一步生成文本时，decoder 都能参考 sensor 内容。

### 12.8 具体架构大小

论文给了四个版本。

| 模型 | sensor encoder 层数 | sensor encoder 参数量 | text encoder 层数 | decoder 层数 | hidden | heads |
|---|---:|---:|---:|---:|---:|---:|
| SensorLM-S | 12 | 3M | 12 | 3 | 128 | 16 |
| SensorLM-B | 12 | 114M | 12 | 3 | 768 | 12 |
| SensorLM-L | 24 | 404M | 24 | 3 | 1024 | 16 |
| SensorLM-XL | 40 | 1.27B | 40 | 3 | 1408 | 16 |

主实验默认报告 SensorLM-B。

### 12.9 完整数据流图

```text
离线 caption 生成阶段
────────────────────────────────────────
X_i: 26 x 1440 sensor matrix
  ├─ compute statistics -> statistical captions
  ├─ sliding-window trend/spike detection -> structural captions
  └─ activity/sleep/mood logs -> semantic captions

得到 paired sample:
  (X_i, T_i)


预训练阶段：contrastive branch
────────────────────────────────────────
X_i -> sensor encoder -> pool/project/norm -> s_i
T_i -> text encoder   -> pool/project/norm -> v_i

batch 内做 sensor-text 对比学习：
  s_i 应接近 v_i
  s_i 应远离 v_j (j != i)


预训练阶段：generative branch
────────────────────────────────────────
X_i -> sensor encoder -> sensor tokens
T_i shifted tokens -> text encoder -> text-side representations
text-side representations + sensor tokens -> multimodal decoder
decoder cross-attends sensor tokens and predicts next caption token


总 loss
────────────────────────────────────────
L = lambda_con * L_contrastive + lambda_cap * L_captioning

主实验中：
  lambda_con = 1
  lambda_cap = 1
```

### 12.10 下游任务怎么使用

#### Zero-shot classification

不需要 decoder。

```text
sensor -> sensor encoder -> sensor embedding
candidate labels -> prompt templates -> text encoder -> label embeddings
选择相似度最高的 label
```

例如把 `Run` 写成多种 prompt：

```text
A period of Run was observed during the session.
Detected a phase of Run.
The main action was Run.
```

每个类别取多个 prompt embedding 平均，再和 sensor embedding 比相似度。

#### Cross-modal retrieval

不需要 decoder。

```text
sensor embedding <-> text embedding
```

可以做：

- sensor 找 caption
- caption 找 sensor

#### Caption generation

需要 decoder。

```text
sensor + [START] -> autoregressively generate caption
```

#### Few-shot / linear probing

通常冻结 sensor encoder，只在其输出表示上训练一个简单线性分类器或回归器。

## 13. 对 Figure 3 的修正理解

根据 Figure 3 左图和正文描述，完整结构应理解为：

```text
sensor -> Sensor Encoder
text   -> Text Encoder

Contrastive Loss:
  Sensor Encoder pooled embedding 与 Text Encoder pooled embedding 对齐

Captioning Loss:
  Text Encoder 的文本侧表示 + Sensor Encoder 的 sensor tokens
  -> Multimodal Text Decoder
  -> 自回归预测 caption token
```

因此，之前将 captioning 流程简化为“text prefix 直接进入 multimodal decoder”是不够准确的。更贴近论文图示的说法是：

```text
文本先经过 Text Encoder 形成文本侧表示，再进入 Multimodal Text Decoder；
decoder 同时通过 cross-attention 读取 Sensor Encoder 输出。
```

但从训练目标角度看，captioning 仍然是自回归的：

```text
给定 sensor 和已知文本前缀，预测下一个文本 token。
```

## 14. 语义 caption 的来源

论文也承认，真实 wearable sensor 数据通常缺少自然存在的 paired text。SensorLM 的做法不是人工为每一天写描述，而是自动生成 caption。

### Statistical captions

来自传感器数值本身：

```text
对每个 feature 计算 mean / max / min / std
再套入多种模板
```

### Structural captions

也来自传感器数值本身：

```text
检测 trend / spike / drop 等时间结构
再套入模板
```

### Semantic captions

来自已有事件信息：

- recognized activities
- sleep periods
- user-logged mood data

论文中的例子包括：

```text
Outdoor Bike was recorded between minute 1121 and 1133.
Sleep took place during the minutes 5 through 449.
The person logged their mood as Frustrated at minute 1110.
```

这些语义 caption 并不是逐分钟人工标注出来的完整行为真值，而是从已有活动记录、睡眠记录和用户 mood 日志等事件数据中模板化生成的。

这不和“标签难”矛盾，因为这里得到的是：

```text
大规模、弱标注、模板化、事件级、可能有噪声的语义文本
```

而不是：

```text
高质量、密集、精确边界、覆盖所有真实复杂活动的人工标签
```

也就是说，SensorLM 通过庞大用户规模和自动模板生成绕开了人工精细标注问题，但并没有彻底解决 HAR 的高质量标签难题。

## 15. HAR 与模糊 caption 的关系

SensorLM 不是用 caption 直接替代标准 HAR 标签。它的逻辑是：

```text
弱语义 caption -> 预训练 sensor encoder -> 下游 HAR 再用标准标签或 prompt 做识别
```

### 15.1 caption 里到底有没有动作信息

semantic caption 中确实包含动作信息，例如：

```text
Outdoor Bike was recorded between minute 1121 and 1133.
Walk was recorded between minute 785 and 822.
Sleep occurred from minute 1303 to 1440.
```

这些动作信息来自已有的 recognized activities、sleep periods、user-logged mood data 等事件记录。

但它和标准 HAR 标签不完全等价。

### 15.2 为什么不能直接当普通 HAR 标签

标准 HAR 标签通常希望是：

```text
每个固定时间窗口 -> 一个准确类别
```

而 SensorLM 的 semantic captions 更像：

```text
一天内某些时间段 -> 一些事件级描述
```

差异包括：

- caption 不是每一分钟、每个窗口都有完整动作标签。
- caption 可能只覆盖用户记录或系统识别到的活动，没覆盖的时间段不代表无活动。
- caption 是事件级和模板化文本，不一定精确到 HAR benchmark 所要求的边界。
- caption 可能有噪声，例如 wearable 自动识别或用户日志本身不一定完全准确。
- caption 常常只描述高层事件，例如 `Outdoor Bike`、`Walk`、`Sleep`，不一定覆盖细粒度动作或组合行为。

所以，caption 包含动作语义，但不是高质量、密集、精确的监督标签。

### 15.3 它为什么还能帮助 HAR

它的帮助方式不是：

```text
caption 生成得很准 -> 直接得到 HAR 标签
```

而是：

```text
大量弱语义配对 -> 让 sensor encoder 学到活动相关表示
```

例如，很多天的数据中，只要 `Outdoor Bike` caption 经常和某种心率、步数、加速度、海拔变化模式共同出现，模型就能逐渐学到：

```text
这类 sensor pattern 与 Outdoor Bike 语义相近
```

随后做 HAR 时，可以有两种用法：

1. Zero-shot prompt matching：

```text
sensor -> sensor embedding
"User performed Walking." -> text embedding
"User performed Outdoor Bike." -> text embedding
选择最相似的文本类别
```

2. Few-shot / linear probing：

```text
sensor -> pretrained sensor encoder -> representation
少量标准 HAR 标签 -> 训练小分类器
```

因此，caption 的作用是提供语义预训练信号，而不是替代最终的标准 HAR 标注。

### 15.4 对当前项目的启发

对方向一来说，SensorLM 的启发不是“我们以后只靠模糊文本做动作识别”，而是：

```text
可以先用弱语义、文本、视频或场景信息，让 IMU 表示靠近人能理解的行为语义；
再用更精确的标签或任务头完成具体 HAR / 手机运镜识别。
```

它适合回答：

```text
如何让 sensor representation 带有语义？
```

但不适合直接回答：

```text
如何获得精确动作边界和精细类别标签？
```

## 16. 结果表现与经典方法对比

SensorLM 的实验结果需要分任务看。它不是在 UCI HAR 这种固定短窗口 benchmark 上和 1D CNN / LSTM 直接比准确率，而是在大规模 wearable sensor-language 设定下评估：

- zero-shot activity recognition
- few-shot adaptation
- cross-modal retrieval
- sensor caption generation
- caption 类型和架构消融

### 16.1 Zero-shot activity recognition

在 20 类 activity recognition 上，SensorLM 明显优于直接把 sensor table 输入给通用 LLM。

| 方法 | AUROC | F1 | Balanced Acc. |
|---|---:|---:|---:|
| Gemma-3-27B | 0.50 | 0.01 | 0.05 |
| Gemini 2.0 | 0.51 | 0.03 | 0.07 |
| SensorLM | 0.84 | 0.29 | 0.31 |

这说明：

```text
通用 LLM 不能直接理解传感器表格；
必须先做 sensor-language alignment。
```

但也要注意：F1 和 Balanced Accuracy 并不高。SensorLM 的强项是 zero-shot 语义对齐，而不是已经达到成熟 HAR 分类器的水平。

### 16.2 细粒度和概念分类

论文还评估了 activity concept classification 和 fine-grained recognition。

| 任务 | SensorLM AUROC | SensorLM F1 | SensorLM BAcc |
|---|---:|---:|---:|
| Environmental context | 0.64 | 0.33 | 0.38 |
| Cardio vs strength | 0.71 | 0.63 | 0.66 |
| Locomotion vs stationary | 0.61 | 0.58 | 0.58 |
| Gym cardio fine-grained | 0.76 | 0.50 | 0.51 |
| Outdoor sports fine-grained | 0.83 | 0.52 | 0.53 |

这些结果说明：模型对某些语义概念有一定理解，但不是所有概念都强。例如 locomotion vs stationary 的 AUROC 只有 0.61，说明长期 wearable 传感器语义和具体活动边界之间仍有差距。

### 16.3 Few-shot learning：与经典 SSL 方法对比

论文把 SensorLM 与三种经典/常用自监督表示学习方法比较：

- SimCLR
- DINO
- MSN, Masked Siamese Network

在 Activity 任务上，SensorLM 的 label efficiency 明显更强。

| 每类标签数 | DINO AUROC | MSN AUROC | SimCLR AUROC | SensorLM AUROC |
|---:|---:|---:|---:|---:|
| 5 | 0.50 | 0.52 | 0.54 | 0.73 |
| 10 | 0.51 | 0.54 | 0.55 | 0.79 |
| 20 | 0.52 | 0.56 | 0.58 | 0.83 |
| 50 | 0.54 | 0.60 | 0.60 | 0.88 |

这说明语言监督对 activity representation 有帮助：即使只有少量标签，SensorLM 的 sensor encoder 也更容易被下游分类器利用。

在 Hypertension 和 Anxiety 这类健康任务上，SensorLM 只表现为 comparable 或略优，并没有像 Activity 任务那样明显拉开。这也说明 SensorLM 的优势主要来自 activity semantic captions。

### 16.4 Cross-modal retrieval

SensorLM 在 sensor-text retrieval 上非常强。

| 任务 | 100 samples R@1 | 100 samples R@5 | 5000 samples R@1 | 5000 samples R@5 |
|---|---:|---:|---:|---:|
| Sensor -> Text | 100.0 | 100.0 | 98.2 | 99.4 |
| Text -> Sensor | 100.0 | 100.0 | 96.7 | 99.2 |

Gemini / Gemma 在该任务上基本不可行或接近失败，主要受上下文长度和传感器表格理解能力限制。

不过这个任务也要谨慎看：retrieval caption 是由同一套自动模板生成体系构造的，和真实开放自然语言检索仍有距离。

### 16.5 Caption generation

SensorLM 能根据 sensor 输入生成 semantic caption，并优于 Gemini/Gemma。

| 方法 | BERTScore P | BERTScore R | BERTScore F1 | METEOR | ROUGE |
|---|---:|---:|---:|---:|---:|
| Gemma-3-27B | 0.82 | 0.85 | 0.83 | 0.19 | 0.11 |
| Gemini 2.0 | 0.86 | 0.87 | 0.86 | 0.26 | 0.20 |
| SensorLM | 0.93 | 0.91 | 0.92 | 0.33 | 0.40 |

这再次说明，专门训练的 sensor-language 模型比直接用通用 LLM 看表格更可靠。

### 16.6 Caption 消融

不同 caption 类型对结果影响很大。

| Caption 类型 | Zero-shot Activity AUROC | Linear Activity AUROC |
|---|---:|---:|
| statistical | 0.51 | 0.76 |
| structural | 0.50 | 0.78 |
| semantic | 0.71 | 0.95 |
| statistical + semantic | 0.66 | 0.84 |
| structural + semantic | 0.84 | 0.94 |
| statistical + structural | 0.49 | 0.79 |
| statistical + structural + semantic | 0.66 | 0.86 |

结论：

```text
semantic caption 是 activity recognition 的关键；
structural + semantic 是整体最好的折中。
```

纯统计或纯结构 caption 几乎不能支持 zero-shot activity recognition，因为里面没有活动概念。

### 16.7 架构消融：CLIP / Cap / CoCa

| 架构 | Zero-shot AUROC | Zero-shot F1 | Linear AUROC | Linear F1 |
|---|---:|---:|---:|---:|
| SensorLM-CLIP | 0.83 | 0.29 | 0.93 | 0.53 |
| SensorLM-Cap | 0.55 | 0.01 | 0.90 | 0.32 |
| SensorLM-CoCa | 0.84 | 0.29 | 0.94 | 0.57 |

结论：

```text
只做 caption generation 不适合 zero-shot classification；
contrastive alignment 是 zero-shot HAR 的关键；
CoCa-like 联合目标整体最好。
```

### 16.8 和“经典 HAR 方法”的关系

这篇论文没有和 1D CNN、LSTM、DeepConvLSTM 在 UCI HAR 这类标准监督 benchmark 上直接比较，因此不能说：

```text
SensorLM 比经典 HAR 分类器更强。
```

更准确的表述是：

```text
在少标签和零样本 sensor-language 设定下，
SensorLM 明显优于直接用通用 LLM，也优于 SimCLR/DINO/MSN 这类自监督 encoder baseline。
```

如果是一个有大量标准标签的封闭 HAR benchmark，传统监督模型仍可能表现更高、更稳定、更轻量。SensorLM 的价值主要在：

- zero-shot / few-shot
- 语义对齐
- sensor-text retrieval
- caption generation
- 大规模弱语义预训练

这对方向一的启发是：如果未来任务涉及人体动作、手机运镜、场景语义和开放标签，SensorLM 这类方法值得关注；如果只是固定类别短窗口分类，它可能过重且不一定优于经典监督模型。

## 17. 结果指标和自监督基线怎么理解

### 17.1 无标签自监督方法如何训练

SensorLM 对比的 SimCLR、DINO、MSN 都不需要人工活动标签。它们的共同思想是：

```text
从同一个 sensor 样本构造不同 view，
要求模型对这些 view 学到一致或可对齐的表示。
```

#### SimCLR

SimCLR 是对比学习。

```text
同一个样本的两种增强 view -> 正样本对
不同样本的 view -> 负样本对
```

训练目标是：

```text
同一样本的两个 view 表示更近；
不同样本的 view 表示更远。
```

在这篇论文里，增强包括 jittering、scaling、time flipping 等时间序列增强。

#### DINO

DINO 是 teacher-student 自蒸馏。

```text
同一个样本的不同增强 view
-> teacher 产生目标表示
-> student 去匹配 teacher 表示
```

它不需要类别标签，目标来自 teacher 网络本身。

#### MSN

MSN 是 Masked Siamese Network。

```text
一个 view 被 mask，只保留可见 patch；
另一个 view 作为较完整参考；
模型学习让 masked view 的表示接近 unmasked view 的表示。
```

它结合了 masked modeling 和对齐思想。

这些方法都不需要人工标签，但它们也没有语言或动作语义。它们主要依靠“同一样本不同增强应表示一致”这个假设来学习表示。

### 17.2 Zero-shot 活动识别结果怎么得到

Zero-shot 的意思是：

```text
测试某个活动识别任务时，不再用该任务的训练标签训练分类器。
```

SensorLM 的做法是：

```text
sensor -> sensor encoder -> sensor embedding

每个候选类别 -> 多个 prompt 文本
例如：
  User performed Run.
  A period of Run was observed.
  Detected a phase of Run.
-> text encoder -> label embedding

选择和 sensor embedding 最相似的 label embedding。
```

所以它是靠 sensor-text embedding space 做分类，不是靠重新训练分类头。

### 17.3 AUROC / F1 / Balanced Accuracy

#### AUROC

AUROC 衡量模型把正样本排在负样本前面的能力。随机猜测通常接近 0.50，越接近 1 越好。

在多分类任务里，通常会把每一类看作 one-vs-rest，再做平均。

SensorLM 在 20 类 activity recognition 上 AUROC 为 0.84，说明它对“这个 sensor 是否更像某类活动”的排序能力不错。

#### F1

F1 是 precision 和 recall 的调和平均，更关心最终离散分类结果是否准确。

20 类 activity recognition 里 SensorLM 的 F1 只有 0.29，说明虽然排序能力不错，但真正选出正确类别仍然不算很强。

#### Balanced Accuracy

Balanced Accuracy 是各类别 recall 的平均，适合类别不均衡场景。20 类随机猜测大约是：

```text
1 / 20 = 0.05
```

Gemma/Gemini 的 BAcc 接近 0.05-0.07，说明几乎随机；SensorLM 为 0.31，明显高于随机，但还不是成熟监督分类器水平。

### 17.4 Few-shot 的“每类标签数”是什么意思

Few-shot learning 不是完全不用标签，而是只用很少标签。

论文里的 `5 / 10 / 20 / 50 labels per class` 意思是：

```text
每个活动类别只抽 5、10、20 或 50 个带标签样本来训练下游线性分类器。
```

例如 20 类 activity recognition，如果是 5 labels per class，总训练样本大约是：

```text
20 x 5 = 100 个带标签样本
```

具体做法是：

```text
冻结预训练好的 sensor encoder
sensor -> representation
用少量标签训练一个线性分类层
```

SensorLM 在 Activity 任务上显著优于 SimCLR/DINO/MSN，说明它的 sensor representation 更容易被少量标签利用。

### 17.5 Sensor-text 检索是什么意思

Sensor-text retrieval 是跨模态检索。

#### Sensor -> Text

输入一个 sensor 样本，在一堆 caption 候选中找最匹配的文本。

```text
sensor query -> sensor embedding
candidate captions -> text embeddings
按 cosine similarity 排序
```

#### Text -> Sensor

输入一段文本，在一堆 sensor 样本中找最匹配的数据。

```text
text query -> text embedding
candidate sensors -> sensor embeddings
按 cosine similarity 排序
```

#### R@1 / R@5

R@K 是 Recall@K：

```text
正确答案是否出现在相似度排名前 K 个结果里。
```

R@1 = 98.2 表示 98.2% 的查询中，正确配对排在第 1 名。

这说明 SensorLM 的 sensor-text embedding 对齐很强。但要注意，这个检索任务用的是同一套模板化 caption 体系构造出来的数据，因此比真实开放自然语言检索更规整。

### 17.6 为什么 CLIP-like 和 CoCa-like 很接近

在 zero-shot classification 上：

```text
CLIP-like: AUROC 0.83, F1 0.29
CoCa-like: AUROC 0.84, F1 0.29
```

确实非常接近。

原因是 zero-shot classification 主要依赖：

```text
sensor embedding 与 label text embedding 的相似度
```

这正是 contrastive loss 学到的能力。因此 CLIP-like 已经很强。

CoCa-like 额外加入 captioning loss，带来的提升主要体现在：

- Linear probing F1 从 0.53 到 0.57。
- Balanced Accuracy 从 0.55 到 0.60。
- 具备 caption generation 能力。
- 更细粒度的文本监督可能改善表示质量。

所以不能说 CoCa-like 在所有指标上大幅超过 CLIP-like。更准确地说：

```text
CLIP-like 已经足够支撑 zero-shot 分类；
CoCa-like 在保持 zero-shot 能力的同时，增加了生成能力，并略微提升下游表示质量。
```

这对当前项目的启发是：如果只想做 activity classification / retrieval，contrastive alignment 可能已经很关键；如果还想要自然语言解释、活动描述和语义生成，就需要加入 captioning/generative objective。
