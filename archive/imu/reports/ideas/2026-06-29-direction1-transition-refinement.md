# 方向一建模细化：Transition Label、自监督 Next Token 与通道独立取舍

日期：2026-06-29

## 1. Transition label 是否可行

用户提出的 `<sit-2-stand>` 这类 transition label 有合理性：某些过渡动作确实有独立动力学模式，例如坐到站、站到坐、开始走、停止走。它们不完全等价于普通边界噪声。

但若把所有过渡都作为普通主类别，会出现类别数量和数据稀疏问题。

若有 `K` 个稳定动作，方向性 transition 最多有：

```text
K * (K - 1)
```

例如 UCI HAR 的 6 类动作会产生最多 30 个有向 transition，加上 6 个稳定动作就是 36 类。这个规模本身不夸张，但真实问题在于：

- 许多 transition 在数据中很少出现。
- transition patch 只占总数据很小比例，类别极不平衡。
- transition 的持续时间不固定，受 patch size 影响很大。
- 如果扩展到人体动作 + 手机/相机运动的组合状态，类别会快速膨胀。

因此更推荐的形式是层级化：

```text
stable_label_t       : 当前稳定动作类别
boundary_flag_t      : 当前 patch 是否处在切换附近
transition_type_t    : 仅在 boundary_flag=1 时预测 source->target
```

也就是说，`<sit-2-stand>` 可以作为辅助 transition-type label，而不应替代主动作标签。

## 2. 预测 next token 的两条路线

### 路线 A：自监督预测未来 IMU token

这是针对标签稀缺更合适的预训练方式：

```text
past IMU patches -> future IMU patches
```

常见实现：

- 连续回归：用 TCN / causal Transformer / encoder-decoder Transformer 预测未来原始 IMU patch，使用 MSE、MAE 或 Huber loss。
- Masked reconstruction：随机遮住时间 patch 或通道 patch，重构被遮住的 IMU 数值。
- CPC / contrastive prediction：用过去表示预测未来表示，通过 InfoNCE 区分真实未来和负样本。
- VQ token prediction：先用 VQ-VAE 或 tokenizer 将 IMU patch 离散化，再像语言模型一样预测下一个离散 code id。

路线 A 学到的是 IMU dynamics，不直接产生 HAR 标签。下游仍需要少量标注数据微调分类头或标签序列预测头。

### 路线 B：监督式未来 label sequence 预测

这是 P2LHAP 更接近的路线：

```text
past / current IMU patches -> future activity label sequence
```

这不是必须先预测未来 IMU token，再把 token 转成 label。更常见做法是：

```text
IMU encoder hidden states
-> future-position queries / decoder
-> 每个未来 patch 的 label logits
```

训练时用 cross entropy，对未来 `H` 个 patch 的标签监督。P2LHAP 采用固定 forecast patch length，例如预测未来 8 个 patch。

在标签稀缺条件下，推荐做成两阶段：

```text
1. 大量无标签 IMU：自监督预训练 encoder
2. 少量有标签序列：微调 future label decoder
```

## 3. 变长序列怎么处理

活动本身是变长的，但训练时不需要让模型一次输出任意长活动段。可以用固定上下文和固定预测 horizon：

```text
context length = L 个历史 patch
forecast horizon = H 个未来 patch
```

模型输出固定长度 `H` 的未来 patch 标签序列。之后再把相邻相同标签合并，得到变长 activity segment：

```text
walk walk walk sit sit
-> walk segment + sit segment
```

若输入序列长度不同，可以用 padding + attention mask。若未来要做更开放的长时预测，再考虑 autoregressive rollout 或 EOS token；当前阶段不建议一开始就做。

## 4. 通道独立是否应该照搬 P2LHAP

P2LHAP 的 channel-independent 设计来自一个明确假设：多通道传感器行为差异较大，噪声通道可能在共享 embedding 或 attention 中干扰其他通道。因此它把每个 patch 限制为单个 sensor channel，并将通道维度展平成 batch 维度，让各通道独立进入 Transformer。

这个思路对手机 IMU 部分成立：

- acc 与 gyro 的物理量不同，尺度和噪声不同。
- x/y/z 轴受手机姿态影响，单轴模式可能差异很大。
- 小数据集上直接早期融合可能让 Transformer 更难学。

但方向一不能完全忽略跨通道关系：

- 三轴 acc 的相对变化反映姿态和步态。
- gyro 三轴共同反映旋转和运镜。
- acc 与 gyro 的相位、能量和耦合关系有助于区分人体运动与手机自身运动。

因此更推荐将 channel independence 作为前端归纳偏置，而不是全程隔离：

```text
每通道独立 patch embedding / shared encoder stem
-> 加 sensor-type embedding 和 axis embedding
-> 轻量 cross-channel fusion
-> temporal encoder / decoder
-> human action head + phone motion head
```

实验上应做三种消融：

1. Early fusion：一个时间 patch 内直接拼接所有通道。
2. P2LHAP-style channel independent：每通道独立编码，再聚合。
3. Hybrid：通道独立前端 + 跨通道融合。

当前最稳的研究选择是先复现或实现 2 作为 P2LHAP 风格基线，再把 3 作为方向一的改进候选。

