# Future Label Sequence：Transformer Encoder-Decoder 结构修正

日期：2026-06-30

## 1. 关键修正

此前将 `MLP future decoder` 称为 decoder 不够准确。MLP 更适合作为最简单的 prediction head，而不是严格意义上的序列 decoder。

考虑到当前研究主线是：

```text
大量无监督 IMU 数据
+ Transformer 表示学习
+ 当前与未来 label sequence prediction
```

更自然的结构应为：

```text
IMU patches
-> Transformer Encoder
-> Transformer Decoder
-> MLP Classifier
-> predicted label sequence
```

其中 Transformer Decoder 输出的是一组当前/未来位置的输出 tokens，MLP classifier 对每个输出 token 做分类。

## 2. 推荐结构

输入历史 IMU patches：

```text
X_{t-L+1:t}
```

Patch embedding 后进入 Transformer Encoder：

```text
z_{t-L+1:t} = PatchEmbed(X_{t-L+1:t})
memory = TransformerEncoder(z_{t-L+1:t})
```

构造一组输出查询 tokens：

```text
q_0, q_1, ..., q_H
```

其中：

- `q_0` 用于当前 patch 分类。
- `q_1...q_H` 用于未来第 1 到第 H 个 patch 的分类。

这些 query tokens 进入 Transformer Decoder，并对 encoder memory 做 cross-attention：

```text
o_0, o_1, ..., o_H = TransformerDecoder(q_0...q_H, memory)
```

最后使用共享 MLP classifier：

```text
logits_k = MLP(o_k)
```

得到：

```text
p(y_t), p(y_{t+1}), ..., p(y_{t+H})
```

## 3. Decoder 形式

第一版更推荐非自回归 query decoder：

```text
learnable current/future queries
-> decoder self-attention
-> cross-attention to encoder memory
-> parallel output all positions
```

原因：

- 不需要 teacher forcing。
- 输出长度固定为 `H+1`，训练稳定。
- 每个未来位置由 query index / horizon embedding 表示。
- 输出 tokens 之间可以通过 decoder self-attention 建模标签序列结构。

自回归 decoder 也可以做，但复杂度更高：

```text
previous predicted labels/tokens -> causal decoder -> next label
```

它更像语言模型，但需要处理训练/推理不一致、错误累积和起始 token 设计。当前阶段不建议优先做。

## 4. 训练目标

监督式 label sequence loss：

```text
L = CE(p(y_t), y_t) + λ * Σ_{k=1}^H CE(p(y_{t+k}), y_{t+k})
```

可以给远期预测更小权重：

```text
L_future = Σ_{k=1}^H w_k CE(p(y_{t+k}), y_{t+k})
```

其中 `w_k` 可随 horizon 增大而减小。

## 5. 与无监督预训练的关系

无监督阶段建议先预训练 Transformer Encoder：

```text
masked IMU patches -> reconstruct masked IMU patches
```

可采用 MAE 风格结构：

```text
visible IMU patches -> encoder
mask queries + encoded visible tokens -> reconstruction decoder
-> reconstruct masked patches
```

预训练后：

- 保留 encoder。
- 丢弃或不复用 reconstruction decoder。
- 新增 label sequence Transformer decoder 和 MLP classifier。
- 用少量标签训练/微调当前与未来 label sequence prediction。

后续也可以尝试联合多任务：

```text
L = L_label_sequence + α * L_reconstruction
```

但第一版建议先分阶段，避免训练目标互相干扰。

## 6. 推荐命名

可以将该方法暂命名为：

```text
Transformer Encoder-Decoder for Future IMU Patch Label Sequence Prediction
```

或更贴近方向一：

```text
Future Label Query Transformer for Smartphone IMU Activity Forecasting
```

