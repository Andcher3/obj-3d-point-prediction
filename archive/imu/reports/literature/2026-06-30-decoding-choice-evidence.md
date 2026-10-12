# Future Label Sequence 中 Decoder 选择的相关方法依据

日期：2026-06-30

## 结论

对于当前方向一的 `past IMU patches -> current/future label sequence` 任务，第一版更应采用非自回归/并行输出式 decoder，而不是自回归 decoder。

这不是单纯工程偏好，而是来自相近任务的趋势：

- P2LHAP 在 sensor-based activity forecasting 中直接输出 patch-level label sequence，包括未来活动标签。
- FUTR 在 long-term action anticipation 中显式比较了 autoregressive decoding 与 parallel decoding，结论是 parallel decoding 更准确、更快，且避免错误累积。
- ActFusion 用 anticipative masking 和 learnable mask tokens 预测不可见未来动作，也不是逐步自回归生成，而是把未来看成待补全的序列区域。

## 1. P2LHAP：Sensor 领域的直接参考

P2LHAP 的任务形式：

```text
sensor data stream -> patches -> patch-level activity labels
```

输出包含未来活动标签，因此切换时机与下一动作来自未来 patch label sequence。

它的重点不是自回归地一个标签一个标签生成，而是用 encoder-decoder 框架得到统一的 patch-level label sequence。

对方向一的启发：

```text
IMU patches -> future patch label sequence
```

优先于：

```text
IMU patches -> transition class
```

## 2. FUTR：明确支持 parallel decoding

FUTR 是 long-term action anticipation 中非常相关的参考。它预测未来 action labels 和 durations。

它采用：

```text
learnable action queries
-> Transformer decoder
-> parallel prediction of future action sequence
```

FUTR 论文明确对比了：

- FUTR-A：autoregressive decoding，使用前一步输出动作作为下一步输入，并使用 causal mask。
- FUTR-M / FUTR：parallel decoding，使用 action queries 并行预测未来动作。

论文报告 parallel decoding 的理由：

- 避免 autoregressive decoding 的错误累积。
- 推理更快。
- decoder 中未来 action queries 可以建模双向关系。

对方向一的启发：

```text
q_0...q_H current/future queries
-> decoder
-> parallel output current/future label tokens
```

这与我们目前构想高度一致。

## 3. ActFusion：未来动作预测作为 mask completion

ActFusion 统一 temporal action segmentation 和 long-term action anticipation。

它的核心不是逐步生成未来标签，而是：

```text
visible part -> observed actions
invisible future part -> learnable mask tokens
decoder denoises / predicts action labels
```

这说明近期方法也常把未来预测建模为“补全不可见序列区域”，而不是必须做自回归 label generation。

对方向一的启发：

未来 IMU label prediction 可以类比为：

```text
observed IMU tokens + future query/mask tokens
-> complete current/future label sequence
```

## 4. 对当前方法的推荐

第一版主结构应为：

```text
IMU patch embedding
-> Transformer encoder
-> non-autoregressive query Transformer decoder
-> shared MLP classifier
-> p(y_t), p(y_{t+1}), ..., p(y_{t+H})
```

其中：

- `q_0` 对应当前分类。
- `q_1...q_H` 对应未来第 1 到 H 个 patch。
- decoder query 之间不使用 causal mask，允许双向交互。
- 每个输出 token 接共享 MLP classifier。

自回归 decoder 可以保留为后续消融：

```text
previous predicted label token -> causal decoder -> next label
```

但不建议作为第一版，因为它更复杂、训练/推理不一致，并且在 FUTR 这类相近任务中未体现为更优选择。

## 5. 应补充的消融实验

为了避免主观选择，后续实验应比较：

1. parallel query decoder
2. causal / autoregressive decoder
3. no decoder, encoder pooling + MLP direct output

若实验结果支持 parallel query decoder，才能把它作为正式方法选择。

