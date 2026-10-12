# LLM 文本序列预测与 IMU/时间序列预测的联系

日期：2026-06-28

## 1. 总体判断

已有方法确实在把“时间序列预测”和“NLP 中的语言模型预测”联系起来，但联系分三层：

1. 架构层：用 Transformer / decoder-only Transformer 做 autoregressive forecasting。
2. Token 层：把连续数值序列量化成离散 token，再像语言模型一样用 cross-entropy 预测未来 token。
3. 语义/对齐层：把时间序列 patch 映射到 LLM embedding 空间，利用冻结 LLM 或语言 prompt 做预测/理解。

对 IMU 来说，第一层已经很常见；第二、三层正在兴起，但还不是手机 IMU 未来序列预测的成熟标准方案。

## 2. 代表工作

### Chronos

Chronos: Learning the Language of Time Series.

核心做法：

```text
real-valued time series
-> scaling + quantization
-> discrete tokens
-> T5-style language model
-> cross-entropy loss 预测未来 token
```

它是最像“把时间序列当语言”的方法之一。

### Lag-Llama

Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting.

核心做法：

```text
past values + lag covariates
-> decoder-only transformer
-> probabilistic future forecasting
```

它更像 LLaMA 风格的时间序列预测模型，但主要是通用单变量 probabilistic forecasting。

### Time-LLM

Time-LLM: Time Series Forecasting by Reprogramming Large Language Models.

核心做法：

```text
time series patches
-> reprogramming / text prototypes
-> frozen LLM backbone
-> forecast projection
```

它不是把数字直接写成文本喂给 LLM，而是把时间序列 patch 映射成 LLM 更容易处理的表示。

### SensorLLM

SensorLLM 把 motion sensor 与 LLM 联系起来，但主要任务是：

```text
sensor-language alignment + HAR classification
```

而不是：

```text
past IMU -> future IMU
```

因此它证明了 LLM 能通过 alignment 理解 sensor trend，但不能直接说明 LLM 已经是 IMU future forecasting 的最佳方法。

## 3. 对当前课题的启发

如果要做“已有 IMU -> 未来 IMU”，可以考虑三种路线：

### 路线 A：连续回归

```text
past IMU -> Transformer/TCN/GRU -> future IMU
loss = MSE / MAE
```

这是最直接、工程上最稳的路线。

### 路线 B：离散 token 预测

```text
past IMU
-> 每个通道量化为 token
-> autoregressive transformer
-> future IMU tokens
-> dequantize
```

这最接近 LLM 文本预测，但要解决多通道、量化误差和物理连续性问题。

### 路线 C：Patch-level foundation model

```text
past IMU patches
-> patch embedding / reprogramming
-> time-series foundation model or frozen LLM
-> future patches
```

适合探索 zero-shot/few-shot 或跨数据泛化，但实现复杂度更高。

## 4. 当前建议

短期不要直接上完整 LLM。更稳妥的实验顺序：

1. 用 TCN/Transformer 做连续 future IMU regression baseline。
2. 再尝试 Chronos/Lag-Llama 类 time-series foundation model，看是否能预测单通道或多通道 IMU。
3. 最后再考虑把 IMU 量化成 token，做真正“类语言模型”的 autoregressive IMU token prediction。

