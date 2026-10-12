# Future Patch Label Sequence：模型结构与训练策略总结

日期：2026-06-30

## 1. 最小可落地模型

第一版目标是先验证：

```text
past IMU patches -> future patch label sequence
```

不急着加入双标签、transition type、语言对齐或复杂自监督。

### 输入

```text
X in R^(L x P x C)
L = 历史 patch 数
P = 每个 patch 内采样点数
C = IMU 通道数，例如 6 或 9
```

### Patch Embedding

每个 patch 映射为一个 token：

```text
x_t in R^(P x C) -> z_t in R^D
```

可选实现：

- early fusion：把 `P x C` flatten 后线性投影。
- channel-independent：每个通道独立 patch embedding，再聚合。
- hybrid：通道独立前端 + 跨通道融合。

第一版建议先做 early fusion 或 channel-independent baseline，后续再比较 hybrid。

### Temporal Encoder

用一个轻量时序编码器建模历史 patch 之间的关系：

```text
[z_1, ..., z_L] -> [h_1, ..., h_L]
```

候选：

- TCN：小数据更稳，训练快。
- Transformer Encoder：更贴近 P2LHAP / PatchTST，适合后续扩展。
- GRU/LSTM：可作为较传统序列基线。

第一版建议：

```text
Patch Embedding + 2-4 层 Transformer Encoder
```

或：

```text
Patch Embedding + TCN
```

### Future Label Decoder

预测未来 H 个 patch 的标签：

```text
[h_1, ..., h_L] -> p(y_{t+1}), ..., p(y_{t+H})
```

最简单做法：

```text
context vector = mean_pool / last hidden state
context vector -> MLP -> H x num_classes logits
```

更强做法：

```text
learnable future queries q_1...q_H
q_i cross-attend to encoder outputs
-> 每个 query 输出一个未来 patch label
```

第一版建议从 MLP decoder 开始，跑通后再换 query decoder。

## 2. 推荐研究版结构

方向一最终更适合：

```text
IMU patches
-> channel-aware patch embedding
-> temporal encoder
-> future query decoder
-> human action head
-> phone motion head
```

结构草图：

```text
IMU sequence
  -> patching
  -> channel-independent / hybrid embedding
  -> positional encoding
  -> temporal Transformer encoder
  -> future queries q_1...q_H
  -> decoder / cross-attention
  -> future human label logits
  -> future phone-motion label logits
```

如果只做人体动作，可以先省略 phone-motion head。

## 3. 训练目标

### 监督式 future label prediction

核心 loss：

```text
L_future = Σ CE(p(y_{t+k}), y_{t+k}), k = 1...H
```

如果同时预测当前标签：

```text
L_current = CE(p(y_t), y_t)
L = L_current + λ * L_future
```

如果做双标签：

```text
L = L_human_current
  + λ1 * L_human_future
  + λ2 * L_phone_current
  + λ3 * L_phone_future
```

### 类别不平衡处理

连续 HAR 中静态动作或长持续动作可能占比过高。可选：

- class-weighted cross entropy
- focal loss
- 对包含切换附近的样本提高采样概率
- 评估时单独报告 transition-near performance

## 4. 自监督预训练策略

由于标签稀缺，建议采用两阶段训练。

### 阶段 1：无标签 IMU 预训练

目标是让 encoder 学到 IMU dynamics。

可选任务：

```text
masked reconstruction:
  mask 部分 patch/channel，重构原始 IMU

future regression:
  past patches -> future IMU patches

contrastive prediction:
  past representation -> future representation
  用 InfoNCE 区分真实未来和负样本
```

第一版最推荐：

```text
masked patch reconstruction
```

原因：实现直接，不需要离散 tokenizer，也比长时未来回归更稳定。

### 阶段 2：有标签微调

保留预训练 encoder：

```text
pretrained encoder + future label decoder
```

用少量连续标签训练：

```text
past IMU patches -> future labels
```

可比较：

- from scratch
- self-supervised pretrained encoder
- frozen encoder + linear/query decoder
- full fine-tuning

## 5. 训练样本构造

从连续数据流中滑动取样：

```text
历史窗口：L 个 patch
预测窗口：H 个 patch
```

训练样本：

```text
input  = patches[t-L+1 : t]
target = labels[t+1 : t+H]
```

如果要同时预测当前：

```text
current_target = label[t]
future_target  = labels[t+1 : t+H]
```

## 6. 评价指标

基础预测指标：

- future label accuracy
- macro-F1
- per-horizon accuracy：分别看未来第 1、2、...、H 个 patch 的准确率

切换相关指标：

- next-action accuracy
- time-to-switch error
- switch within horizon AUROC / F1
- continuation probability calibration

其中：

```text
time_to_switch = 预测序列第一次不同于当前标签的位置
```

## 7. 推荐实验顺序

1. 监督式 baseline：
   - early fusion patch embedding
   - TCN 或 Transformer encoder
   - MLP future decoder

2. 模型结构消融：
   - early fusion
   - P2LHAP-style channel independent
   - hybrid channel-independent stem + cross-channel fusion

3. Decoder 消融：
   - MLP decoder
   - future query decoder
   - autoregressive decoder

4. 预训练消融：
   - no pretraining
   - masked reconstruction pretraining
   - future IMU regression pretraining
   - contrastive prediction pretraining

5. 方向一扩展：
   - single human action sequence
   - human action + phone motion dual sequence

