# P2LHAP 外部总结核查与勘误

日期：2026-06-28

核查对象：

- `presentation/P2LHAPWearable sensor-based human activity recognition, segmentation and forecast through  Patch-to-Label Seq2Seq Transformer.txt`
- `presentation/P2LHAPWearable sensor-based human activity recognition, segmentation and forecast through  Patch-to-Label Seq2Seq Transformer/`

参考原文：

- P2LHAP: Wearable sensor-based human activity recognition, segmentation and forecast through Patch-to-Label Seq2Seq Transformer
- arXiv: https://arxiv.org/abs/2403.08214

## 总体判断

外部总结主线基本正确：

- 论文目标是统一处理 HAS、HAR、HAF。
- 核心是 Patch-to-Label Seq2Seq 框架。
- 输入是传感器 patch 序列，输出是 patch-level activity label sequence。
- 使用 channel-independent Transformer。
- 使用 smoothing strategy 缓解 over-segmentation。
- 在 WISDM、PAMAP2、UNIMIB SHAR 三个数据集上实验。

但存在几处容易误导的表述。

## 需要纠正的点

### 1. “不用滑动窗口”不能理解为完全不需要固定长度

论文说不使用 fixed-length sliding window 来分割传感器数据，但仍然使用固定 patch size `P`，并且 patch 可以 overlapping 或 non-overlapping。

更准确的说法是：

```text
P2LHAP 不再把一个固定窗口当作一个完整活动样本；
而是把连续数据流切成 patch 序列，再预测 patch-level 标签序列。
```

它缓解了固定窗口难以覆盖变长活动的问题，但没有彻底消除长度超参数。

### 2. “单任务模型解决三任务”要谨慎表述

论文强调用一个 Patch-to-Label Seq2Seq 输出统一支持三类任务，而不是三个独立模型或多个任务头。

但训练时仍然包含：

- patch-level classification loss
- forecast loss
- smoothing-related objective / strategy

所以更准确的说法是：

```text
它用统一的标签序列输出形式覆盖 recognition、segmentation、forecast，
不是完全意义上的只有一个普通分类任务。
```

### 3. Forecast 既包括未来标签预测，也涉及未来传感器信号预测实验

外部总结主要讲了“预测未来活动标签”，这是核心没错。

但原文实验部分还提到：

```text
future activity label prediction
future sensor signal prediction / MSE comparison
```

尤其 Table III 用 MSE 比较 forecasted sensor signals。因此不能只把 forecast 理解成“下一活动类别预测”，它在实验里还包含未来信号预测对比。

### 4. “全面超越 SOTA”应保留论文语境

论文确实声称在三个任务上优于对比方法，但需要注意：

- 数据集是 WISDM、PAMAP2、UNIMIB SHAR。
- 划分方式是连续数据 0.7/0.1/0.2 train/validation/test。
- 不是跨受试者验证。
- 不是 UCI HAR。
- 不是手机运镜或双标签任务。

因此汇报时应说：

```text
在作者设定的三个公开数据集和评价协议下，P2LHAP 表现优于选定对比方法。
```

不要泛化成“它已经解决 HAR/HAF/HAS”。

### 5. 活动边界来自标签序列合并，不是模型直接回归边界

P2LHAP 的边界不是一个独立 regression head 输出的。

流程是：

```text
patch-level label sequence
-> smoothing
-> merge adjacent same-label patches
-> derive segment boundaries
```

这对我们很重要，因为如果方向一要输出“切换趋势”，可以参考这种标签序列派生边界，而不是一开始设计复杂边界回归头。

### 6. 图像页应视为解释性材料，不应把其中具体数值当论文原图

文件夹中的多张图像更像外部 AI 生成的讲解图。它们有助于理解流程，但其中可视化、图标和部分数值不应直接当作论文原始图表引用。

正式汇报时，具体数值应以论文 Table/Figure 为准。

## 对当前方向一的重要补充

P2LHAP 对“未来延续概率”和“切换趋势预测”的最大价值是输出形式：

```text
预测未来 H 个 patch 的标签序列
```

从这个序列可以派生：

- 当前状态是否延续。
- 第一次切换发生在哪个 patch。
- 下一状态是什么。
- 切换概率随 horizon 如何变化。

如果迁移到方向一，可以扩展为：

```text
human_label_sequence
phone_motion_label_sequence
```

或：

```text
combined_state_sequence = (human_action, phone_motion)
```

这样比单独设计 `continue_prob_head` 和 `switch_head` 更系统。

