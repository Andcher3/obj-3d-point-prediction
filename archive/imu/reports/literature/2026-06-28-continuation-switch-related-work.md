# 未来延续概率与切换趋势预测的相关工作

日期：2026-06-28

## 1. 总体结论

“未来延续概率”和“切换趋势预测”不是凭空定义的输出形式，但在 HAR 文献中通常不会直接用这两个中文说法。它们更常出现在以下几类问题里：

- activity transition recognition / transition-aware HAR：识别动作切换或过渡动作。
- activity segmentation：在连续流中找动作边界。
- activity forecast / next activity prediction：预测未来动作标签。
- duration modeling / semi-Markov model：建模某个状态会持续多久。
- future IMU forecasting + classification：先预测未来传感器信号，再分类未来动作。

对于当前方向一，最接近的定义是：

```text
当前状态识别 + 未来状态分布预测 + 是否发生切换/边界预测
```

其中“未来延续概率”可以看成当前状态在未来 horizon 内保持不变的概率；“切换趋势预测”可以看成未来 horizon 内是否出现状态边界，以及下一状态是什么。

## 2. 相关工作类型

### A. Transition-aware HAR / Postural Transition Recognition

代表：

- Reyes-Ortiz et al., Transition-Aware Human Activity Recognition Using Smartphones, Neurocomputing 2015/2016.
- UCI Smartphone-Based Recognition of Human Activities and Postural Transitions dataset.

这类工作直接处理活动之间的过渡，例如：

```text
stand-to-sit
sit-to-stand
sit-to-lie
lie-to-sit
stand-to-lie
lie-to-stand
```

它们更接近“切换发生了什么”，但通常是把过渡动作当成额外类别来识别，而不是显式输出未来切换概率。

### B. Activity Segmentation + Forecast

代表：

- P2LHAP: Wearable sensor-based human activity recognition, segmentation and forecast through Patch-to-Label Seq2Seq Transformer.

这类工作把连续传感器流切成 patch，输出 patch-level activity label sequence。它同时做：

```text
recognition
segmentation
forecast
```

它和当前问题非常接近，因为如果模型输出未来 patch 的标签序列，就可以自然得到：

```text
未来是否仍是当前状态
未来第几个 patch 发生状态变化
未来可能切换到哪个状态
```

### C. Duration Modeling / HSMM

代表：

- Bayesian Hidden Semi-Markov Model with Covariate-Dependent State Duration Parameters for High-Frequency Data from Wearable Devices.

HSMM 与 HMM 的关键区别是显式建模状态持续时间。它适合解释“延续概率”：

```text
P(当前状态还会持续多久 | 当前状态已持续多久, 最近观测特征)
```

这类方法不一定追求深度学习最高分类精度，但对“状态持续时间”和“切换风险”有清晰概率意义。

### D. Future IMU Forecasting + Future Activity Classification

代表：

- Human Activity Prediction Based on Forecasted IMU Activity Signals by Sequence-to-Sequence Deep Neural Networks, Sensors 2023.

这类工作先预测未来 IMU 片段，再用分类器识别未来活动。它更像：

```text
past IMU -> future IMU -> future activity label
```

它适合“未来动作预测”，但缺点是未来 IMU 预测误差会传递到分类器。

## 3. 对方向一的输出定义建议

不建议一开始设计复杂的 `time_to_switch` 回归。更稳妥的是先定义离散 horizon 输出：

```text
输入：过去 L 个采样点或最近 K 个窗口的 IMU

输出：
1. 当前人体动作类别 p_h(t)
2. 当前手机/相机运动类别 p_m(t)
3. 未来 Δ 内当前组合状态是否保持 p_continue(Δ)
4. 未来 Δ 的下一状态分布 p_next(s_{t+Δ})
```

其中：

```text
state_t = (human_action_t, phone_motion_t)
p_continue(Δ) = P(state_{t+Δ} = state_t | history)
p_switch(Δ) = 1 - p_continue(Δ)
```

如果输出未来多个 patch 的状态序列：

```text
[state_{t+1}, state_{t+2}, ..., state_{t+H}]
```

则可以从这串序列后处理得到：

- 是否延续。
- 第一次切换位置。
- 切换到哪个状态。
- 切换趋势是否增强。

## 4. 对当前阶段的判断

当前最有参考价值的是 P2LHAP，因为它把 recognition、segmentation、forecast 统一到 patch-level label sequence。它比“单独加一个 continue_prob head”更系统，也比早期 transition-aware HAR 更适合作为当前深度学习方法参考。

对比来看：

- 早期 transition-aware HAR：适合作为“过渡动作/姿态转移曾被作为类别处理”的背景，不适合作为当前方法主线。
- P2LHAP：直接面向连续传感器流，输出 patch-level 标签序列，并覆盖当前识别、边界分割和未来活动预测，更符合方向一中“当前状态 + 是否延续 + 如何切换”的输出需求。

P2LHAP 的局限也需要保留：

- 它主要预测人体活动标签，不包含手机/相机运动标签。
- 它是 closed-set supervised 分类，不能天然处理未知行为。
- 它处理 wearable sensor HAR，不是手机拍摄场景下的人体动作与运镜行为联合建模。
- 如果迁移到当前方向一，需要把单一活动标签序列扩展为双标签序列或组合状态序列。

但如果要做最小原型，可以先用更简单的多任务头：

```text
current human action classification
current phone motion classification
continue / switch binary classification over Δ
next state classification over Δ
```

后续再升级为 patch-level future label sequence prediction。

## 5. 对 P2LHAP 的定位修正

相比早期 `Transition-Aware HAR`，P2LHAP 更适合作为当前“延续概率/切换趋势/未来状态预测”问题的核心参考。

原因：

1. 它是较新的深度学习方法，正式版本发表在 IEEE Internet of Things Journal 2025，arXiv 版本为 2024。
2. 它不是只识别 transition class，而是统一处理：
   - activity recognition
   - activity segmentation
   - activity forecast
3. 它的输出是 patch-level activity label sequence，因此天然适合派生：
   - 当前 patch 的动作类别
   - 活动边界位置
   - 未来 patch 的动作类别
   - 当前状态是否延续
   - 未来是否切换以及切换到哪里
4. 它使用 Patch-to-Label Seq2Seq Transformer，比传统滑动窗口分类更接近连续流式行为理解。

需要注意的限制：

- P2LHAP 主要处理人体活动标签，不直接处理手机/相机运动标签。
- 它的 forecast 主要是未来 activity label sequence，不是未来 IMU 数值序列。
- 它使用 WISDM、PAMAP2、UNIMIB SHAR，不是 UCI HAR，也不是手机拍摄场景数据。
- 如果迁移到方向一，需要把 label 从单一人体活动扩展为组合状态：

```text
state_t = (human_action_t, phone_motion_t)
```

因此，P2LHAP 适合作为“输出形式和序列建模框架”的参考，而不是可以直接照搬的最终方案。
