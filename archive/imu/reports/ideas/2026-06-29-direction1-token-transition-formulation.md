# 方向一想法草案：IMU Token、动作切换与双标签序列预测

日期：2026-06-29

## 1. 核心判断

这次想法的主线是有价值的：把连续 IMU 信号切成 patch/token 序列，再让模型输出当前和未来的动作状态序列。这比单个窗口分类更贴近方向一里“未来延续概率”和“切换趋势预测”的要求，也更容易和 P2LHAP 的 patch-to-label Seq2Seq 思路连接。

需要修正的一点是：不建议把“切换动作”直接当作一种普通动作类别混入标签空间。切换不是 walking、standing 这类稳定语义动作，而是相邻状态之间的边界或过渡现象。更稳妥的定义是：

```text
IMU patch token sequence
-> 每个 patch 的人体动作标签序列
-> 每个 patch 的手机/相机运动标签序列
-> 由标签序列派生或辅助预测切换边界、延续概率和下一状态
```

## 2. 推荐的任务形式

设连续 IMU 输入被切成 patch：

```text
x_t in R^(P x C)
P = 每个 patch 的时间长度
C = 传感器通道数，例如 acc_x/y/z + gyro_x/y/z
```

经过 tokenizer 和 encoder 后得到 token 表示：

```text
z_t = Encoder(Tokenize(x_t))
```

方向一更合适的输出不是单一动作类别，而是两个标签流：

```text
human_label_t       : 人体动作，例如 walking / upstairs / sitting
phone_motion_label_t: 手机或相机运动，例如 static / pan / tilt / shake / raise
```

如果要做未来预测，可以输出未来 H 个 patch 的标签序列：

```text
human_label_{t+1:t+H}
phone_motion_label_{t+1:t+H}
```

然后从预测标签序列中派生：

```text
continue_prob(k) = 未来 k 个 patch 内保持当前状态的概率
switch_prob(k)   = 第 k 个 patch 前后发生状态切换的概率
time_to_switch   = 第一次预测到标签变化的位置
next_state       = 切换后的下一状态类别
```

这样比单独设计一个模糊的 `continue/switch head` 更系统，也比把 transition 当成一个普通类别更清晰。

## 3. Patch 内包含切换怎么办

如果某个 patch 跨过了动作边界，不建议简单把它替换成一个特殊动作 `SWITCH`。更好的做法按标注精度分三种：

1. 有精确时间戳标注：
   - 对 patch 内各动作占比做 soft label。
   - 例如 70% walking + 30% standing，则标签分布为 `[walking: 0.7, standing: 0.3]`。
   - 额外设置 `boundary_flag = 1`，表示该 patch 含有边界。

2. 只有 patch/window 级硬标签：
   - 可以用中心点标签或多数标签作为主标签。
   - 同时根据相邻 patch 标签是否变化构造 `boundary_flag`。

3. 完全没有连续边界标注：
   - 不能真正训练切换预测，只能做普通窗口分类。
   - UCI HAR 基本属于这一类，因此它适合做 tokenizer、encoder 和当前动作分类的 sanity check，不适合直接验证“切换趋势预测”。

因此，`SWITCH` 更适合作为辅助边界变量，而不是替代原始动作标签。

## 4. “预测 next token”有两种含义

这里要区分两类 next-token prediction：

### A. 预测未来 IMU token

这是自监督或预测式预训练：

```text
历史 IMU tokens -> 未来 IMU tokens
```

如果 token 是连续向量，可以做回归预测，例如 MSE、MAE、Huber loss。如果先用 VQ/VAE 把 IMU 离散化，也可以像语言模型一样预测离散 token id。

这类任务能学到运动动力学，但不直接给出 HAR 标签。

### B. 预测未来 label token

这是更贴近 P2LHAP 和方向一需求的监督式序列预测：

```text
历史 IMU tokens -> 未来动作标签 tokens
```

输出的是人体动作标签序列和手机/相机运动标签序列。延续概率、切换趋势、下一状态都可以从这个标签序列中推导。

当前更建议先从 B 开始，因为它和 HAR、segmentation、forecast 的关系最直接。

## 5. “多通道分离”应如何理解

多通道分离不应该简单理解为把 x/y/z、acc/gyro 完全独立处理。IMU 中的通道关系很重要，例如：

- acc 三轴的相对变化反映姿态和步态方向。
- gyro 三轴反映旋转、抖动和运镜。
- acc 与 gyro 的耦合有助于区分人体移动和手机自身运动。

更合理的做法是：

```text
channel-wise stem / patch embedding
+ sensor-type embedding
+ axis embedding
+ cross-channel fusion
+ shared temporal encoder
+ human-motion head
+ phone-motion head
```

也就是说，前端可以保留通道结构，但后面必须允许跨通道交互。对方向一而言，更重要的是“人体动作因素”和“手机/相机运动因素”的表示解耦，而不是把原始传感器轴完全拆开。

## 6. 一个可落地的模型草图

```text
IMU stream
  -> patching
  -> channel/sensor-aware patch embedding
  -> temporal Transformer / TCN encoder
  -> current patch recognition heads
       - human action classification
       - phone/camera motion classification
  -> future label decoder
       - future human label sequence
       - future phone motion label sequence
  -> auxiliary transition heads
       - boundary flag
       - time-to-switch
       - continue probability
```

训练目标可以包括：

```text
L = L_current_human
  + L_current_phone
  + L_future_human
  + L_future_phone
  + lambda_boundary * L_boundary
  + lambda_consistency * L_sequence_boundary_consistency
```

其中 `boundary` 最好与预测标签序列保持一致，例如相邻 patch 标签变化时 boundary 应该更高。

## 7. 当前阶段的落地路线

第一阶段：UCI HAR 原型

- 目标：验证 IMU patch/tokenizer + encoder + 当前动作分类是否跑通。
- 输出：单标签人体动作分类。
- 不做：真实切换预测，因为 UCI HAR 没有连续动作边界。

第二阶段：连续 HAR 数据集

- 目标：复现或改造 P2LHAP 风格的 patch-to-label 序列预测。
- 输出：人体动作标签序列、segmentation、forecast。
- 重点：验证 future label sequence 是否能稳定派生 continuation/switch 指标。

第三阶段：方向一扩展

- 目标：从单标签人体动作扩展为双标签序列。
- 输出：
  - `human_action_sequence`
  - `phone_motion_sequence`
  - derived continuation/switch outputs
- 数据：可能需要自采手机 IMU + 视频，或从 gyro/optical flow 构造手机/相机运动伪标签。

## 8. 可写成的研究问题雏形

```text
面向手机 IMU 行为理解的双标签 patch-to-label 序列预测：
联合建模人体动作、手机/相机运动及其未来切换趋势
```

更技术化的版本：

```text
Transition-aware Dual-Stream Patch-to-Label Transformer
for Joint Human Activity and Phone Motion Forecasting from Smartphone IMU
```

这个方向的关键不在于“把动作切换写成特殊 token”本身，而在于把方向一从静态 HAR 分类推进到连续序列建模：当前识别、边界分割、未来标签预测和双因素动作分解可以在同一个 patch-level 输出框架下统一起来。

