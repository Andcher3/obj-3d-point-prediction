# 动作切换时机与下一动作预测的相关任务形态

日期：2026-06-29

## 核心结论

预测“什么时候切换”和“切换到什么动作”不是强行构造出来的问题。在近年相关工作中，它通常以以下任务形式出现：

1. Human Activity Forecasting / patch-level future label prediction
2. Long-Term Action Anticipation / future action segmentation
3. Segment-level action anticipation with action duration
4. Short-term action anticipation with time-to-action regression

这些任务大多不把 `sit-to-stand` 当作和 `sit`、`stand` 平级的普通动作类，而是预测未来标签序列、未来动作段及其持续时间，或预测下一动作类别加 time-to-action。

## 1. Sensor HAR 中的 future label sequence

代表工作：

- P2LHAP: Wearable-sensor-based HAR, segmentation, and forecast through Patch-to-Label Seq2Seq Transformer.

做法：

```text
past/current sensor patch sequence
-> Transformer encoder-decoder
-> future patch-level activity label sequence
```

它直接预测未来若干 patch 的动作类别。切换时机和下一动作可以从预测序列中派生：

```text
current: sit
future: sit sit sit stand stand
                 ^
time_to_switch = 第一次标签变化的位置
next_action = stand
```

优点：最贴近 IMU/HAR。

局限：需要连续传感器流和 patch-level 标签；不是 UCI HAR 这种独立窗口数据最擅长验证的任务。

## 2. Long-Term Action Anticipation / future action segmentation

代表任务：

- Ego4D Long-Term Action Anticipation：预测未来动作序列。
- Breakfast / 50Salads 等视频动作数据上的 long-term action anticipation。

做法：

```text
observed video/action features
-> future action sequence
```

部分方法输出的是未来动作 label 序列；部分方法输出未来动作段：

```text
[(action_1, duration_1), (action_2, duration_2), ...]
```

如果输出 duration，那么切换时间天然由累计 duration 得到：

```text
time_to_switch_1 = duration_1
time_to_switch_2 = duration_1 + duration_2
```

这类任务比单纯 next-action classification 更接近“什么时候切换 + 切换到什么动作”。

## 3. FUTR 类 segment-level decoder

Future Transformer (FUTR) 的关键思想是用一组有序 action queries 并行预测未来动作段。每个 future query 可以预测：

```text
future_action_label
future_action_duration
```

它还引入 action segmentation loss，说明识别已观察片段中的动作边界有助于预测未来动作。

迁移到 IMU 时，可以理解为：

```text
IMU encoder
-> M 个 future segment queries
-> 每个 query 输出 action label + duration
```

优点：直接输出“下一个动作”和“动作持续多久/何时切换”。

风险：duration 标注和连续动作段标注要求更高。

## 4. Time-to-action / time-to-contact 回归

短期动作预期任务常见形式是：

```text
next_action_class + time_to_action
```

例如 Ego4D 的 short-term object interaction anticipation 会为未来交互对象预测 verb，并估计 time-to-contact。

迁移到 IMU 时可以定义为：

```text
next_human_action
time_to_human_action_change
next_phone_motion
time_to_phone_motion_change
```

这就是分类头 + 时间回归头的形式。

优点：直接回答用户关心的问题。

局限：如果未来有多个可能动作，只输出一个 next action 会损失不确定性。

## 对方向一的建议

当前不必坚持 `<sit-2-stand>` transition token 作为主类。更有领域依据的三种实现是：

1. Patch-level future label sequence
   - 预测未来 H 个 patch 的动作标签。
   - 从第一次标签变化得到切换时机和下一动作。

2. Segment-level action + duration
   - 预测未来若干动作段，每段包含类别和持续时间。
   - 从 duration 得到切换时机。

3. Next action + time-to-switch
   - 直接预测下一动作类别和距离切换还有多久。
   - 适合短期实时预警。

对于方向一，优先级建议：

```text
P2LHAP-style future label sequence
-> segment-level action + duration
-> next action + time-to-switch head
```

这样既贴近近期 sensor HAR 工作，又能自然解释“延续概率”和“切换趋势”。

