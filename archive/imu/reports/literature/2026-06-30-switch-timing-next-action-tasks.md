# 动作切换时机与下一动作预测的相关任务整理

日期：2026-06-30

## 结论

较新的相关任务通常不把每一种动作切换都定义成独立主类别，而是采用更结构化的输出：

1. 预测未来 patch/frame-level 标签序列，再由标签变化推导切换时机。
2. 预测未来动作段的类别与持续时间，切换时机由持续时间累加得到。
3. 在 action localization 中直接预测动作段起止边界与类别。

因此，方向一中“动作切换时机以及下一个动作”的实现不必强行设计多个 transition label head。更自然的做法是先预测未来标签序列或未来动作段。

## 代表任务

### 1. Sensor-based Activity Forecasting / Patch-to-Label

代表：P2LHAP。

做法：

```text
连续传感器流 -> patch 序列 -> patch-level activity label sequence
```

未来预测：

```text
预测未来 H 个 patch 的活动标签
```

切换时机：

```text
未来标签序列中第一次 label != 当前 label 的位置
```

下一动作：

```text
第一次变化后的标签
```

边界：

```text
平滑预测标签序列 -> 合并相邻同类 patch -> 得到 start/end boundaries
```

### 2. Long-Term Action Anticipation

代表：FUTR / Future Transformer。

做法：

```text
观察过去一段视频/动作序列 -> 预测未来多个 action segments
```

输出不是简单逐帧标签，而是：

```text
[(action_1, duration_1), (action_2, duration_2), ...]
```

切换时机：

```text
duration_1, duration_1 + duration_2, ...
```

下一动作：

```text
action_1 或当前动作结束后的 action_2
```

这类任务对方向一很有启发，因为它把“下一个动作是什么”和“什么时候切换”统一在 label + duration 结构中。

### 3. Temporal Action Segmentation + Anticipation

代表：ActFusion。

做法：

```text
可见片段做 segmentation
未来不可见片段用 mask token 表示
模型生成/去噪完整 action sequence
```

它更像“补全未来动作序列”，而不是专门设计 transition 类别。未来切换点仍然来自预测序列中的标签变化。

### 4. Temporal Action Localization / Online Temporal Action Localization

代表：ActionFormer、MATR。

做法：

```text
输入连续视频特征 -> 输出 action instance = (start, end, class)
```

ActionFormer 类方法常见形式是：

```text
每个时间点预测 action class
+ 到动作 start/end 的距离
```

Online TAL 进一步要求在流式输入中尽早定位正在发生的动作边界。它更适合“精确定位当前动作何时开始/结束”，但不一定天然预测未来下一动作。

## 对方向一的建议

近期不要优先采用大量 `<A-2-B>` transition 类别作为主输出。

更自然的两条路线是：

### 路线 A：Patch-level future label sequence

```text
past IMU patches -> future human labels / phone-motion labels
```

派生：

```text
continue probability
switch time
next action
```

这是最贴近 P2LHAP、也最容易在连续 HAR 数据上落地的路线。

### 路线 B：Segment-level label + duration

```text
past IMU patches -> [(label_1, duration_1), (label_2, duration_2), ...]
```

它对“什么时候切换以及下一个动作是什么”表达更直接，但训练和标注要求更高。

## 推荐阶段性选择

1. 第一阶段采用路线 A：future patch label sequence。
2. 如果未来数据具有较清晰动作段边界，再尝试路线 B：label + duration。
3. `<sit-2-stand>` 这类 transition label 暂时只作为分析标签或辅助实验，不作为主任务定义。

