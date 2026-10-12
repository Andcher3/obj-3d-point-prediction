# F2 两次 TrackCraft3R 坐标差异的含义说明

- 日期：2026-09-27
- 范围：澄清观测期 TrackCraft3R、完整视频 TrackCraft3R 与 MotionForesight 原始训练坐标链的差异。
- 状态：ANALYZED；未运行新实验。

## 关键区分

F2 中用于关系学习的量是点对之间的相对向量：

```text
r_ij = p_j - p_i
```

因此，所有点共同增加同一个平移量时，边向量不变：

```text
(p_j + t) - (p_i + t) = p_j - p_i
```

单纯的整体平移不会污染关系头。会污染关系监督的是不同 query 点的偏差不一致，或残余的尺度、旋转和局部几何误差。

## F2 的实际链路

```text
观测期 TrackCraft3R -> observed_t0
完整视频 TrackCraft3R -> teacher_t0 + teacher_future
```

代码先用 t0 query 估计相似变换，把 `teacher_t0` 和 `teacher_future` 对齐到观测坐标。对齐后仍有残差，说明两次运行不是一个简单的共同平移/旋转/尺度变化。

Relation Head 实际学习的是：

```text
aligned_teacher_future_edge - frozen_MotionForesight_baseline_edge
```

如果残余 t0 几何误差已经接近这个差值的大小，head 会把一部分静态几何误差当作未来关系修正。

## 与 MotionForesight 的区别

原始 MotionForesight 训练协议把观测点、未来轨迹和相机变换统一到最后观测相机坐标，不需要用两次不同上下文的 TrackCraft3R 运行结果拼出监督。因此它不会天然遇到 F2 这类跨 TrackCraft 上下文的关系标签污染。

如果把同样的错位标签直接用于重新训练 MotionForesight，它同样会受到影响。区别在于 F2 中 backbone 已冻结，只有新增 Relation Head 被迫拟合这类残差；baseline 只被用于比较，并不会根据这批伪标签更新。

## 边界

这项分析说明 F2 监督链存在污染风险，不等于证明它是所有负结果的唯一原因。特征可迁移性、图连接和 Relation Head 表达能力仍可能共同影响结果。
