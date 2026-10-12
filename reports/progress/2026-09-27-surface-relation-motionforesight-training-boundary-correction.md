# 修正 MotionForesight 原始训练边界与 F2 额外坐标链的区别

- 日期：2026-09-27
- 范围：根据官方项目页和公开 `SparseTrackDataset` 代码，修正对 MotionForesight 伪标签、观测输入和 F2 坐标链的描述。
- 状态：ANALYZED；未运行新实验。

## 正确的原始训练链

MotionForesight 的官方描述是：完整视频离线生成未来伪 3D tracks，训练时只把观测帧和观测几何提供给预测器，未来观测被隐藏。公开数据加载代码对完整 `tracks_xyz` 与观测深度生成的 `Pj` 使用同一个最后观测相机变换 `w2c[obs-1]`，并使用观测 `Pj` 的同一归一化统计量。

因此原始训练链存在的可能误差是：

```text
观测 DA3 pointmap 与完整视频 TrackCraft 伪轨迹在 t0 的几何估计不完全一致
```

但它没有额外使用一条“只看观测帧运行的 TrackCraft3R”轨迹作为训练输入。

## F2 额外增加的链路

本项目 F2 在原始 `Pj -> MotionForesight` 之外又加入：

```text
观测期 TrackCraft3R -> observed_t0 / observed history
完整视频 TrackCraft3R -> pseudo_gt future
```

并用观测期 TrackCraft3R 的 t0 query 对完整视频教师轨迹做相似变换。这条额外的 `TrackCraft_observed` 与 `TrackCraft_full` 对齐关系不是 MotionForesight 原始训练接口的一部分。

## 影响判断

因此不能再说“MotionForesight没有输入/标签坐标不一致问题”。它也可能受到 DA3 观测几何与 TrackCraft 未来伪标签之间的误差影响；原始模型依靠大规模训练和可训练适配器学习这种映射。

F2 的特殊风险在于：backbone 已冻结、样本很少，Relation Head 直接拟合 `pseudo_gt - P0`，同时还使用了额外的观测期 TrackCraft 关系特征。此前关于“F2 负结果首先由双 TrackCraft 坐标污染造成”的表述过强，应改为“F2 同时包含原始输入/伪标签几何误差和额外双 TrackCraft 对齐误差，当前实验不能将两者分离”。
