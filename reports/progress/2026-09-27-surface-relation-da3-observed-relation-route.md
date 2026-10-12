# 仅用 DA3 pointmap 构造观测关系特征的路线评估

- 日期：2026-09-27
- 状态：ANALYZED；尚未运行新实验

## 判断

可以取消观测期 TrackCraft3R，只用 DA3 pointmap 构造观测关系特征。该方案能移除 observed-only TrackCraft 与 full-video TrackCraft 之间的跨上下文登记误差，更接近 MotionForesight 的原始输入链。

## 推荐的最小版本

以最后观测帧为关系参考帧：

1. 在 query 像素位置采样 DA3 pointmap 的三维坐标；
2. 按 query 像素位置建立 2D symmetric kNN 图；
3. 对每条边输入两点的三维相对向量、距离、二维相对位置，以及冻结 MotionForesight 的 `P0/H` 特征；
4. 完整视频 TrackCraft3R 只用于离线未来伪标签；
5. 观测 pointmap 和未来标签均使用最后观测相机坐标及同一归一化约定。

该版本不需要观测期 TrackCraft3R，也不需要估计两次 TrackCraft 输出之间的 t0 登记变换。

## 仍需保留的限制

- DA3 pointmap 与完整 TrackCraft 标签之间仍可能存在局部几何误差；这类误差不能被路线自动消除。
- 2D 邻近点不一定是 3D 邻近点，图拓扑仍需保守使用。
- 若使用多帧 DA3 关系历史，需要把各帧点图变换到同一相机坐标，并解决同一 query 在不同帧的像素对应。仅使用最后观测帧可避免重新引入 TrackCraft 或其他跟踪器。

## 实验含义

建议将该路线作为新的干净对照：

```text
baseline
baseline + DA3-pointmap relation feature
baseline + observed-only TrackCraft relation feature
```

三组使用相同训练/测试划分、相同未来伪标签和相同关系头。若 DA3 版本仍不优于 baseline，说明当前输出端关系头和监督目标本身存在问题的证据增强；若 DA3 版本改善而 TrackCraft 版本恶化，主要问题位于双 TrackCraft 坐标链，而非点间关系思想本身。
