# MotionForesight 与 F2 两次 TrackCraft3R 链路区别

- 日期：2026-09-27
- 状态：ANALYZED

## 结论

用户对“完整视频生成离线伪标签、预测时只使用观测前缀”的理解是对的。需要修正的是：MotionForesight 的观测几何输入是 DA3 估计并经过相机变换的 pointmap，不是先运行一次只含过去帧的 TrackCraft3R 轨迹。

官方链路是：

```text
完整视频 -> TrackCraft3R -> 离线未来伪轨迹标签
观测 RGB + 观测 DA3 pointmap -> MotionForesight 前向
```

观测 pointmap 与完整视频 TrackCraft3R 标签都被表达在最后观测相机坐标中。这个统一的是坐标表达，不保证每个 query 的几何数值完全相同；剩余差异属于伪标签噪声或观测几何误差，由训练数据和可训练适配器共同吸收。

## F2 增加的链路

F2 使用：

```text
观测期 TrackCraft3R -> observed_t0/history/图关系特征
完整视频 TrackCraft3R -> future pseudo-GT
```

然后估计两次 TrackCraft3R 的 t0 全局相似变换，把教师轨迹映射到 observed-only TrackCraft 坐标。该跨上下文 TrackCraft 登记步骤不是 MotionForesight 原始训练接口，因此引入了额外的局部几何残差。

## 关于 t0 残差

若完整教师点与观测点只差一个共同平移，例如：

```text
O0=(10,20,10), T0=(15,20,10), T1=(15,25,10)
```

则应把教师轨迹整体平移 `(-5,0,0)`：

```text
T1_aligned=(10,25,10)
```

这保留了教师的运动位移，不会产生形变。

若每个 query 的 `O0_i-T0_i` 不同，则逐点把该残差加到所有未来帧不是坐标系变换，而是人为的点特定形变修正；它可能改变点间距离，不能直接视为正确的伪标签对齐。F2 使用的是全局相似变换，变换后仍有局部残差。

## 对当前判断的修正

MotionForesight 也可能受到“观测 DA3 几何与完整 TrackCraft 伪标签不完全一致”的影响，但它没有 F2 中“两次不同上下文 TrackCraft3R 之间的额外登记误差”。因此，F2 的负结果不能单独归因于 MotionForesight 同样存在的原始监督误差；它混合了原始观测—标签差异、额外 TrackCraft 跨上下文误差、冻结 backbone、关系头泛化和小样本等因素。
