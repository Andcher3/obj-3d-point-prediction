# F2 伪标签坐标对齐机制说明

- 日期：2026-09-27
- 范围：明确 MotionForesight、观测期 TrackCraft3R 与完整视频 TrackCraft3R 之间的坐标对齐位置和公式。
- 状态：ANALYZED；未运行新实验。

## 实际执行位置

对齐由 `scripts/surface_relation_e3_formal_preflight.py` 的伪标签预处理完成，不是 MotionForesight 模型内部操作。

```text
观测期 TrackCraft3R -> observed_t0
完整视频 TrackCraft3R -> pseudo_t0_raw, pseudo_gt_raw
伪标签预处理 -> pseudo_gt_aligned
MotionForesight 只读取观测输入并输出 P0/H
```

## 对齐公式

脚本使用 48 个对应 query 的 t0 点，拟合一个全局相似变换：

```text
p_aligned = scale * (p_raw @ row_rotation) + translation
```

其中包含统一尺度、统一旋转和统一平移。优化目标是让完整视频 TrackCraft3R 的 `pseudo_t0_raw` 尽量接近观测期 TrackCraft3R 的 `observed_t0`。变换只用 t0 点估计，然后同一个变换被应用到所有未来教师点。

它不是逐点添加：

```text
observed_t0[i] - pseudo_t0_raw[i]
```

也没有为每个 query 单独学习一个平移。

## 残差含义

若两次 TrackCraft3R 只差一个共同平移，全局相似变换可以基本消除该差异。对齐后仍存在的 t0 残差，表示不同点之间还有局部几何、尺度或姿态误差。F2 使用的 `pseudo_gt_aligned` 仍带有这类残差。

## 与 MotionForesight 的关系

MotionForesight 的观测输入和输出均在观测坐标链内处理；它没有使用完整视频 TrackCraft3R 的 t0 或未来轨迹。F2 是将外部教师伪标签对齐后，再与冻结 MotionForesight 的 `P0/H` 配对训练 Relation Head。
