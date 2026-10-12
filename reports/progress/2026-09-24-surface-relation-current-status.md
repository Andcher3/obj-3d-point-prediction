# 点间关系方案当前实验状态

- 日期：2026-09-24
- 状态：工程可行性已完成；真实视频监督路线因标签来源和坐标一致性问题暂停

## 当前进度

E0、E1 已完成，E2-G/E2-G.1 已完成，MotionForesight observed-only `P0/H` 接口审计已完成，E3-controlled 工程测试已完成。当前没有生成真实视频 Relation Head 训练缓存。

## 当前阻塞实验及目的

阻塞实验是按 MotionForesight 官方 `T=22`、`7+15` 协议获取完整 TrackCraft3R dense 轨迹，并验证其能否与 observed-only `P0/H` 在同一坐标链上配对。该实验的目的不是测试 Relation Head 性能，而是确认训练监督是否存在：完整未来 15 步标签、正确的 reference frame、坐标/归一化一致性、validity 和无未来信息接口。

目前只生成了 22 帧 DA3 输入；TrackCraft3R 两次均未生成完整 dense map，因此监督 gate 未打开。

## 假设状态

保留：同一物体点之间存在可利用的相关运动；时变软关系可能减少点群散开；constrained-L2 图积分数学可用。

当前方案不需要固定硬刚体边长假设；该假设已被排除，不能列为本方法的必要前提。仍未成立的是：跨 DA3/TrackCraft 上下文可以直接登记绝对 3D pseudo-GT；当前 TrackCraft3R/DA3 结果可直接监督真实 Relation Head。

尚未验证：`H_output` 是否包含关系信息；Relation Head 是否优于 node residual；2D `k=3` 是否具有语义邻接优势；关系修正是否改善真实预测。

## TrackCraft3R 获取路线修正

此前 22 帧失败集中发生在全图 dense `track_map` 获取，不能直接推出 query-only TrackCraft3R 也失败。后续应沿用既有成功调用路径，优先生成 22 帧、48-query 的 query tracks，并记录 t0 到 frame-0 的 anchor 映射。当前代码任务尚未产生新的 query-only 输出文件，因此这一修正仍处于待执行状态。

## 可行性结论

方法的数学和工程接口可行，真实学习可行性尚未建立。当前最大风险位于监督数据链，而非图积分或 Relation Head 代码。现阶段应将方法表述为“观测图上的时变软关系修正候选”，不能称为已验证的关系增强或刚体保持方法。
