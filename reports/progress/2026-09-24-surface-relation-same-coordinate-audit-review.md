# 同坐标监督链与 head 对照审计复核

- 日期：2026-09-24
- 状态：坐标接口检查通过；真实监督链未恢复；edge head 优势未成立

## 同坐标审计的正确解释

源码检查确认 `SparseTrackDataset` 的设计会把 `Pj`、t0 anchor 和 GT tracks 一起变换到最后观测相机，并使用观测 `Pj` 的共享归一化参数。五个缓存也满足形状、有限值、时间槽位和 validity 结构检查。

这些结果只证明接口格式和预期变换规则一致。缓存中的 `pseudo_gt_source` 仍为 `trackcraft3r_offline_future`，且没有找到原始 `*_sparse.npz`。因此尚不能证明缓存是由 MotionForesight 原始训练标签链生成，也不能把 `status=PASS` 解释为监督可用。

五个缓存均应继续标记为 `relation_loss_ready=false`。

## edge/node 对照解释

- edge head 与 node residual 的参数量接近；两者均能拟合单样本 oracle 目标。
- edge 的损失比例中位数较低，但绝对最终损失中位数为 `0.00209`，略高于 node 的 `0.00203`。
- 测试中的 `H` 和目标均为随机单样本，目标不依赖真实模型表示；结果主要反映优化和参数化差异，不反映真实关系信号或泛化能力。

因此不能选择 edge head 作为已验证主方法，也不能据此淘汰 node residual。

## 决策

1. 保持真实视频 Relation Head 训练暂停。
2. 将源码审计和缓存审计记录为“同坐标接口已明确、原始监督文件缺失”。
3. 优先查找或重建官方 `*_sparse.npz` 生成链；若原始标签无法获得，应停止继续扩大 TrackCraft3R 伪标签路线。
4. 获得可靠同坐标标签后，先用固定训练/验证划分比较 decoded-coordinate relation loss、edge head 和 node residual；在此之前不冻结结构选择。
