# 检索动作切换时机与下一动作预测的相关任务

- 日期：2026-06-29
- 完成内容：围绕用户对“预测动作切换时机和下一个动作是否是强行实现”的疑问，检索并整理近年相关任务形态，包括 sensor HAR forecasting、long-term action anticipation、future action segmentation、segment-level duration prediction 和 time-to-action regression。
- 本地产出：
  - `reports/literature/2026-06-29-switch-timing-next-action-tasks.md`
  - `reports/progress/2026-06-29-switch-timing-next-action-tasks.md`
- 关键理解/结果：相关领域通常不把 `sit-to-stand` 作为与 `sit`、`stand` 平级的主类，而是预测未来 patch-level label sequence、未来 action segment + duration，或 next action + time-to-action。对方向一最自然的落地形式是先采用 P2LHAP-style future label sequence，再从第一次标签变化派生切换时机和下一动作。
- 下一步：若继续推进模型设计，应在 `future label sequence`、`segment-level action + duration` 和 `next action + time-to-switch` 三种输出形式中选择一个最小可实现版本。

