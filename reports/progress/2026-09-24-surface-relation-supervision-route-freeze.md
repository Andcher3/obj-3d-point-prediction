# E3 监督路线冻结与单上下文关系审计

五个视频的跨上下文绝对 3D pseudo-GT 审计全部未通过。当前不能继续通过调整历史窗口长度放行 Relation Head 监督。

补充完成：

- 将辅助诊断字段更名为 `query_pairwise_spacing_px`，明确其表示 query 点彼此间距，不是 dense track 候选距离；
- 保留五视频审计作为跨上下文三维登记失败证据；
- 在浇水壶单一 16 帧 TrackCraft3R 上下文中构造离线关系目标，未训练 Relation Head。

单上下文审计结果：

- `48/48` query 有效；
- t0 全局匹配中位误差 `0.60 px`，最大误差 `67.70 px`；
- 90 条 2D `k=3` 边；
- 未来 9 步边位移变化中位 `0.028 m`、p90 `0.136 m`；
- 规范化形状变化中位 `0.264`、p90 `1.184`。

该结果只证明可以在单一坐标链中生成关系监督候选，不能证明身份质量或 Relation Head 可训练性。异常 t0 anchor 仍需在正式目标使用前单独处理，当前不启动训练。

结果目录：`experiences_results/surface-relation-e3-single-context-audit-2026-09-24/`。
