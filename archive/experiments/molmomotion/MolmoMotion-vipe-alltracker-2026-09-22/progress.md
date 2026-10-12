# MolmoMotion 官方 ViPE+AllTracker A/B 实验

- 日期：2026-09-22
- 完成工作：在 GPU 服务器上运行官方 ViPE、官方 AllTracker、官方 3D lift，并将其输出接入 `MolmoMotion-4B-H3-F30`；固定同一视频、8 个点和四条指令完成 greedy 对照。
- 输出：`reports/analysis/2026-09-22-molmomotion-vipe-alltracker-test/report.md`；`experiences_results/MolmoMotion-vipe-alltracker-2026-09-22/`；`scripts/run_molmomotion_vipe_alltracker_probe.py`。
- 关键结果：ViPE 25 帧和 AllTracker `(25,8,2)` 均成功，8 点全程可见；四条文本输出完全相同，预测段内部恒定，路径长度约 `0.000832 m`。DA3+TrackCraft3R 对照为 `0.000420 m`，同样近静止。
- 研究判断：坐标、尺度和跟踪分布差异不是该样本近静止和文本不敏感的唯一主因；结论限定于 ViPE+AllTracker 几何/跟踪组件，因为本轮使用手工查询点并关闭 VDA/SAM。
- 下一步：将该结果与动作后移窗口、不同 H3 历史截点和公开非饮料瓶样本一起整理，避免把“早期三帧信息不足”和“checkpoint 条件退化”混为单一原因。
