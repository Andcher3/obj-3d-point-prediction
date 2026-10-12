# MolmoMotion ViPE+AllTracker 实验归档

- 归档日期：2026-09-22
- 状态：已完成、已归档
- 原始结果：`experiences_results/MolmoMotion-vipe-alltracker-2026-09-22/`
- 分析报告：`report.md`
- 进度记录：`progress.md`
- 运行器：`run_molmomotion_vipe_alltracker_probe.py`

归档内容保留了 ViPE 深度、位姿和内参、AllTracker 2D 轨迹、3D lift 结果、MolmoMotion 四条指令的数组与文本输出、可视化视频/图片及运行日志。

本轮使用官方 ViPE+AllTracker 几何与跟踪组件，手工固定 8 个查询点并关闭 VDA/SAM。四条指令输出完全相同，预测段内部恒定，结论限定为该组件替换实验，不代表完整官方数据生成管线的端到端复现。

原始 `experiences_results` 目录保留，归档目录为独立副本。
