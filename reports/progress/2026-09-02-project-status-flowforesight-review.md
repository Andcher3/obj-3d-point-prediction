# 2026-09-02 项目现状与 FlowForesight 构想复盘

- 日期：2026-09-02
- 完成工作：检查项目路线图、进展索引、UCI HAR 处理产物、IMU 方法演化记录、MotionForesight 方法审查、FlowForesight 图稿与发言稿，形成当前项目和研究构想快照。
- 本地产物或改动：`reports/ideas/2026-09-02-project-status-flowforesight.md`；本进展记录；`reports/progress_log.md`。
- 关键结果：当前最新方法候选是基于 MotionForesight 任务接口、使用 `C_point + C_scene` 条件和 Flow Matching 产生可变 `S` 组未来三维点轨迹的 FlowForesight。显式 VLM 事件语义与非文本潜变量均讨论过，但 2026-08-05/06 的最新主干暂不包含显式语义。现阶段停留在任务与方法设计、论文汇报材料阶段，尚无模型复现、训练代码或实验结果。
- 下一步：先固定最小实验协议并复现单未来基线，再实现无显式语义的多未来 Flow 基线；随后把语义条件作为独立消融变量测试。
