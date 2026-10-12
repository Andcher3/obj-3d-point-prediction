# MotionForesight 视频未来轨迹研究进展

ACTIVE_RESEARCH_SCOPE=VIDEO_MOTION_FORECASTING

本索引只记录当前完全基于视频的研究方向。旧 IMU/HAR/OIS 索引已保存到 `archive/imu/progress_log.pre-2026-09-03.md`。

## 进展索引

- 2026-08-04 [审查 MotionForesight 类多未来方法](progress/2026-08-04-motionforesight-method-review.md)：检查“语义等价类 + 条件 Flow”任务、训练边界和多未来评测问题。
- 2026-08-04 [细化语义条件与点轨迹处理](progress/2026-08-04-motionforesight-points-5-7-solutions.md)：形成开放事件关系、语义条件训练、无序点集与伪轨迹处理候选方案。
- 2026-08-04 [复查 MotionForesight 修订](progress/2026-08-04-motionforesight-revision-recheck.md)：保留语义利用、零权重 token、兼容事件和固定分支边界问题。
- 2026-08-04 [整理组会方法构想](progress/2026-08-04-motionforesight-method-discussion-review.md)：将开放事件图、语义条件 Flow 和多未来评测整理为阶段性讨论版本。
- 2026-08-04 [澄清开放事件词表](progress/2026-08-04-event-vocabulary-problem-explanation.md)：固定字段结构，保留开放事件内容，区分等价、蕴含、兼容和互斥关系。
- 2026-08-04 [审查 VLM 联合枚举未来](progress/2026-08-04-joint-vlm-future-enumeration.md)：确认同一回答中的候选频数不能解释为概率。
- 2026-08-04 [设计批量事件假设协议](progress/2026-08-04-batched-event-hypothesis-generation.md)：提出多次独立调用、每次少量候选的生成方案。
- 2026-08-04 [端到端审查语义条件 Flow](progress/2026-08-04-motionforesight-end-to-end-method-audit.md)：统一任务、候选语义、轨迹监督、条件 Flow 和多样本推理边界。
- 2026-08-04 [推演非文本未来潜变量](progress/2026-08-04-nontext-future-latent-redesign.md)：将文本事件替换为从未来视频学习的连续交互潜变量候选。
- 2026-08-05 [筛选相关论文并生成方法图](progress/2026-08-05-motion-forecast-literature-method-figure.md)：整理 MotionForesight、Generative Point Tracking and Forecasting、MYRIAD，并生成无显式语义的方法总览图。
- 2026-08-05 [澄清未来样本数量](progress/2026-08-05-variable-future-sample-count.md)：确认样本数是可变推理预算 `S`，固定六路仅为展示示例。
- 2026-08-05 [完成运动预测论文介绍页](progress/2026-08-05-motion-forecast-paper-slide-images.md)：完成两篇论文的 10 张介绍页和发言稿。
- 2026-09-02 [复盘项目现状与 FlowForesight 构想](progress/2026-09-02-project-status-flowforesight-review.md)：确认当前尚处于方法设计和汇报材料阶段。
- 2026-09-03 [固定视频预测范围并归档 IMU 工作](progress/2026-09-03-motionforesight-scope-reset.md)：将当前范围固定为视频输入的指定物体未来三维轨迹分布预测，归档旧 IMU 工作，并记录连续概率和语义—物理分层两个核心创新目标。
