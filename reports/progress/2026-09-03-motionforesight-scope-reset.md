# 2026-09-03 MotionForesight 研究范围确认与 IMU 归档

- 日期：2026-09-03
- 完成工作：依据用户最新确认，将活动研究范围固定为完全基于视频输入的指定物体未来三维轨迹预测；归档旧 IMU/HAR/OIS 数据、代码、文献和组会材料；重写当前路线图和协作说明；整理连续概率多未来与宏观语义—局部物理分层两个创新目标及困难。
- 本地产物或改动：`AGENTS.md`、`AGENTS_zh.md`、`research_roadmap.md`、`reports/ideas/2026-09-03-motionforesight-probabilistic-semantic-design.md`、`reports/progress_log.md`、`archive/imu/README.md`。
- 关键结果：当前唯一活动方向为 `p(D_future | V_obs, M_target)`。MotionForesight 的三维点轨迹输出形式被保留；研究重点是连续概率意义的多未来分布，以及由长程高层语义控制宏观模式、最近帧动态与几何控制局部物理细节。两项创新仍有单未来监督、概率校准、语义条件利用和多尺度耦合等未解决问题。
- 下一步：先定义连续概率输出的可观测量和评测方式，再固定全局语义分支、局部动态分支与轨迹生成器的最小接口，并准备 MotionForesight 单未来复现协议。
