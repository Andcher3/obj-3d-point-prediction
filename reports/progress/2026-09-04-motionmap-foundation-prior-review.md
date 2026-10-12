# 2026-09-04 MotionMap 与预训练动作先验重设计审查

- 日期：2026-09-04
- 完成工作：将 2026 年 8 月组会版 FlowForesight 及其全部具体数字标记为非约束草案；对照 MotionForesight、MotionMap、Mix-and-Match stochastic conditioning、MolmoMotion 与 What Happens Next，重新审查单真实未来监督和外部动作先验的作用。
- 本地产物或改动：更新 `AGENTS.md`、`AGENTS_zh.md`、`research_roadmap.md`、两份既有 idea 文档和 `reports/progress_log.md`；新增 `reports/ideas/2026-09-04-motionmap-foundation-prior-redesign.md`。
- 关键结果：MotionMap 提供显式模式图和模式置信度，但其多未来集合、二维空间与 codebook 均由训练数据构造；随机条件化主要防止模型忽略噪声。MotionForesight 已经复用视频先验，因此新候选聚焦于把广域预训练动作支持显式化为连续运动潜变量概率场，再由观察视频进行条件重加权，并由最近帧动态与几何完成模式内轨迹细化。
- 仍未解决：外部模型分数不能直接解释为真实条件概率；单后续监督不能确认所有反事实；先验潜变量到任意物体三维点轨迹的映射、概率归一化、模式提取和校准均需实验验证。
- 下一步：复现 MotionForesight 确定性基线；建立目标数据条件 Flow；设计目标训练集整类留出实验，对比目标数据先验与预训练三维轨迹先验是否产生可测的模式召回增益。
