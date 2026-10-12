# 2026-08-04 MotionForesight 方法第 5、6、7 点解决方案

- 日期：2026-08-04
- 完成工作：对照 Semantic Entropy、Logical Graph Uncertainty、EgoMAN、MolmoMotion、Classifier-Free Guidance、Set Transformer、TrackCraft3R 和 MotionForesight，细化语义聚类、语义条件 Flow 与点轨迹有效性处理方案。
- 本地产物或改动：新增本记录，并更新 `reports/progress_log.md`。
- 关键理解：语义候选应先规范化为开放词汇事件记录，再构建等价、蕴含和互斥关系图；Flow 采用教师语义预训练、预测语义对齐、互斥语义排序损失和条件 dropout；未来点采用时间编码与参考坐标身份编码，移除任意点序号 RoPE，并在进入 Flow 前过滤、补全和置信加权伪轨迹。
- 下一步：按该版本固定最小实现，先验证语义类纯度、条件交换敏感性、点排列等变性和有效轨迹覆盖率。
