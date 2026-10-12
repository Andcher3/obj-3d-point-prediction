# 核查 P2LHAP 外部总结

- 日期：2026-06-28
- 完成内容：检查用户保存的 P2LHAP 外部 AI 总结文本和图像材料，并对照论文原文核查主线、方法、输出形式和实验结论。
- 本地产出：
  - `reports/literature/2026-06-28-p2lhap-summary-check.md`
  - `reports/progress/2026-06-28-p2lhap-summary-check.md`
- 关键理解/结果：外部总结总体主线正确，准确抓住 HAS/HAR/HAF 统一、Patch-to-Label Seq2Seq、channel-independent Transformer 和 smoothing strategy。但需纠正几处表述：P2LHAP 仍有固定 patch size；“单任务模型”应理解为统一标签序列输出而非完全单一普通分类任务；forecast 实验不只涉及未来标签，也涉及未来传感器信号预测；实验结论不能脱离 WISDM、PAMAP2、UNIMIB SHAR 和对应评价协议泛化。
- 下一步：如果继续精读 P2LHAP，应重点记录其 patch 构造、forecast horizon、loss 设计、smoothing 公式和评价指标，以判断如何迁移到方向一的人体动作 + 手机运动双标签序列预测。
