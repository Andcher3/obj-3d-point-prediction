# 总结 Future Patch Label Sequence 的模型结构与训练策略

- 日期：2026-06-30
- 完成内容：围绕 Future Patch Label Sequence 方案，整理模型结构和训练策略，包括 patch embedding、temporal encoder、future label decoder、监督式训练、自监督预训练、样本构造、评价指标和实验顺序。
- 本地产出：
  - `reports/ideas/2026-06-30-future-label-sequence-model-training.md`
  - `reports/progress/2026-06-30-future-label-sequence-model-training.md`
- 关键理解/结果：第一版应采用 `IMU patches -> patch embedding -> TCN/Transformer encoder -> future label decoder` 的轻量结构；训练上先做监督式 future label prediction，再加入 masked reconstruction 等自监督预训练；结构消融应比较 early fusion、P2LHAP-style channel independent 和 hybrid cross-channel fusion。
- 下一步：选择具体连续 HAR 数据集或实验协议，开始实现最小监督 baseline，并保留后续接入自监督预训练的接口。

