# 澄清 HAR 自监督目标、因果注意力、LLM 语义趋势和标注困难

- 日期：2026-06-23
- 完成内容：围绕用户对 masked reconstruction、future prediction、全局/因果注意力、LLM/多模态语义推理和 HAR 标注困难的疑问进行资料核对与概念整理。
- 本地产出：
- reports/literature/2026-06-23-masked-reconstruction-causal-and-labels.md
- reports/progress_log.md
- reports/progress/2026-06-23-masked-causal-llm-labels.md
- 关键理解/结果：HAR 表示学习可用全局注意力和 masked reconstruction；OIS 短时预测必须使用因果建模和 future regression/event prediction。LLM/多模态 foundation model 是近年上升趋势，但更适合语义对齐、辅助标注、解释和预训练，不宜直接作为实时控制核心。
- 下一步：在下一轮自监督 HAR 阅读中重点关注 masked reconstruction、forecasting、contrastive learning 这三类预训练目标，并据此设计 UCI HAR 原型实验。
