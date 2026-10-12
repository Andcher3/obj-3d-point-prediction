# 固定 Teacher Target 与 Future Query 设计

- 日期：2026-07-07
- 完成内容：确认当前正式方法中的 Teacher target 取值层级和 Future Query Decoder 的 self-attention mask 设定。
- 本地产出：
  - `reports/progress/2026-07-07-method-final-target-query-confirmation.md`
- 关键理解/结果：
  - Teacher target 固定取 channel fusion 后最终层 contextual states `G^T_{t+1:t+n}`。
  - 当前不采用多层 hidden-state averaging，也不使用 data2vec 式 multi-layer target aggregation。
  - Future queries 的 self-attention 固定为 non-causal，允许固定 horizon latent slots 之间互相交互。
  - Future predictor 的信息约束来自 cross-attention memory，仅允许访问历史 `G^S_{1:t}`。
  - 当前方法结构可以视为正式固定版本。
- 下一步：后续方法图、PPT 和代码设计文档按该版本保持一致。
