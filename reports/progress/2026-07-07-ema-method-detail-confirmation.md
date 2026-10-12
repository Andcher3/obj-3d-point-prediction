# 确认 EMA Future Context 方法细节

- 日期：2026-07-07
- 完成内容：确认 Future Context Predictor 输出空间、target normalization、预训练/微调损失协议和文献归因措辞。
- 本地产出：
  - `reports/progress/2026-07-07-ema-method-detail-confirmation.md`
- 关键理解/结果：
  - Future Predictor 最终输出固定为 `D_f` 维 contextual state，与 teacher `g^T_{t+k}` 对齐。
  - Target normalization 固定为无可学习参数的 per-state LayerNorm，即 `LayerNorm(elementwise_affine=False)`，不使用 L2 normalization。
  - 训练协议固定为 `pretrain: L_future only`，`finetune: L_sup only`；微调阶段丢弃 EMA Teacher，不继续维护 future alignment loss。
  - 文献归因中 data2vec-style EMA teacher contextual target prediction 可以明确说明，其他模块在逐篇核验前使用 `inspired by`。
- 下一步：在方法图、PPT 和代码设计文档中保持该版本表述一致。
