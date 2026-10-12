# 重新生成 EMA Future Context 方法页

- 日期：2026-07-07
- 完成内容：基于最新正式方法版本，使用 imagegen 重新生成 PPT 方法介绍部分的 5 张整页幻灯片图片。
- 本地产出：
  - `reports/weekly/2026-07-07-ema-future-context-method-slides/method-01-overview.png`
  - `reports/weekly/2026-07-07-ema-future-context-method-slides/method-02-local-motion-token.png`
  - `reports/weekly/2026-07-07-ema-future-context-method-slides/method-03-history-encoder-current-heads.png`
  - `reports/weekly/2026-07-07-ema-future-context-method-slides/method-04-ema-pretraining.png`
  - `reports/weekly/2026-07-07-ema-future-context-method-slides/method-05-finetune-inference.png`
- 关键理解/结果：新方法页已同步为连续 Local Motion Token、channel-independent causal encoder、EMA Teacher future contextual target、non-causal future query self-attention、`pretrain: L_future only` 与 `finetune: L_sup only` 的正式版本。
- 下一步：将这 5 张图片替换进原 PPT 的方法介绍页，并根据实际汇报时长压缩讲稿。
