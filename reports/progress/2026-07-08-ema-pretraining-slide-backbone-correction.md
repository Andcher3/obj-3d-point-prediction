# EMA pretraining slide backbone correction

- 日期：2026-07-08
- 完成内容：Regenerated the unlabeled pretraining slide with Teacher Backbone shown as an EMA copy of the Student Backbone, including Local Motion Tokenizer, Shared Causal Encoder, and Channel Fusion.
- 本地产出：
- reports/weekly/2026-07-07-ema-future-context-method-slides/method-04-ema-pretraining-v2.png
- reports/progress/2026-07-08-ema-pretraining-slide-backbone-correction.md
- reports/progress_log.md
- 关键理解/结果：The EMA update now applies only to backbone parameters and no longer implies that the Parallel Future Context Predictor is part of the teacher.
- 下一步：Use method-04-ema-pretraining-v2.png when replacing the unlabeled pretraining page in the deck.
