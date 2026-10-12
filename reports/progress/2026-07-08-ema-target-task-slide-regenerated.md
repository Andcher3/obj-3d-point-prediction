# 重生成目标任务页

- 日期：2026-07-08
- 完成内容：基于 EMA Future Context 正式方法版本，重新生成 `目标任务` 幻灯片页。
- 本地产出：
  - `reports/weekly/2026-07-08-ema-target-task-slide/target-task-ema-context-v2.png`
  - `reports/weekly/2026-07-08-ema-target-task-slide/target-task-ema-context-v3.png`
  - `reports/weekly/2026-07-08-ema-target-task-slide/target-task-ema-context-final.png`
- 关键理解/结果：新版目标任务页移除了旧的 future token / re-encoder 表述，改为 `Local Motion Tokenizer -> Causal Encoder + Channel Fusion -> G^S_{1:t}` 和 `Parallel Future Context Predictor -> hat G_{t+1:t+n}`；右下示例明确拆分 `Φ^H` 与 `Φ^C`，分别对应人体与相机两条流的不同切换过程。
- 下一步：将 `target-task-ema-context-final.png` 替换进当前 PPT 的目标任务页。
