# 确认 EMA Future Context 版方法更新

- 日期：2026-07-07
- 完成内容：读取并确认最新方法说明，识别出当前版本相对上一版的结构变化。
- 本地产出：
  - `reports/progress/2026-07-07-ema-future-context-method-update.md`
- 关键理解/结果：
  - 未来预测分支从“预测 future local motion tokens 并 causal re-encode”改为“通过 parallel future queries 直接预测 future contextual states”。
  - 无标签训练目标从 local token-level alignment 改为 EMA Teacher-Student future contextual state distillation。
  - Teacher 看真实完整 `IMU_{1:t+n}`，用 causal encoder 产生 `g^T_{t+1:t+n}`；Student 只看 `IMU_{1:t}`，预测 `hat g_{t+1:t+n}`。
  - 有监督微调阶段丢弃 EMA Teacher，当前 `g_t` 和未来 `hat g_{t+1:t+n}` 共用四个 prediction heads。
  - 当前状态使用 raw head output，趋势与切换时机仍从平滑后的 label sequence 读取。
- 下一步：确认 future predictor 输出空间、target normalization、是否在微调阶段保留无标签辅助损失，以及文献归因是否需要逐篇核对。
