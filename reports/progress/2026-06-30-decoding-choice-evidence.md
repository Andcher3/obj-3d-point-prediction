# 整理 Future Label Sequence Decoder 选择的相关方法依据

- 日期：2026-06-30
- 完成内容：根据用户提醒，补充查证相近新任务中对 future label/action sequence 的 decoder 选择，避免仅从工程可行性闭门设计。
- 本地产出：
  - `reports/literature/2026-06-30-decoding-choice-evidence.md`
  - `reports/progress/2026-06-30-decoding-choice-evidence.md`
- 关键理解/结果：P2LHAP 直接输出未来 patch-level label sequence；FUTR 明确比较自回归与并行 decoder，发现 parallel decoding 更准确、更快且避免错误累积；ActFusion 用 anticipative masking 和 learnable mask tokens 补全未来动作序列。这些相关工作共同支持当前第一版采用非自回归 query Transformer decoder，而将自回归 decoder 作为消融或后续版本。
- 下一步：将方法草案中的 decoder 选择明确写为“以 parallel query decoder 为主，自回归 decoder 作为对照消融”，并在实验设计中保留 decoder 消融。

