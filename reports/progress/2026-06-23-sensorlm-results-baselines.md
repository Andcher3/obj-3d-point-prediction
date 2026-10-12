# 补充 SensorLM 结果表现与基线对比

- 日期：2026-06-23
- 完成内容：整理 SensorLM 在 zero-shot activity recognition、few-shot learning、cross-modal retrieval、caption generation、caption 消融和 CLIP/Cap/CoCa 架构消融中的主要结果，并与 Gemini/Gemma、SimCLR、DINO、MSN 等基线进行对比。
- 本地产出：
- reports/literature/2026-06-23-sensorlm-reading-note.md
- reports/progress_log.md
- reports/progress/2026-06-23-sensorlm-results-baselines.md
- 关键理解/结果：SensorLM 在 zero-shot 20 类活动识别上 AUROC 0.84，显著优于直接用 Gemini/Gemma 看 sensor table；few-shot Activity 上 50 labels/class 达 AUROC 0.88，明显优于 SimCLR/DINO/MSN；但论文没有与 1D CNN/LSTM 等标准监督 HAR 分类器在 UCI HAR 式 benchmark 上直接比较，因此其优势应表述为少标签、零样本和语义对齐能力，而非封闭集监督分类准确率。
- 下一步：继续阅读 SensorLLM，观察其在短窗口 motion sensor HAR 上是否提供更直接的标准分类对比。
