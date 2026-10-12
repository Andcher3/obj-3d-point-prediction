# 解释 SensorLM 自监督基线和评估指标含义

- 日期：2026-06-24
- 完成内容：围绕 SensorLM 论文中的 SimCLR、DINO、MSN 自监督基线、zero-shot activity recognition 指标、few-shot labels per class、sensor-text retrieval 和 CLIP-like/CoCa-like 结果差异进行概念解释，并更新本地阅读笔记。
- 本地产出：
- reports/literature/2026-06-23-sensorlm-reading-note.md
- reports/progress_log.md
- reports/progress/2026-06-24-sensorlm-metrics-and-ssl-baselines.md
- 关键理解/结果：SimCLR/DINO/MSN 通过增强视图、教师学生一致性或 masked view 对齐进行无标签预训练；SensorLM 的 zero-shot 分类主要靠 sensor-text embedding 相似度，few-shot 是冻结 encoder 后用每类少量标签训练线性头，retrieval 是跨模态最近邻检索；CLIP-like 与 CoCa-like 在 zero-shot 上接近，因为分类主要依赖 contrastive alignment。
- 下一步：继续阅读 SensorLLM，比较其短窗口 motion sensor 对齐方式和指标是否更贴近标准 HAR。
