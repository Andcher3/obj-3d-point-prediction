# 澄清 SensorLM 中弱语义 caption 与标准 HAR 标签的关系

- 日期：2026-06-23
- 完成内容：围绕用户对 SensorLM caption 如何帮助 HAR 的疑问，区分弱语义 caption、事件级活动记录和标准窗口级 HAR 标签，并更新本地 SensorLM 阅读笔记。
- 本地产出：
- reports/literature/2026-06-23-sensorlm-reading-note.md
- reports/progress_log.md
- reports/progress/2026-06-23-sensorlm-caption-vs-har-label.md
- 关键理解/结果：SensorLM 的 caption 包含 Outdoor Bike、Walk、Sleep 等动作语义，但它是大规模弱标注、事件级、模板化文本，不等同于密集精确的 HAR 标签；其作用是预训练 sensor encoder 学到语义化表示，后续 HAR 仍需 prompt matching 或少量标准标签训练分类头。
- 下一步：继续阅读 SensorLLM，观察其是否更接近短窗口 IMU HAR 的标准分类设定。
