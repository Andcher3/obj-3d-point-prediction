# 修正 SensorLM Figure 3 架构理解并补充 patch 与 caption 来源分析

- 日期：2026-06-23
- 完成内容：根据 SensorLM Figure 3 和正文/附录重新确认 text encoder、sensor encoder、multimodal text decoder 的数据流，修正此前对 captioning 分支的简化表述，并补充 2D sensor patch 的合理性与局限、semantic caption 的事件来源。
- 本地产出：
- reports/literature/2026-06-23-sensorlm-reading-note.md
- reports/literature/sensorlm_fig3_page6.png
- reports/progress_log.md
- reports/progress/2026-06-23-sensorlm-figure3-patch-caption-correction.md
- 关键理解/结果：Figure 3 表明 captioning 分支中 text 也先经过 Text Encoder，再与 Sensor Encoder 输出一起进入 Multimodal Text Decoder；2D patch 依赖 feature 排列顺序，是工程归纳偏置而非天然空间局部性；semantic captions 来自 recognized activities、sleep periods 和 user-logged mood 等已有事件信息的模板化生成，属于大规模弱标注文本。
- 下一步：继续阅读 SensorLLM，关注其是否更清晰地处理短窗口 IMU 与语言模型的对齐问题。
