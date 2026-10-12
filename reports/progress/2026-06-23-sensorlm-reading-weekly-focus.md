# 完成 SensorLM 论文阅读总结并压缩组会论文数量

- 日期：2026-06-23
- 完成内容：精读 SensorLM: Learning the Language of Wearable Sensors，整理其问题定位、层级 caption 构造、sensor-language foundation model 结构、实验结果、对方向一的启发和局限；同时将组会草稿压缩为主讲 2 篇、最多补充 1 篇。
- 本地产出：
- reports/literature/2026-06-23-sensorlm-reading-note.md
- reports/weekly/2026-06-24-group-meeting-draft.md
- reports/progress_log.md
- reports/progress/2026-06-23-sensorlm-reading-weekly-focus.md
- 关键理解/结果：SensorLM 表明，直接把传感器表格交给通用 LLM 并不可靠，关键是构造 statistical/structural/semantic caption 并进行 sensor-language alignment；对方向一的启发是手机 IMU 可进一步学习与人体动作、手机运镜和场景语义对齐的表示。
- 下一步：继续精读 SensorLLM，比较其短窗口 motion sensor 与 LLM 对齐方式是否比 SensorLM 更贴近 UCI HAR 和方向一。
