# 补充 SensorLM 核心方法细节解释

- 日期：2026-06-23
- 完成内容：围绕 SensorLM 的 caption 输出含义、contrastive/generative objectives、CLIP-like/Cap-like/CoCa-like 变体、encoder/decoder 结构和完整数据流进行细化解释，并更新本地阅读笔记。
- 本地产出：
- reports/literature/2026-06-23-sensorlm-reading-note.md
- reports/progress_log.md
- reports/progress/2026-06-23-sensorlm-method-details.md
- 关键理解/结果：SensorLM 应理解为两条训练流：contrastive branch 用 sensor encoder 和 text encoder 对齐 embedding；generative branch 用 sensor encoder 和 causally masked multimodal decoder 自回归生成 caption。主实验使用 structural+semantic caption，并以 CoCa-like 的 contrastive+generative 联合目标取得更好折中。
- 下一步：继续阅读 SensorLLM，比较其 sensor-language alignment 是否更贴近短窗口 IMU HAR 和方向一任务。
