# 完成 SensorLLM 论文第一轮阅读总结

- 日期：2026-06-24
- 完成内容：阅读并总结 SensorLLM: Aligning Large Language Models with Motion Sensors for Human Activity Recognition，梳理其两阶段框架、UCI-HAR 等数据集形式、sensor-language QA 生成、Chronos/MLP/LLaMA 数据流、训练与推理流程、实验结果和局限。
- 本地产出：
- reports/literature/2026-06-24-sensorllm-reading-note.md
- reports/progress_log.md
- reports/progress/2026-06-24-sensorllm-reading.md
- 关键理解/结果：SensorLLM 说明通用 LLM 不能直接可靠理解 IMU 数字，关键是先用趋势/摘要 QA 做 sensor-language alignment；在 UCI-HAR 上 SensorLLM F1-macro 为 91.2，略低于 Attend 的 93.2，但去掉 alignment 后性能明显下降，说明对齐阶段是核心贡献。
- 下一步：继续阅读 EgoVideoIMU-MAE，比较 video-IMU masked autoencoder 是否更贴近方向一的手机视频与 IMU 联合建模。
