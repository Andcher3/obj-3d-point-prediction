# SensorLM 论文最终总结与学术海报

- 日期：2026-06-27
- 完成内容：对 `SensorLM: Learning the Language of Wearable Sensors` 做阶段性最终总结，并制作一版 4:3 横版学术海报，用于清晰介绍论文的问题定位、数据形态、层级 caption、模型结构、训练目标、实验结果、局限和对方向一的启发。
- 本地产出：
  - `reports/posters/2026-06-27-sensorlm-academic-poster.html`
  - `reports/posters/2026-06-27-sensorlm-academic-poster.pdf`
  - `reports/posters/2026-06-27-sensorlm-academic-poster.png`
- 关键理解/结果：SensorLM 的核心不是直接替代标准 HAR 分类器，而是用 statistical、structural、semantic caption 构造大规模弱语义 sensor-text pair，再通过 contrastive alignment 和 caption generation 联合训练 sensor-language representation。实验中 contrastive alignment 是 zero-shot 分类和检索的关键，semantic caption 是活动识别性能提升的关键来源，CoCa-like 变体在保留识别能力的同时增加生成能力。
- 对当前课题的启发：方向一后续可以借鉴 sensor-language alignment 思路，把手机 IMU 与人体动作、手机运镜、视频事件或场景语义进行对齐；但短期不应直接照搬 SensorLM 到 UCI HAR，因为它的数据形态、标注方式和任务尺度都不同。
- 验证情况：使用系统 Chrome 导出 PDF，使用 Poppler 将 PDF 转为 PNG 预览，并通过浏览器 DOM 检查确认海报主要区块无文本溢出。
- 下一步：继续阅读 SensorLLM，比较其短窗口 motion sensor 与 HAR 设定是否更适合作为本项目近期可落地路线。
