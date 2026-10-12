# 整理动作切换时机与下一动作预测的相关任务

- 日期：2026-06-30
- 完成内容：围绕“是否需要强行设计多种 transition 标签来预测切换时机”的疑问，检索并整理了相关任务设定，包括 sensor-based activity forecasting、long-term action anticipation、temporal action segmentation + anticipation，以及 temporal action localization / online temporal action localization。
- 本地产出：
  - `reports/literature/2026-06-30-switch-timing-next-action-tasks.md`
  - `reports/progress/2026-06-30-switch-timing-next-action-tasks.md`
- 关键理解/结果：较新的任务通常不把每种动作切换设成独立主类别，而是预测未来标签序列或未来动作段的类别与持续时间，再从中推导切换时机和下一动作。对方向一而言，P2LHAP 风格的 future patch label sequence 是近期更稳的落地形式；FUTR 风格的 label + duration segment decoder 是更优雅但要求更高的进阶形式。
- 下一步：优先把方向一输出定义为 `past IMU patches -> future human/phone-motion label sequence`，再讨论是否进一步升级到 segment-level label + duration。

