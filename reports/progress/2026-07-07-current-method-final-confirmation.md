# 确认当前主方法版本

- 日期：2026-07-07
- 完成内容：确认当前主方法的核心结构、训练目标、推理读数方式和暂不纳入的模块。
- 本地产出：
  - `reports/progress/2026-07-07-current-method-final-confirmation.md`
- 关键理解/结果：
  - 主方法采用连续 Local Motion Unit / Local Motion Token，不使用 VQ/codebook 或 discrete primitive index。
  - 无标签训练使用 future local motion representation alignment，并对真实未来表示使用 stop-gradient target。
  - phase 是有监督微调输出，标签来源暂不放入模型结构说明。
  - 输出序列统一为 `t:t+n`，切换时机定义为第一个不同于当前标签的位置偏移 `tau`。
  - camera motion 采用 dominant-motion single-label classification。
  - human-camera coupling 暂不进入主路径。
  - smoothing 只用于趋势展示、切换时机读取和序列展示；当前状态识别使用 `t` 时刻 raw head output。
- 下一步：将该干净数据流同步到后续 PPT、发言稿和方法图中。
