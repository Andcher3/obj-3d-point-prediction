# 整理 LLM 文本预测与 IMU 时间序列预测的联系

- 日期：2026-06-28
- 完成内容：围绕“已有 IMU 序列预测未来 IMU 序列是否类似 LLM 文本预测”，检索并整理 Chronos、Lag-Llama、Time-LLM、GPT4TS 和 SensorLLM 等相关方向。
- 本地产出：
  - `reports/literature/2026-06-28-llm-time-series-imu-forecasting.md`
  - `reports/progress/2026-06-28-llm-time-series-imu-forecasting.md`
- 关键理解/结果：时间序列预测与语言模型预测的联系主要有三层：共享 Transformer/自回归架构、把连续数值量化成离散 token 后像语言一样预测、把时间序列 patch 映射到 LLM embedding 空间进行 reprogramming。对手机 IMU 来说，连续回归仍是最稳 baseline，Chronos/Lag-Llama/Time-LLM 代表了可进一步探索的 foundation model 路线。
- 下一步：若进入方向二或未来 IMU 预测实验，先实现 TCN/Transformer 连续回归 baseline，再评估是否引入 Chronos/Lag-Llama 类时间序列 foundation model。
