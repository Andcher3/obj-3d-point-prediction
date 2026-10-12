# 整理 IMU/HAR 深度学习方法路线

- 日期：2026-06-22
- 完成内容：将 UCI HAR 的信号观察过渡到深度学习方法路线，整理 1D CNN、LSTM/GRU、DeepConvLSTM、TCN、InceptionTime、Transformer/PatchTST 和自监督学习在 IMU/HAR 与两个课题中的作用。
- 本地产出：
- reports/imu_deep_learning_method_map.md
- reports/progress_log.md
- reports/progress/2026-06-22-imu-deep-learning-method-map.md
- 关键理解/结果：下一步最合理的深度学习闭环是先做 1D CNN baseline；TCN/DeepConvLSTM 作为后续强基线；Transformer/PatchTST 和自监督学习作为进一步研究拓展。
- 下一步：选择可用深度学习环境，优先跑通 9 通道 IMU 输入的 1D CNN 六分类 baseline。
