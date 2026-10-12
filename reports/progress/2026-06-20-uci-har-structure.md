# 理解 UCI HAR 数据集结构

- 日期：2026-06-20
- 完成内容：检查 `HAR/UCI HAR Dataset/` 的目录结构、官方说明文件、标签文件和 Inertial Signals 数据形状。
- 本地产出：
  - 对话中形成的数据结构说明
  - `AGENTS_zh.md` 中的 UCI HAR 数据结构说明
- 关键理解/结果：`X_train/X_test` 是 561 维人工特征表；`Inertial Signals` 是 9 路 IMU 时间序列；每一行代表一个 2.56 秒窗口样本。
- 下一步：读取 9 路 IMU 信号并绘制不同活动类别的波形。
