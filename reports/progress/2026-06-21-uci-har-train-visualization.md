# 整理 UCI HAR 训练集并生成 IMU 曲线图

- 日期：2026-06-21
- 完成内容：使用 conda 环境 HAR 将 UCI HAR 训练集整理为 7352 x 128 x 9 的统一时序布局，并为 6 类动作各生成 4 个样本的 body_acc/body_gyro SVG 曲线图。
- 本地产出：
- data/processed/uci_har/X_train_raw_float32.bin
- data/processed/uci_har/shape.json
- data/processed/uci_har/train_metadata.csv
- reports/figures/uci_har/index.html
- reports/figures/uci_har/*.svg
- reports/uci_har_visualization_guide.md
- src/data/uci_har.py
- scripts/plot_uci_har_samples.py
- 关键理解/结果：训练集二进制大小与 7352*128*9*float32 完全匹配；可视化图将每类动作按样本行展示，左侧为 body_acc 三轴，右侧为 body_gyro 三轴，便于观察动态与静态动作差异。
- 下一步：先打开 reports/figures/uci_har/index.html 观察 6 类动作波形，再补充 total_acc_x/y/z 图以观察坐、站、躺的重力方向差异。
