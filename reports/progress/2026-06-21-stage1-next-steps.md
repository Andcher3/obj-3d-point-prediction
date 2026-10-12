# 明确 UCI HAR 阶段一下一步行动

- 日期：2026-06-21
- 完成内容：整理出理解 UCI HAR 数据集后的下一步行动清单：先读取 9 路 IMU 信号，再做不同活动类别的波形可视化和观察小结。
- 本地产出：
- reports/stage1_uci_har_next_steps.md
- reports/progress_log.md
- reports/progress/2026-06-21-stage1-next-steps.md
- 关键理解/结果：当前最优先的工作不是训练模型，而是建立 IMU 波形直觉，先把 X_train_raw 组织为 7352 x 128 x 9 并可视化各类动作。
- 下一步：创建 src/data/uci_har.py，读取 Inertial Signals 并确认数据形状。
