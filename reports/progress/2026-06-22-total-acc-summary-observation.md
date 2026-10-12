# 解释 total_acc 特征并生成统计汇总图

- 日期：2026-06-22
- 完成内容：根据 total_acc 曲线和全训练集统计，解释为什么肉眼难以区分部分动作，并生成 total_acc_summary.svg 展示各类别三轴均值和窗口内波动强度。
- 本地产出：
- reports/figures/uci_har/total_acc_summary.svg
- scripts/summarize_uci_har_total_acc.py
- scripts/plot_uci_har_total_acc_summary.py
- reports/uci_har_observations.md
- reports/uci_har_visualization_guide.md
- reports/progress_log.md
- reports/progress/2026-06-22-total-acc-summary-observation.md
- 关键理解/结果：total_acc 的类别差异主要体现在重力方向对应的三轴均值和窗口内波动强度，而不是肉眼曲线形状；动态/静态容易区分，躺下较容易通过重力方向区分，上下楼和普通走路仍需更细特征。
- 下一步：从看曲线进入提取窗口级统计特征，如均值、标准差、能量、峰值和频率特征，为传统 HAR baseline 做准备。
