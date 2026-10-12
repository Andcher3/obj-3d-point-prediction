# 检索未来延续概率与切换趋势预测相关工作

- 日期：2026-06-28
- 完成内容：围绕方向一输出中的“未来延续概率”和“切换趋势预测”，检索并整理相关工作类型，包括 transition-aware HAR、postural transition recognition、activity segmentation/forecast、duration modeling/HSMM 和 future IMU forecasting。
- 本地产出：
  - `reports/literature/2026-06-28-continuation-switch-related-work.md`
  - `reports/progress/2026-06-28-continuation-switch-related-work.md`
- 关键理解/结果：HAR 文献中较少直接使用“未来延续概率”这个名称，但已有相近任务：过渡动作识别把切换当成类别，segmentation/forecast 预测未来标签序列，HSMM 显式建模状态持续时间，future IMU forecasting 可间接预测未来动作。对当前方向一，建议先将输出定义为当前动作分类、当前手机运动分类、未来固定 horizon 内的 continue/switch 二分类，以及下一状态分布。
- 下一步：进一步阅读 P2LHAP 等 activity forecast/segmentation 工作，确定是否采用 patch-level future label sequence 作为比单一切换概率更系统的输出形式。
