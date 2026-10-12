# 修正方向一延续/切换输出的相关工作主参考

- 日期：2026-06-28
- 完成内容：根据用户反馈，重新判断“未来延续概率”和“切换趋势预测”的相关工作价值，将 P2LHAP 作为更符合当前任务要求的近期深度学习主参考，而把早期 transition-aware HAR 仅保留为历史/数据集背景。
- 本地产出：
  - `reports/literature/2026-06-28-continuation-switch-related-work.md`
  - `reports/progress/2026-06-28-p2lhap-output-formulation.md`
- 关键理解/结果：P2LHAP 的 patch-to-label Seq2Seq Transformer 直接输出连续 patch-level 活动标签序列，可同时覆盖 recognition、segmentation 和 forecast，因此可以自然派生未来延续概率、切换概率、首次切换位置和下一状态类别。它比单独设计 continue/switch head 更系统，但仍需扩展到人体动作与手机/相机运动双标签或组合状态。
- 下一步：精读 P2LHAP，重点记录其输入 patch 化、encoder-decoder 输出、forecast horizon、评价指标和可迁移到双任务输出的部分。
