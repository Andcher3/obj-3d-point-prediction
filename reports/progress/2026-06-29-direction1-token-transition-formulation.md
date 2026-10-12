# 梳理方向一 IMU token 与切换预测建模草案

- 日期：2026-06-29
- 完成内容：围绕用户提出的“构造 IMU token 序列、用特殊 label 标注切换动作、预测 next token、再做分类、多通道分离”等初步想法，整理为更严谨的方向一任务建模草案。
- 本地产出：
  - `reports/ideas/2026-06-29-direction1-token-transition-formulation.md`
  - `reports/progress/2026-06-29-direction1-token-transition-formulation.md`
- 关键理解/结果：不建议把“切换动作”直接作为普通动作类别；更合适的方式是将连续 IMU 切成 patch/token 序列，输出人体动作标签序列和手机/相机运动标签序列，再由标签序列派生延续概率、切换概率、首次切换位置和下一状态。若 patch 内包含动作边界，应优先使用 soft label + boundary flag；UCI HAR 可用于 tokenizer/encoder 和当前分类 sanity check，但不适合直接验证真实切换预测。
- 下一步：将该草案转化为最小实验路线：先在 UCI HAR 上跑通 patch/tokenizer + Transformer/TCN 分类 baseline，再选择连续 HAR 数据或自采数据验证 patch-level future label sequence。

