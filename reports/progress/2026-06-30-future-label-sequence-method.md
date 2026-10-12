# 梳理 Future Patch Label Sequence 方法草案

- 日期：2026-06-30
- 完成内容：根据用户确认的 A 方案，将“通过未来 token/patch 的 label 序列推导切换时机、切换概率和下一动作”的方法重新系统梳理为可落地的模型草案。
- 本地产出：
  - `reports/ideas/2026-06-30-direction1-future-label-sequence-method.md`
  - `reports/progress/2026-06-30-future-label-sequence-method.md`
- 关键理解/结果：方法主线从“transition label 分类”调整为“future patch label sequence prediction”。模型对未来每个 patch 输出标签概率分布，首次标签变化给出切换时机，变化后的标签给出下一动作，标签置信度可计算延续概率与切换概率。该方案可先在人体动作单标签上落地，再扩展为 `human_action_sequence + phone_motion_sequence`。
- 下一步：选择连续 HAR 数据集或构造连续序列实验协议，验证 `past IMU patches -> future labels` 的预测、切换时机误差和下一动作准确率。

