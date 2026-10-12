# 回查事件多样性教师生成的已有讨论

日期：2026-09-08

## 完成工作与用户澄清

- 用户要求以既有 narration 设计为基准，比较潜状态 H 对第二阶段的相对价值。文本和潜状态同样存在信息遗漏、事件可分性与条件利用问题；不对 H 单独设置绝对充分性门槛。即使 H 与文本承载近似信息，更容易供轨迹模型利用也可能构成优势。
- 回查联合枚举、批量事件假设、语义比较、多样性训练及上一轮互补生成的记录。当前焦点是教师在生成阶段主动探索不同事件，单纯重采样后删除同义句不足以填补遗漏模式。
- 复核 DIVE 官方摘要、Mix-and-Match 作者页与 FUTURE-L1 第 3.3 节。本轮没有模型推理、微调或质量实验。

## 已有讨论的层次

1. **联合枚举与互补生成**：旧记录讨论过让同一次回答中的候选看到前面的候选；上一轮提出在相同观测事实下要求另一个合理后续。这是已有思路中直接改变下一次候选生成条件的一项。此前所有调用次数、条数均是未验证草案。
2. **事件语义比较**：此前讨论过同义描述归并及条件化事件差异。仅用于生成后聚类时，它整理已有候选；将判重结果反馈给后续生成，才能对新的提议过程施加影响。后一连接仍需具体设计。
3. **训练时促进多样性**：此前讨论过随机条件利用和多样性约束，但未确定针对 VANS/FUTURE-L1 的事件级训练目标。DIVE 的通用推断过滤与对比检索学习可作为既有训练层面参考，尚未迁移到这两个教师。
4. **MotionMap 式取峰**：它依赖已学到的模式图，不能自行补齐教师和目标构造中遗漏的事件。当前缺口位于模式图监督来源。

## 当前分析（非已验证方案）

- 可以把“与观测相容”及“为已有候选增加新的事件含义”共同用于教师提议；生成器仍基于同一观测，不把候选未来当作已经发生的历史。
- 固定权重时，可改变候选之间的条件关系并反馈重复原因；若要求模型本身更稳定地探索互补事件，则需要研究解码控制或事件级训练奖励。这两条路线尚未选定。
- 仅增大文本随机性或拉远潜向量距离，不直接指定动作、作用对象、结果关系和事件顺序中的有效差异。
- FUTURE-L1 的 temporal-diversity reward 处理同一推演内相邻潜区间的重复；它与不同采样结果覆盖不同未来事件是不同目标。
- 主动鼓励新模式会改变采样分布。候选可服务于覆盖型模式图构造，次数或被保留条数不自动给出真实事件频率。此前记录中的“跨独立回答复现率”也只是一种教师/生成协议下的复现统计。

## 依据

- `D:\Pycharm\post_graduate_project\reports\progress\2026-08-04-joint-vlm-future-enumeration.md`
- `D:\Pycharm\post_graduate_project\reports\progress\2026-08-04-batched-event-hypothesis-generation.md`
- `D:\Pycharm\post_graduate_project\reports\progress\2026-09-07-hierarchical-modes-and-critic-boundary.md`
- `D:\Pycharm\post_graduate_project\reports\ideas\2026-09-07-multibranch-future-teachers-and-trajectory-supervision.md`
- `D:\Pycharm\post_graduate_project\reports\progress\2026-09-08-multinarration-and-latent-future-interface.md`
- [DIVE 官方论文摘要](https://aclanthology.org/2023.emnlp-main.601/)
- [Mix-and-Match 作者项目页](https://mix-and-match.github.io/)
- [FUTURE-L1 方法](https://arxiv.org/html/2606.05769#S3.SS3)

## 文件与后续讨论

- 新增本记录，并补充前次潜状态接口记录、更新 `D:\Pycharm\post_graduate_project\reports\progress_log.md`。
- 待讨论：怎样把事件等价判断转化为教师生成时的有效反馈，以及是否需要进一步训练。现阶段保持概念讨论，不固定损失、架构或调用预算。
