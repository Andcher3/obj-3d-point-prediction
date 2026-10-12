# L1 潜状态分叉与事件前瞻反馈候选

日期：2026-09-09

## 后续确认与审阅补充

用户随后暂定采用随机扰动主动版，见 [确认记录](2026-09-09-active-l1-selection-and-method-gaps.md)。再次审阅外部 AI 评价后，保留随机 latent 提议与事件前瞻反馈主干；不把单一事件诊断为扰动过小，不用文字高似然认证视觉状态，不默认增加自动调幅、两级 beam 或范数惩罚。原生 SemDiD 的句向量评分需要事件级适配。详见 [批判性审阅及 MotionMap 迁移核查](2026-09-09-active-review-and-motionmap-transfer.md)。下文保留最初提议状态和证据。

## 完成工作

在此前首个 reason 内容核查基础上，查阅 Parallel Test-Time Scaling for Latent Reasoning Models（LatentTTS）的 ACL 论文与公开代码，以及 GTS 的论文全文。形成一个新的待讨论方向：保持已观察内容的描述稳定，在连续 latent 生成阶段提出不同分支，用后续事件文本的语义比较指导分支的保留与继续扩展。

本次仅完成文献及静态代码核查和概念建议；未下载权重、未运行 L1、未修改模型代码，也未确定新架构。

## 已核查证据

- LatentTTS 在 latent 推理中引入 MC-dropout 或加性高斯扰动，提供不更新推理骨干权重的采样途径；其 LatentRM 另行训练，支持 best-of-N 与 beam search。实验为算术推理，不能据此声称已验证视频未来事件覆盖。
- 作者仓库公开了采样及搜索代码。`src/models/perturbation.py` 在 Linear 输出后加 dropout；`src/generation_mixin.py` 分别控制 latent 与显式文本的随机性，包含 beam search 路由。该代码尚未接入 FUTURE-L1。
- GTS 保持骨干冻结，训练上下文条件高斯扰动采样器。其奖励使用答案正确性与置信度，实验包含 COCONUT/CODI 与算术数据。固定噪声增大造成更大分布偏移，并不总带来更好的有效候选。
- GTS 的扰动策略密度描述探索机制，不应作为真实未来事件的发生概率。其正确答案奖励也没有直接解决本项目的多未来弱监督问题。

## 新建议：latent 提议、事件前瞻、分支反馈

以下为本项目的适配假设，不属于两篇参考论文已经验证的结论。

1. 共享观测编码与首段内容，在 latent 生成阶段引入分支差异；最简单的无参数更新提议器可参考局部随机扰动。
2. 对候选状态临时向后推演到足以说明目标未来事件的文字，用事件等价判断比较分支，并同时考虑与观测的相容性。
3. 比较结果决定哪些完整状态分支继续扩展，减少重复事件占据的探索预算。不能只修改文本或保留导出的 latent 数组而丢失对应生成上下文。
4. 在后续生成中重复提议与比较，使语义差异参与搜索过程。最终保留成套 latent、事件描述及其生成分支。

与已有保守讨论的区别是明确连续 latent 内的随机提议机制，并将语义反馈用于尚未完成的分支扩展。该方式仍有采样及前瞻开销，不声称消除了过采样，也不保证一定找到所有事件。

## 关键未决项

- 受扰动状态可能产生文字变化，却损害或未改变对轨迹有用的视觉预测信息；需要检查相对 narration 基准的下游价值。
- 紧随 latent 的一段 reason 未必足以区分完整未来。例如两个分支都先拿起杯子，后续可分别喝水或递给别人。前瞻比较应针对同一目标时域中的事件，避免过早按中间步骤合并。
- 文本前瞻只是该状态后续行为的代理测量，不是其视觉内容的完美解释。扰动后的语义差异和合理性都尚未验证。
- 若提议器始终只探索同一事件，语义筛选本身不能创造缺失的事件；搜索反馈只能重新分配已有探索机会。

## 少量训练的备选

若无训练扰动的有效分支比例不足，可参考 GTS：保持 L1 冻结，仅训练小型状态提议器。以事件差异与观测相容性提供反馈是新的研究假设，尚缺可靠奖励及监督来源，不能将原论文的数学答案奖励直接移植。

## 后续讨论焦点

比较“冻结模型的 latent 提议与语义前瞻搜索”和“小型学习式提议器”的取舍；先判断状态扰动是否产生有用的事件分叉，再讨论完整模式图。当前没有选定具体控制位置、扰动强度或分支数。

## 来源

- [LatentTTS 论文](https://aclanthology.org/2026.acl-long.2069.pdf)
- [LatentTTS 作者代码](https://github.com/ModalityDance/LatentTTS)
- [扰动模块](https://github.com/ModalityDance/LatentTTS/blob/main/src/models/perturbation.py)
- [生成逻辑](https://github.com/ModalityDance/LatentTTS/blob/main/src/generation_mixin.py)
- [GTS 论文](https://arxiv.org/html/2602.14077)
- [FUTURE-L1 论文](https://arxiv.org/html/2606.05769)

## 本地输出

- `D:\Pycharm\post_graduate_project\reports\progress\2026-09-09-l1-latent-branch-search-options.md`
- `D:\Pycharm\post_graduate_project\reports\progress_log.md`
