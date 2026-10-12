# 澄清分层概率、两阶段动机与评判边界

- 日期：2026-09-07
- 完成工作：记录用户对两阶段动机、预训练评判先验、轨迹概率形式和多样性机制的回答，并核对 Stochastic Conditioning 的训练与评估区别。
- 用户确认：两阶段用于区分模式选择和模式内轨迹实现，兼顾概率解释与模型能力分工；希望改善稀有模式覆盖。模式内概率分布与合理性置信度均可接受，具体实现未定。
- 分析界限：模式互斥且完备时概率质量求和为 1；模式内连续密度积分为 1。显式分层便于组织输出，稀有模式仍需要被第一阶段提出和保留。直接生成完整轨迹分布在数学上同样成立。
- 例子修正：此前杯子例子未说明观察截止时刻。已观察到拿起与真实未来后来发生拿起对应不同条件；预测时不将后者作为已知事实。
- 主要待讨论：若候选与观测相容，仅因推测的意图较少见而剔除，可能与低频模式覆盖目标冲突。预训练知识可以作为判断来源，其充分性尚未验证。
- 论文核对：Mix-and-Match 通过随机条件扰动促进使用随机变量；训练涉及重建和 KL；Diversity、Quality、IS 是评估指标，质量分类器属于评估流程。依据为 CVF 作者摘要及论文第 3.2–3.4、4 节的可检索正文段落，未检查代码。
- 来源：[CVPR 2020 论文](https://openaccess.thecvf.com/content_CVPR_2020/papers/Aliakbarian_A_Stochastic_Conditioning_Scheme_for_Diverse_Human_Motion_Prediction_CVPR_2020_paper.pdf)；[作者项目页](https://mix-and-match.github.io/)。用户提供的 alphaXiv 私有链接本轮未返回正文。
- 输出：`D:/Pycharm/post_graduate_project/reports/ideas/2026-09-06-two-stage-two-critic-idea-synthesis.md`；本记录；`D:/Pycharm/post_graduate_project/reports/progress_log.md`。
- 后续讨论：明确模式评判器对观测相容但意图低频的候选应采用何种接受标准。具体多样性损失和参数暂不固定。
- 验证范围：概念讨论、原文核对与文档更新；未进行模型实验。
