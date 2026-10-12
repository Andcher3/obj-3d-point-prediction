# 语义距离后续文献与候选教师复核

日期：2026-09-07

## 已完成

- 核查 KLE、Semantic Density、SNNE 对语义关系与不确定性计算的处理。
- 调研 C-STS 后续 CASE、CCL、Hyper-CL、SEAVER 及标注改进工作；形成以任务适配性为依据的阅读与比较优先级，未选定实现。
- 复核 GePSAn 的任务域，AntGPT 的词表、场景与事件步数协议，以及 ViNa 的输入消融、多样本输出、对象与时间范围。
- 修正此前对 ViNa 多输出机制交代不足的问题，保留其尚未验证对象级模式覆盖的界限。

## 关键结果

CASE（EACL 2026）与 CCL（COLING 2025）更直接对应条件化语义空间学习。AntGPT 的实验包含非厨房日常活动，但输出仍受动作标签协议约束。ViNa 明显依赖历史叙述，原文也写有多候选生成；这两点应同时保留。当前未来候选教师仍未选定。

## 输出

- 调研报告：`D:\Pycharm\post_graduate_project\reports\ideas\2026-09-07-conditional-event-similarity-and-teacher-audit.md`
- 更新思路记录：`D:\Pycharm\post_graduate_project\reports\ideas\2026-09-06-two-stage-two-critic-idea-synthesis.md`
- 补充前次研究记录：`D:\Pycharm\post_graduate_project\reports\progress\2026-09-07-event-semantics-and-future-teacher-review.md`
- 更新索引：`D:\Pycharm\post_graduate_project\reports\progress_log.md`

## 后续讨论

讨论事件比较应关注的语义维度；继续保留候选教师的观察条件、目标物体覆盖和时长适配问题。本次只完成文献核查，没有模型实验。
