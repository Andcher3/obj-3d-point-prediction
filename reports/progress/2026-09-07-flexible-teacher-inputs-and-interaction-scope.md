# 放宽未来候选教师输入，并复核人—物交互范围

日期：2026-09-07

状态：完成概念讨论所需的定向来源核查；未下载或运行模型，未选定教师、数据集和最终预测接口。

## 用户本轮明确的调整

- 研究可以聚焦人—物交互中的目标物体运动，不把无人参与的物体自主运动能力作为候选教师的必要条件。
- 候选教师最终需提供未来自由文本描述；输入可以包含历史叙述、动作记录、视觉信息等，并与数据集共同选择。
- ViNa 缺少已确认的公开实现与权重，移出可直接使用的候选，仅保留论文方法参考。

## 完成的核查与理解

1. [MotionForesight 第 3.1、4.1 节](https://arxiv.org/html/2607.16192)确认使用约 40K 条 SSv2 人—物交互视频训练，输出被操作物体的未来三维点轨迹。此前把候选教师要求扩大到无人参与的开放物理事件，没有必要。
2. [ViNa 论文](https://arxiv.org/html/2503.01416v1)仍使用将发布代码和标注的表述；本轮检查论文页面并检索标题、模型名和作者，未找到作者公开实现或权重。该结论是有范围的检索结果。前轮将其列为方法相关工作，并未证明其可直接运行。
3. [Ego4D 标注文档](https://ego4d-data.org/docs/data/annotations-schemas/)提供带时间戳的 narration_text 及有时间范围的摘要字段。这使“历史叙述/状态 → 未来文本”的教师路线具有数据接口依据；具体样本还需检查观察截止时间、对象指向和文本覆盖范围。
4. [User-in-the-Loop Evaluation of Multimodal LLMs for Activity Assistance，WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Verghese_User-in-the-Loop_Evaluation_of_Multimodal_LLMs_for_Activity_Assistance_WACV_2025_paper.html)比较了视觉历史转为文字后由 LLM 预测，以及直接视觉条件语言模型。这支持扩展模型搜索范式；本轮没有确认该论文的公开实现，不将其列为可直接使用的教师。
5. [EgoPlan 论文](https://arxiv.org/html/2312.06722v3)、[官方仓库](https://github.com/ChenYi99/EgoPlan)及[权重页面](https://huggingface.co/ChenYi99/EgoPlan-Video-LLaMA-2-7B)提供目标条件下一步规划的实例和公开代码/微调权重入口。原任务给定目标，主要通过候选答案排序评测；该证据尚未验证未知意图下的多分支自由文本预测。

## 当前判断

- 放宽输入可让文本语言模型、视觉与文字联合模型、目标条件规划模型进入考察范围。未来描述模型不必原生直接读取 RGB，也不必自身输出三维轨迹。
- 人类行为描述可表达目标物体的粗运动分支；“人拿起杯子”等描述适合作为模式候选。细节轨迹生成仍留给轨迹阶段。
- 教师和最终预测器的输入可以分别设计。历史文字只在教师训练阶段可得、推理时也可得、或由观测视频生成，分别对应不同的使用设定；本轮不冻结任何一种。
- 预测时可用的任务目标可以作为条件；由完整后续总结出的目标需要注明监督/特权信息身份，不直接按未知目标的自主预测评测。
- 自由文本接口、代码与权重可获得、多分支覆盖、额外未来标签可靠性是不同层次的要求。选择范围扩大，不自动解决真值可靠性。
- Ego4D 的叙述有利于语义监督，目标物体三维轨迹的提取、可见性及时间对应仍需另外建立。

## 本地输出

- `D:/Pycharm/post_graduate_project/reports/progress/2026-09-07-flexible-teacher-inputs-and-interaction-scope.md`
- 更新 `D:/Pycharm/post_graduate_project/reports/progress_log.md`。
- 补充 `D:/Pycharm/post_graduate_project/reports/ideas/2026-09-07-conditional-event-similarity-and-teacher-audit.md` 的教师可用性与输入范围修订。

## 待讨论

按人—物交互、自由未来描述、可获得实现/权重、可匹配的历史信息继续筛选；候选生成教师与数据集保持未选定状态。
