# MotionForesight 视频未来轨迹研究进展

ACTIVE_RESEARCH_SCOPE=VIDEO_MOTION_FORECASTING

本索引只记录当前完全基于视频的研究方向。旧 IMU/HAR/OIS 索引已保存到 `archive/imu/progress_log.pre-2026-09-03.md`。

## 进展索引

当前讨论提炼：[思路与待解决问题（含 2026-09-07 修订）](ideas/2026-09-06-two-stage-two-critic-idea-synthesis.md)。当前优先关注开放世界三维轨迹的多模态预测；两阶段为候选方式，语义—近期动态分工降低优先级，每条轨迹的概率表达尚未选定。既有方法笔记继续作为非约束的历史方案。

- 2026-08-04 [审查 MotionForesight 类多未来方法](progress/2026-08-04-motionforesight-method-review.md)：检查“语义等价类 + 条件 Flow”任务、训练边界和多未来评测问题。
- 2026-08-04 [细化语义条件与点轨迹处理](progress/2026-08-04-motionforesight-points-5-7-solutions.md)：形成开放事件关系、语义条件训练、无序点集与伪轨迹处理候选方案。
- 2026-08-04 [复查 MotionForesight 修订](progress/2026-08-04-motionforesight-revision-recheck.md)：保留语义利用、零权重 token、兼容事件和固定分支边界问题。
- 2026-08-04 [整理组会方法构想](progress/2026-08-04-motionforesight-method-discussion-review.md)：将开放事件图、语义条件 Flow 和多未来评测整理为阶段性讨论版本。
- 2026-08-04 [澄清开放事件词表](progress/2026-08-04-event-vocabulary-problem-explanation.md)：固定字段结构，保留开放事件内容，区分等价、蕴含、兼容和互斥关系。
- 2026-08-04 [审查 VLM 联合枚举未来](progress/2026-08-04-joint-vlm-future-enumeration.md)：确认同一回答中的候选频数不能解释为概率。
- 2026-08-04 [设计批量事件假设协议](progress/2026-08-04-batched-event-hypothesis-generation.md)：提出多次独立调用、每次少量候选的生成方案。
- 2026-08-04 [端到端审查语义条件 Flow](progress/2026-08-04-motionforesight-end-to-end-method-audit.md)：统一任务、候选语义、轨迹监督、条件 Flow 和多样本推理边界。
- 2026-08-04 [推演非文本未来潜变量](progress/2026-08-04-nontext-future-latent-redesign.md)：将文本事件替换为从未来视频学习的连续交互潜变量候选。
- 2026-08-05 [筛选相关论文并生成方法图](progress/2026-08-05-motion-forecast-literature-method-figure.md)：整理 MotionForesight、Generative Point Tracking and Forecasting、MYRIAD，并生成无显式语义的方法总览图。
- 2026-08-05 [澄清未来样本数量](progress/2026-08-05-variable-future-sample-count.md)：确认样本数是可变推理预算 `S`，固定六路仅为展示示例。
- 2026-08-05 [完成运动预测论文介绍页](progress/2026-08-05-motion-forecast-paper-slide-images.md)：完成两篇论文的 10 张介绍页和发言稿。
- 2026-09-02 [复盘项目现状与 FlowForesight 构想](progress/2026-09-02-project-status-flowforesight-review.md)：确认当前尚处于方法设计和汇报材料阶段。
- 2026-09-03 [固定视频预测范围并归档 IMU 工作](progress/2026-09-03-motionforesight-scope-reset.md)：将当前范围固定为视频输入的指定物体未来三维轨迹分布预测，归档旧 IMU 工作，并记录连续概率和语义—物理分层两个核心创新目标。
- 2026-09-04 [审查 MotionMap 与预训练动作先验重设计](progress/2026-09-04-motionmap-foundation-prior-review.md)：将旧组会方法及数字降为非约束草案，区分显式模式概率、随机条件利用与外部先验扩展，提出连续运动潜空间概率场候选。
- 2026-09-04 [细化伪分布监督、随机条件与对抗训练](progress/2026-09-04-distribution-target-adversarial-training.md)：说明怎样构造 MotionMap 式软分布目标，分离宏观模式与模式内随机性，并界定判别器、多样性损失和 IS 类指标的作用。
- 2026-09-06 [提炼两阶段、两评判器的当前思路](progress/2026-09-06-two-stage-two-critic-idea-synthesis.md)：保留文本粗模式与模式内三维轨迹的分工，澄清两级合理性筛选与全局轨迹密度的差别，整理监督、语义等价、模式空间及评估问题。
- 2026-09-07 [调整多模态三维预测的研究优先级](progress/2026-09-07-multimodal-forecasting-priority-clarification.md)：将多模态未来预测置于首位，降低语义分工优先级；撤回助手对全局轨迹密度需求的过度解释，讨论监督、两阶段价值、概率事件与评估问题。
- 2026-09-07 [澄清分层概率与评判边界](progress/2026-09-07-hierarchical-modes-and-critic-boundary.md)：记录模式选择与轨迹实现的两阶段动机，区分已观察状态、未知意图和低频模式；确认两种轨迹数值表达均可讨论，并纠正 Stochastic Conditioning 的损失与评估指标混淆。
- 2026-09-07 [讨论模式偏离与模式图覆盖](progress/2026-09-07-mode-drift-and-map-peak-coverage.md)：保留合理跨模式候选的讨论空间，区分候选质量、源模式覆盖和概率修正；核对 MotionMap 目标构造与峰值提取的不同作用。
- 2026-09-07 [讨论语义模式图的构造](progress/2026-09-07-semantic-mode-map-construction.md)：区分共享模式空间与条件响应图，梳理 MotionMap 的多未来目标来源；提出未来事件表示及预训练候选弱监督的待验证方向。
- 2026-09-07 [核查事件语义空间与未来候选教师](progress/2026-09-07-event-semantics-and-future-teacher-review.md)：核对双向蕴含、AMR、TMR 及 GePSAn、AntGPT、ViNa；确认存在多未来文字提议先例，适用于当前模式图的额外正标签可靠性仍待验证。
- 2026-09-07 [调研语义距离后续方法并复核教师](progress/2026-09-07-semantic-distance-followups-and-teacher-audit.md)：比较 KLE、SD、SNNE 与 C-STS 后续 CASE、CCL 等工作；核对 AntGPT 的非厨房场景及固定词表、ViNa 的文字依赖和多候选生成，教师继续保持未选定。
- 2026-09-07 [放宽教师输入并复核交互范围](progress/2026-09-07-flexible-teacher-inputs-and-interaction-scope.md)：聚焦人—物交互中的物体轨迹，教师输入与数据集联合选择；ViNa 移出可直接使用候选，核查历史叙述路线及 EgoPlan 的代码、权重入口与目标条件边界。
- 2026-09-07 [调研多分支文本教师与单真实未来轨迹监督](progress/2026-09-07-multibranch-teachers-and-single-future-supervision.md)：核查 VisualCOMET、KM-BART、DIVE、VANS 等自由文本多结果路线及公开资源；回查语义匹配配对训练，明确额外文字分支尚缺当前场景的几何监督，补充 MolmoMotion 数据与已发布自回归模型。
- 2026-09-08 [核查教师输出质量及三篇新方法](progress/2026-09-08-stage1-teacher-quality-and-new-methods.md)：检查 NEP、Video-CoE、FUTURE-L1 及旧模型的公开输出、评价条件与资源；区分已知结尾选择题成绩、开放问答评分和多分支伪标签质量，记录图例疑点并调整候选优先级，教师继续保持未选定。
- 2026-09-08 [复核四篇工作的权重与推理入口](progress/2026-09-08-teacher-checkpoints-and-inference-readiness.md)：确认 VANS 与 FUTURE-L1 均已公开训练后权重和推理实现，更正此前漏查 FUTURE-L1 权重的结论；核对文本接入所需适配，NEP 与 Video-CoE 暂未确认现成检查点，尚未运行任何教师。
- 2026-09-08 [讨论多未来文本与潜状态接口](progress/2026-09-08-multinarration-and-latent-future-interface.md)：核对 VANS 随机文本解码、FUTURE-L1 完整推演采样及潜状态导出；讨论离线候选生成与直接潜状态条件，保持模式覆盖、目标轨迹信息和教师用途为待验证问题。
- 2026-09-08 [回查事件多样性教师生成](progress/2026-09-08-event-diverse-teacher-generation-recap.md)：按 narration 基准澄清潜状态的相对价值，回查联合枚举、互补生成、语义比较和多样性训练；明确尚缺针对教师生成过程的事件级鼓励机制，候选覆盖与模式权重保持分别讨论。
- 2026-09-08 [讨论冻结 L1 的语义前瞻解码](progress/2026-09-08-frozen-l1-semantic-lookahead-decoding.md)：记录用户优先考虑 L1 且少调或不调权重的偏好；核查 SemDiD、NLI 重采样及分支搜索，提出事件级前瞻与分支反馈候选，保留潜状态配对和推理适配问题，尚未实现或测试。
- 2026-09-08 [审阅 SemDiD 与 L1 主动潜状态调整](progress/2026-09-08-l1-semantic-latent-steering-review.md)：读取共享讨论，核查冻结权重下文本目标反向调整潜状态的依据；区分文字可控性、未来视觉信息、候选发现与协同分支，保留两版为待验证候选，未实现或运行模型。
- 2026-09-08 [澄清 L1 联合多分支生成与分叉位置](progress/2026-09-08-l1-joint-branch-generation.md)：记录一次输入联合维护多个候选的目标；核对 SemDiD 的逐步搜索与 L1 连续状态更新，提出在对应潜状态形成前探索文字事件分叉的讨论方向，尚未确定或验证。
- 2026-09-08 [核查 L1 首个 reason 的实际内容](progress/2026-09-08-l1-first-reason-content-audit.md)：逐一检查图 15–18 及训练/提示模板；四个展示案例首段均以观测概括及活动背景为主，暂缓将首段作为默认未来事件分叉点，已向此前记录追加修正，未运行模型。
- 2026-09-09 [讨论 L1 潜状态分叉与事件前瞻反馈](progress/2026-09-09-l1-latent-branch-search-options.md)：核查 LatentTTS 论文、采样及搜索代码和 GTS；提出冻结 L1 的 latent 提议与事件级前瞻搜索候选，保留小型学习式采样器为备选，尚未验证视觉未来分支效果。
- 2026-09-09 [暂定随机扰动主动版并梳理方法缺口](progress/2026-09-09-active-l1-selection-and-method-gaps.md)：记录用户暂定采用 L1 latent 随机扰动与事件前瞻反馈；梳理模式图、阶段接口、额外轨迹监督、评判器、输出数值及数据评估六项未决内容，未新增架构或开展实验。
- 2026-09-09 [审阅主动版评价并核对 MotionMap 迁移](progress/2026-09-09-active-review-and-motionmap-transfer.md)：保留随机提议和事件前瞻，指出自动增噪及似然认证等推断过强；核实 MotionMap 以最大值合并高斯目标、codebook 平均与解码适配，明确可借用建图流程，表示与监督迁移仍待确定。
