# L1 latent 接口与 TrajLoom 三维适配讨论

日期：2026-09-10

## 本轮完成

依据用户最新五点回复更新方法状态；核对 TrajLoom 论文、官方仓库，以及 MotionForesight 论文与官方仓库。仅完成文献和接口分析，未运行模型、下载权重或验证组合效果。

## 用户确认与候选

- 第一阶段向第二阶段交付 FUTURE-L1 的未来 latent。相对 narration 的信息优势是研究假设，接口形式已由用户选定。
- 主线按在线 L1 + 无 heatmap 的直接分支方式讨论；离线 L1 + MotionMap 式地图为备选。离线无图学生属于助手额外提出的选项，尚未选用。
- 用户提出 TrajLoom × MotionForesight 作为第二阶段候选，并希望在此路线下省去独立轨迹评判器；该组合及删评判器效果尚未验证。
- 数据集候选收敛到 Ego4D、SSv2。用户倾向沿用 MotionForesight 的目标指定和坐标方式，时长参考其原设置或稍长，具体采样协议尚未确定。

## 核查所得

- [TrajLoom 论文](https://arxiv.org/html/2603.22606) §3 的轨迹是二维图像坐标，配可见性；其轨迹 VAE latent 与 L1 的未来 latent 属不同表示空间。
- [官方仓库](https://github.com/zewei-Zhang/TrajLoom) 提供推理说明，Training 部分仍写训练代码后续发布。现有输出包含轨迹与可见性，未展示本项目需要的每条轨迹发生概率接口。
- [MotionForesight 附录 A](https://arxiv.org/html/2607.16192) 提供参考点、坐标系及监督有效性定义；7 帧观察、15 帧未来的帧数设置须结合采样时间解释。
- [MotionForesight 官方仓库](https://github.com/brains-bots-n-behavior/motionforesight) 的预处理保存 frame_indices，并提供不同时间抽帧方案；不能从帧数单独推断固定秒数。

## 分析与建议（尚未选定实现）

1. 目标 mask 与查询点接口可以沿用；同时需让 L1 分支围绕同一目标和预测时间段。第二阶段选点不会自动改变第一阶段关注对象。
2. 控制工作量的方向是保留一个三维轨迹生成主干，借鉴另一工作的生成和一致性机制。直接串接两个完整系统会增加表示、坐标和训练适配；L1 latent 宜作为条件，不能假定可替换轨迹 VAE latent。
3. 离线无图路线依靠学生预测分支集合：L1 在预处理生成并缓存条件与多个 latent，学生据此训练，推理时由学生提出分支。地图是可选输出表示；该学生需要额外设计，当前不纳入主线。
4. 可以先讨论无独立轨迹评判器的简化版本；时空一致性训练、可见性预测、物理合理性和轨迹发生概率各有不同含义。每条轨迹的数值仍未定义。
5. 同一真实未来如何与 L1 分支配对，以及额外事件分支缺少当前场景三维目标的问题，仍需解决。生成式训练可以跨样本学习条件规律，不自动产生额外真值。
6. 轨迹伪标签指从已录制视频离线估计的三维点路径；监督有效性规则决定哪些点/时刻的标签可参与损失或指标。它与候选轨迹合理性评判不同。预测时仅使用观测前缀；遮挡不自动等于标签无效。

## 下一讨论位置

明确 TrajLoom × MotionForesight 实际保留哪个轨迹 latent/解码接口，以及如何训练 L1 条件适配；随后统一两个数据集的采样时间、分支对应时间与三维标签协议。暂不固定层数、分支数、损失权重或更长预测窗口。

## 本地输出

- 本记录：`reports/progress/2026-09-10-latent-interface-trajloom-feasibility.md`
- 更新方法状态提示：`reports/progress/2026-09-10-direct-branch-method-status.md`
- 更新路线：`research_roadmap.md`
- 更新历史设计的最新补充：`reports/ideas/2026-09-06-two-stage-two-critic-idea-synthesis.md`
- 更新索引：`reports/progress_log.md`
