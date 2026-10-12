# 正式 E2 的研究决策与代码执行分工

日期：2026-09-22

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-22
- Verification Status: ANALYZED
- Version Label: formal_e2_responsibility_v1

## 当前判断

正式 E2 的图方案已足够明确：2D symmetric kNN `k=3` 为主候选，observed 3D symmetric kNN `k=3` 为最小对照，hybrid 暂缓。当前不需要继续修改核心算法，阻塞来自新增独立视频和高密度 query 数据。

## 需要研究负责人确认的事项

1. **独立视频单位**：不同 source video ID 才算独立样本；prompt、stride、起始窗口和重复推理不增加独立样本数。
2. **query 数量与采样规则**：将 32--64 冻结为具体值，并固定目标 mask 内的采样方式、最小点间距、随机种子和是否做 farthest-point/分层采样。
3. **视频覆盖**：至少 5 个独立视频，并确认是否覆盖 rigid、articulated、deformable 三类运动；现有 3 个视频只能作为先导数据或附加分析。
4. **时间对应规则**：明确预测 30 步与 pseudo-GT 9 步的时间戳映射，采用精确匹配、最近邻或插值中的一种，并在正式运行前固定。
5. **主判据与停止条件**：固定 temporal RVE AUC 为主要结构指标；预先确定是否以三视频宏平均、视频级中位数、最差视频或分层结果作为图选择依据。
6. **样本纳入与排除**：确认遮挡、断轨、负深度、目标 mask 不完整、伪标签不可用等情况的有效性规则。

这些事项属于研究设计和证据解释，代码 AI 不应自行决定。

## 我负责的工作

- 审核新增数据是否满足独立视频、query 数量、目标类型和时间协议；
- 检查是否存在未来帧泄漏、prompt 变体伪重复、坐标系/单位不一致和 pseudo-GT 错配；
- 在正式运行前审查最终实验 manifest，确认主指标、分层聚合和停止条件已冻结；
- 复核实验输出，区分视频级证据、随机种子方差和图结构效应；
- 根据预先设定的门槛判断 2D `k=3` 是否继续保留、是否需要恢复 3D 对照、是否可以进入 E3；
- 在 E3 前确认 MotionForesight 的 `P0,H` 接口和 observed-only 边界，避免把 oracle/E1/E2 结果误写成 Relation Head 已学会。

## 代码 AI 负责的工作

- 编写数据预检器：视频唯一性、query 数、数组形状、NaN/Inf、深度、坐标单位和文件去重；
- 生成并保存 query sampling manifest、随机种子和 source-video split；
- 实现时间戳匹配报告，但不自行选择最终映射规则；
- 实现 2D `k=3` 与 3D `k=3` 的正式 E2 runner；
- 计算 query subset/density、桥边、谱间隙、KKT 条件数、跨历史帧拓扑变化和 temporal RVE AUC；
- 输出逐视频、逐运动类型和总体汇总，不将 prompt 变体当成独立样本；
- 完成本地/AutoDL 复跑、哈希或数值差异检查，以及可重运行的 manifest。

## 当前可以立即开展的准备工作

即使新增视频尚未到位，代码 AI 仍可完成：

1. 正式 E2 输入 schema 和预检脚本；
2. query 采样与去重 manifest 生成器；
3. 时间戳候选映射报告器；
4. 固定 2D/3D `k=3` 的正式 runner 和结果 schema；
5. 一组故意不合格数据的预检单元测试。

这些工作不会把现有 8-query 样本伪扩展为正式 E2。

## 用户需要提供的最小决策包

- 新增独立视频清单或数据来源；
- 具体 query 数量和采样策略；
- 运动类别覆盖要求；
- 30→9 时间映射规则的确认；
- 正式 E2 的主判据和纳入/排除规则。

完成这五项后，代码 AI 可以直接执行正式 E2；结果出来后再由我进行方法学复核和 E2→E3 放行判断。

## 下一步

等待新增独立视频和高密度 query 数据，同时先完成正式 E2 的数据预检、manifest 和时间映射审计工具。
