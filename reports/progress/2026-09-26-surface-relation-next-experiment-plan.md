# 点间关系方法后续可行性实验计划

- 日期：2026-09-26
- 范围：基于已有 E0--E3 结果制定后续验证顺序；未运行新实验
- 核验状态：PLANNED

## 已有实验不再重复

- E0 已验证 constrained-L2、组件 gauge、零修正、置换和梯度。
- E1 已验证 oracle 关系可恢复结构误差，并给出边噪声和断连的传播边界。
- E2 已完成当前规模的观测图工程筛选。
- E3-controlled 已验证接口和受控优化。

## 后续实验顺序

### F0：同坐标监督链单样本闸门

目标是得到至少一个完整、可审计的 `P0,H` 与未来轨迹标签配对。可接受来源包括作者标签或自行重建的同一坐标链；不把文件来源本身作为方法条件。

必须检查：

- query identity 与 t0 anchor；
- 最后观测相机坐标和归一化；
- 7+15 时间协议，或明确标注为 7+9 pilot；
- 未来标签不进入预测前向；
- validity、t0 自一致性和留出 query 误差。

当前 22 帧 OOM 属于资源阻塞，不能用 12/16 帧 smoke 代替完整标签。若暂时只能得到 7+9，结果只能作为 pilot，不宣称完整官方协议复现。

### F1：真实 baseline 的结构漂移审计

在 F0 标签上先不训练 Relation Head，比较原始 MotionForesight 的：

- relative-vector error；
- pairwise-distance/shape drift；
- component-centered point error；
- drift 随 horizon 的变化；
- 刚体、articulated、deformable 分层结果。

若 baseline 没有稳定的内部结构漂移，原方法的主要动机应收缩；若漂移存在，才进入学习性验证。

### F2：关系信息可学习性 probe

冻结 MotionForesight，按视频划分训练/验证/测试，比较：

1. geometry-only；
2. `H` query readout + geometry；
3. 等参数量 node residual；
4. edge Relation Head proposal。

目标为 `delta r` 或最终 relation。主要指标为 held-out proposal RVE、projected RVE 和 structural-drift AUC，同时报告 ADE/FDE 和 outlier-point rate。

只有在未见视频上 `H`/edge 方案相对 geometry-only 与 node residual 有稳定优势时，才继续最小训练。

### F3：最小端到端消融

固定观测图和训练预算，比较：

1. 原始 MotionForesight；
2. decoded-coordinate relation loss；
3. node residual；
4. Relation Head + constrained-L2 integration。

首轮冻结 backbone，只训练新增小模块；至少 3 个随机种子，按视频而非 query 或 prompt 划分。主指标为 structural-drift AUC、projected RVE 和 centered shape error；ADE/FDE 作为不退化约束。

判断条件应预先登记为：结构指标在多数测试视频和多数种子上改善，且点级预测没有系统恶化。若 edge head 与 node residual 相当，采用更简单方案。

### F4：图和采样稳健性

仅在 F3 出现收益后做：

- 2D `k=3`、observed 3D、修复 3D 的同模型对照；
- 32/48/64 query 子集；
- 历史帧拓扑变化、bridge、谱间隙和条件数；
- rigid/articulated/deformable 分层。

不提前引入 IRLS、Huber、area gauge 或 anchor graph。

### F5：异常边鲁棒性

只有 F3 的普通 L2 版本已显示实际错误边传播明显时，才比较 L2 与 Huber/IRLS。否则保持 L2，避免用鲁棒求解器掩盖图构造或 Relation Head 的系统误差。

## 关键停止条件

- F0 失败：暂停真实训练，问题在监督链。
- F1 不存在稳定结构漂移：收缩研究动机。
- F2 `H`/edge 不超过 geometry-only 或 node residual：删除 Relation Head 主张。
- F3 不改善结构指标或损害 ADE/FDE：不继续增加图和求解器复杂度。
- 只有 F3 通过后，才讨论泛化和鲁棒扩展。
