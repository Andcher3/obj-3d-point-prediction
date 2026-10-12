# MotionMap 表示与平均 codebook 的训练依据

日期：2026-09-09

## 用户确认与讨论边界

- 用户已另行启动远程 L1 扰动有效性测试。本轮将其作为用户报告的进行中工作；未连接服务器或核验测试结果。
- 暂按当前 L1 主动候选生成假设继续讨论，停止重复展开扰动设计。
- 两阶段判别器一直保留：第一阶段模式判别器在候选进入监督目标图前筛选；第二阶段轨迹判别器检查具体轨迹。
- L1 与 SemDiD 式事件探索承担减轻遗漏的任务，模式判别器承担减少错误候选的任务。模块已在概念设计中，具体模型及效果仍待验证。

## 完成的核查

核对 MotionMap 第 3、4.1、4.2.1、4.3、4.4、5 节及官方 autoencoder.py、dataset.py、multimodal.py。区分原论文训练安排、实现证据、机制推断与实验支持；本轮未运行 MotionMap。

### 1. 表示如何反映运动差异

- 历史 X 和未来 Y 分别编码为 z_x、z_y，经过拼接、映射 f 和解码器 D 重建 X 与 Y 的拼接序列。
- 训练目标是带预测方差的运动重建负对数似然。autoencoder.py 训练循环输入 x、mm_y，先做骨架尺寸转换，再重建 x+mm_y 并反向更新。
- 机制推断：同一个 X 对应不同 Y 时，准确重建需要 z_y 保留区分这些未来的信息。这为运动差异提供间接任务约束。
- 该目标没有直接规定潜空间距离应等于事件语义距离，也未规定同事件必然聚在一起。t-SNE 投影使用已学表示的相似关系，不单独提供事件等价监督。
- 论文动作标签的对应与可控性展示、运动多样性展示和预测指标提供经验支持；它们不构成任意事件划分、每个峰语义纯净的证明。动作标签在所核查的自编码器训练循环中未作为分类监督。

### 2. 为什么 codebook 平均向量能作为当前条件

- dataset.py 的 get_codebook 将训练集未来向量按二维量化格归组，对同格向量取均值；没有把所有未来混成一个全局均值。
- 解码始终保留当前观测 X 的条件。平均未来向量不独自承担当前人的完整状态。
- FineTuneDataset 按被选中未来的网格坐标取平均向量，同时返回当前 X 与对应未来 Y。
- multimodal.py 将 X 与平均向量送入 decode，再以经过当前骨架尺寸对齐的 X+Y 作为重建目标。
- 论文第 4.3 节明确为平均向量微调 f 和 D。因而“平均后仍可用”具有直接适配训练依据，不只依赖平均运算本身。
- 若量化将不同运动合并，平均可能损失信息；训练和完整流程实验提供实用性依据，没有普遍的无损平均保证。本轮未找到能单独量化平均向量微调贡献的消融证据，不将整体结果当作该步骤的独立因果证明。

## 对方法讨论的修正

此前对迁移提出的两个要求应同样用于 MotionMap。其依据分别是“运动重建间接塑造表示”和“当前观测条件下对平均向量直接适配”，再由实验检验。项目可以借鉴这一完整逻辑，不必先证明表示天然具有完美事件结构或平均天然有效。

当前概念流程：

L1 主动候选 → 模式判别器 → 保留候选 → 监督目标图；
预测模式条件 + 当前视频 → 三维轨迹生成 → 轨迹判别器。

模式判别器解决候选内容筛选；表示训练与平均向量适配解决筛选后候选如何被编码和使用，两者任务不同。未新增判别器、损失权重或确定新的表示架构。

## 来源

- 论文：https://arxiv.org/html/2412.18883v2
- 表示训练：https://github.com/vita-epfl/MotionMap/blob/main/code/autoencoder.py
- codebook 与微调样本：https://github.com/vita-epfl/MotionMap/blob/main/code/dataset.py
- 平均向量解码适配：https://github.com/vita-epfl/MotionMap/blob/main/code/multimodal.py

## 输出与下一讨论位置

- 本记录：D:\Pycharm\post_graduate_project\reports\progress\2026-09-09-motionmap-representation-and-codebook-evidence.md
- 定向补充前轮记录：D:\Pycharm\post_graduate_project\reports\progress\2026-09-09-active-review-and-motionmap-transfer.md
- 更新索引：D:\Pycharm\post_graduate_project\reports\progress_log.md
- 后续从上述训练机制讨论项目的事件表示与第二阶段条件，不重新将已保留的模式判别器当作缺失模块。
