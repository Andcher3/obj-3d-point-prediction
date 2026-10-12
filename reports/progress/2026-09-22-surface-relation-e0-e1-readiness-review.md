# E0--E1 结果复核与后续实验门槛

日期：2026-09-22

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-22
- Verification Status: VERIFIED
- Version Label: validation_v1

## 判断

E0 已充分通过。E1 达到工程筛选门槛，可以开展 E2-lite。当前证据只支持求解器正确、内部结构误差可被关系积分恢复，以及图错误会传播；尚不支持 Relation Head 有效、完整方法有效或 IRLS 应进入主方案。

E3、E4 暂缓，等待 MotionForesight 的 observed-only `P0,H` 和正式时间对应协议。E5 暂不启动。

## 复现核对

- 使用固定种子 `20260922`、`k=3` 重新运行 E0、E1。
- 新生成的 `e0_solver_unit_tests.json`、`e1_oracle_noise.json`、`summary.json` 与原始本地文件逐字节一致。
- 本地与 AutoDL 的 E1 最大数值差异为 `2.66e-15`；路径差异仅来自 Windows 与 Linux 分隔符。
- 本地与 AutoDL 的 E0 最大数值差异为 `2.42e-9`，主要来自有限差分梯度误差；两端均远低于 `1e-3` 门槛。

## 结果解释

### E0

- 最大 component-centered RMSE 为 `5.65e-16`；
- zero correction 为 `0`；
- 最大节点置换误差为 `1.07e-14`；
- 最大有限差分梯度相对误差为 `3.71e-9`。

这些结果足以支持当前 KKT component-gauge L2 求解器进入后续接口和图实验。

### E1 的视频间差异

| 视频组 | 5% Gaussian 恢复率 | 10% Gaussian 恢复率 | 10% outlier 恢复率 | 人工断连恢复率 |
|---|---:|---:|---:|---:|
| drink bottle | 42.4% | -11.4% | -64.0% | 81.4% |
| sewing scissors | 92.2% | 85.4% | 78.7% | 38.5% |
| watering can | 90.8% | 83.8% | 71.3% | 59.2% |
| 三组宏平均 | 75.1% | 52.6% | 28.7% | 59.7% |

宏平均通过原定门槛，但饮料瓶组对噪声非常敏感。负恢复率表示积分结果比原预测的结构误差更大。正式图选择不能只依赖三组宏平均。

## 有效性限制

1. 21 个文件来自 3 个源视频。按数组哈希统计，实际有 5 组不同的 history--pseudo-GT 配置和 12 组不同的 history--GT--prediction 配置；prompt 与时间协议变体仍然高度相关。
2. `gaussian_0.05` 表示每个坐标轴的噪声标准差为 GT 边长中位数的 5%，不能直接解释成三维边误差恰好为 5%。
3. outlier 实验只有一个随机种子。8 点图约 15 条边，10% 污染通常只有 1--2 条边，扰动幅度为 `0.5 × median edge length`，且时间上的扰动独立采样。
4. 人工断连按 query 索引前后两半划分，只是数值断连检查，不对应真实物体部件或真实桥边失效。
5. `structural_drift_auc` 当前实现等于全时域平均 RVE，尚未形成独立的时域 AUC 指标。
6. `future_3d[:9]` 与 9 步 pseudo-GT 的直接对应尚未由时间戳验证。涉及恢复率的正式实验需要固定时间映射。
7. E1 使用 MolmoMotion 输出，只验证模型外部的坐标积分器；Relation Head 的可学习性仍未测试。

## E2-lite 范围

当前可立即执行：

- 比较 2D kNN、observed 3D kNN、hybrid graph，使用 `k=2/3/4`；
- 报告连通分量、最大分量覆盖率、桥边比例、谱间隙或等价稳定性指标、KKT 条件数和历史边稳定性；
- 每个噪声设置至少运行 20 个随机种子，报告中位数、IQR、最差视频和失败比例；
- 按源视频报告结果，并单独标记同一视频的 prompt、时间窗口和预处理变体；
- 将 RVE 按预测时刻展开，另行定义真正的 horizon AUC。

正式 E2 还需要 5--10 个视频、每个视频 32--64 个 query，并覆盖 rigid、articulated、deformable 运动。query density/subset 稳定性应在该规模上判断。

## 后续门槛

- E2 图选择：简单图若与 hybrid 相当，采用简单图；任何方案若在多数种子或某一运动类别持续恶化，停止增加图复杂度。
- E3：获得 MotionForesight observed-only `P0,H` 后，验证 query feature head 是否稳定优于 geometry-only 和 zero correction。
- E4：E3 通过后，再比较 node residual、decoded-coordinate relation loss 和 Relation Head + integration。
- IRLS：先测量 Relation Head 的真实边残差分布。当前合成 outlier 结果不足以决定采用鲁棒积分。

## Fallacy Scan

- Coverage: 11/11 checked

| 类型 | 结果 | 说明 |
|---|---|---|
| Simpson's paradox | CAUTION | 宏平均通过而饮料瓶组未通过；尚无方向反转，存在聚合掩盖异质性的风险。 |
| Ecological fallacy | NOTE | 未从视频组均值推断单点行为。 |
| Berkson's paradox | CAUTION | 三个现成视频属于便利样本，不能代表目标数据分布。 |
| Collider bias | NOTE | 当前无协变量控制或回归分析。 |
| Base-rate neglect | NOTE | 当前无诊断概率指标。 |
| Regression to the mean | NOTE | 当前无极端样本筛选后的前后测推断。 |
| Survivorship bias | NOTE | 21 个 NPZ 均进入分析；源视频选择偏差仍未量化。 |
| Look-elsewhere effect | NOTE | 多个噪声条件属于探索性压力测试，未进行显著性筛选。 |
| Garden of forking paths | CAUTION | 仅一个随机种子，时间对齐和图选择仍有研究者自由度。 |
| Correlation != causation | NOTE | 当前未提出因果结论。 |
| Reverse causality | NOTE | 当前不涉及方向性因果推断。 |

## 输出

- `experiences_results/surface-relation-feasibility-validation-2026-09-22/`
- `reports/progress/2026-09-22-surface-relation-e0-e1-readiness-review.md`
- `reports/progress_log.md`

## 下一步

开展 E2-lite，保留 L2 为当前求解器；暂不进入 E3、E4 或将 IRLS 写入主方法。
