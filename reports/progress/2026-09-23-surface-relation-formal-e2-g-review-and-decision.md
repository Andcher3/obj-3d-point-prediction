# E2-G 结果复核与图选择流程调整

日期：2026-09-23

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent / research-progress-logger
- Origin Mode: validate
- Origin Date: 2026-09-23
- Verification Status: ANALYZED
- Version Label: formal_e2_g_review_v1

## 结果判断

### 数据和工程闸门

E2-G 的数据与工程条件已通过：5 个独立 Ego4D 视频、每个视频 64 个 query、真实时间戳、三帧历史、九步 pseudo-GT、投影误差和有效性检查均通过。dense TrackCraft3R 轨迹图的 query 轴分块没有改变小张量数值，480×832、12 帧任务已成功完成。

这证明了 dense query 数据构建和图积分器辅助实验可以运行，不证明 MolmoMotion 或 Relation Head 能够输出 dense 预测。

### 2D 与 3D 图的冲突结果

- 2D symmetric kNN `k=3`：5/5 连通、无桥边；在当前受控噪声下共享评估边集的 RVE AUC 宏平均比 3D 高约 14.56%，因此没有通过原定的 5% 保留门槛。
- observed 3D symmetric kNN `k=3`：4/5 视频断连，部分视频的组件规模为 29/19、27/21、41/7 和 44/4；全局 `lambda_2` 接近 0，不能作为同一目标的全局点间约束图直接替代 2D。
- 3D 的较低 RVE 不能单独构成选择依据。它来自受控扰动下的辅助积分实验，且断连图只在组件内施加 gauge 约束；该结果不能解释为 3D 图已经更适合最终预测模型。

### Query density

32→48 的 35 个视频—噪声条件比较没有出现主指标增加，支持保留 48 点作为主结果。48→64 尚未完成，不能据此宣称 64 点密度稳定性已经验证。

### 数据覆盖范围

当前 5 个视频只有 rigid 和 articulated 类别，没有可靠 deformable 样本。E2-G 可以作为两类运动上的辅助证据，但不能支持覆盖三类运动的泛化表述；不因此把 SSv2 直接混入主实验。

## 需要修改的流程决策

原协议中“若 2D 比 3D 恶化超过 5%则改用 3D”的自动切换规则暂时失效，因为比较用的 3D 图在多数视频上不连通。当前不应把默认图改成 3D，也不应继续宣称 2D `k=3` 已通过默认图保留闸门。

当前图选择状态改为：

> 2D `k=3` 保留为工程候选；3D `k=3` 保留为数值对照；正式默认图未冻结。

## 下一项受限实验：E2-G.1

只增加一个无学习、无未来信息的图候选，避免重新进行大规模图搜索：

1. 先构造 observed 3D symmetric kNN `k=3`；
2. 求其连通分量；
3. 从 observed 2D symmetric kNN `k=3` 的边中，按归一化 2D 距离的 Kruskal 规则加入连接不同 3D 分量的最少边，直到整体连通；
4. 保留原 3D 局部边，不使用未来帧、pseudo-GT 或学习门控。

该候选暂命名为 `3D-k3-plus-2D-component-connectors`。连接边数量、长度、是否成为 bridge、噪声恢复和历史拓扑稳定性必须单独报告。它是诊断性对照，不自动成为新方法。

E2-G.1 同时补做 48→64 的 query-density 对照，并报告三类指标：

- graph-native edge RVE：只在该图实际约束的边上评价；
- shared-union edge RVE：评价全局覆盖能力；
- component-centered point error：评价组件内几何恢复。

这样可以区分“局部边关系更准确”和“图覆盖了整个目标”两个因素，避免单一 union 指标同时承担两种解释。

## 暂不调整的部分

- 32/48/64 query 协议、48 点主结果、15 fps、三帧历史、九步未来和 Ego4D 主数据源保持不变；
- E2-M 继续等待可变 query checkpoint 或经过验证的 adapter；
- E3 继续等待 observed-only `P0,H`；
- E4、IRLS、Huber、learned gate 和更复杂 hybrid 继续暂停；
- 不把 TrackCraft3R pseudo-GT 或 controlled perturbation baseline 写成 MolmoMotion 预测结果。

## 当前放行结论

E2-G 数据构建和辅助积分实验通过；图选择未通过最终冻结条件。代码 AI 可以继续执行 E2-G.1、48→64 密度检查和连接原因诊断，完成前不进入 E2-M、E3 或 E4。

