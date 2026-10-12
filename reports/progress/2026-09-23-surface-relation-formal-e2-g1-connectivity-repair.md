# E2-G.1 连通修复图实验

日期：2026-09-23

## 实验定义

- 保留 observed 3D symmetric kNN `k=3` 的局部边。
- 对 3D 图的连通分量，从 observed 2D symmetric kNN `k=3` 边中选择跨分量边。
- 按 2D 边长度升序、端点编号字典序进行确定性 Kruskal 连接。
- 比较原始 2D、原始 3D 和 3D 连通修复图。
- 继续使用 32/48/64 nested query、7 条非 oracle 条件和每条件 20 个固定种子。

## 结构结果

- 5 个视频中，4 个 3D 图需要各加入 1 条连接边；1 个视频原始 3D 图已经连通。
- 修复后 5/5 图连通。
- 新增边长度约为 `18.81--26.29 px`。
- 4 条新增边全部是 bridge，说明修复图依赖少量跨分量连接，存在明显的全局拓扑脆弱性。
- 修复图历史拓扑 Jaccard 均值约为 `0.733--0.985`，最小值约为 `0.581--0.968`。

## 误差结果

- 修复图的 graph-native RVE AUC 基本贴近原始 3D 图，说明局部 3D 边的数值优势仍然存在。
- 采用 shared-union edge RVE 后，修复图相对原始 3D 的宏平均增加约 `4.8%--10.9%`，不同噪声条件均出现回升。
- 修复图仍优于 2D 图的 shared-union RVE，比例约为 `0.904--0.968`；但它通过 bridge 边恢复全局覆盖，不能据此宣称 3D 图已经稳定可靠。
- component-centered point error 在修复图上通常高于原始 3D，表明连通修复会把跨组件约束误差传入点坐标。
- 48→64 query 的 7 个非 oracle 条件中，三种图的 shared-union RVE 均未出现视频级增加；密度扩展暂未显示系统性恶化。

## 决策

- E2-G.1 解释了原始 3D RVE 优势的一部分来源：断连图减少了需要约束的跨组件关系。
- 2D `k=3` 保留为工程候选，3D `k=3` 保留为数值对照。
- 正式默认图不冻结，也不切换为 3D。
- query、时间戳、5 个 Ego4D 视频和 rigid/articulated 覆盖范围不变。
- E2-M、E3、E4、IRLS、Huber、learned gate 和复杂 hybrid 继续暂停。

## 输出

- `scripts/surface_relation_formal_e2_g1.py`
- `scripts/summarize_surface_relation_formal_e2_g1.py`
- `experiences_results/surface-relation-formal-e2-trackcraft-bundles-2026-09-23/formal_e2_g1_results.json`
- `experiences_results/surface-relation-formal-e2-trackcraft-bundles-2026-09-23/formal_e2_g1_gate_summary.json`
