# E2-G.1 结果复核与图候选层级

日期：2026-09-23

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent / research-progress-logger
- Origin Mode: validate
- Origin Date: 2026-09-23
- Verification Status: ANALYZED
- Version Label: formal_e2_g1_review_v1

## 结果判断

E2-G.1 成功解释了原始 3D 图的一部分优势来源。4 个断连视频各加入 1 条 2D 跨分量边后，图全部连通；修复图的 graph-native RVE 接近原始 3D，但 shared-union RVE 比原始 3D 回升约 5%--7%，说明原始 3D 的部分优势来自未约束跨分量关系。修复图仍比 2D shared-union RVE 低约 3%--10%，但 4 条连接边全部是 bridge，并且 component-centered error 通常高于原始 3D。

这组结果支持“3D 局部关系可能更准确”，同时显示“用少量长连接边恢复全局约束会引入拓扑脆弱性”。它还没有证明修复 3D 图在真实模型关系误差下更可靠。

48→64 query 的三种图、7 个非 oracle 条件均未出现系统性 RVE 增加，因此 48 点主结果和 64 点敏感性分析协议可以保持不变。该结果用于稳定性检查，不用于事后把主 query 数改成 64。

## 连接边轻量诊断补充

诊断脚本已改为从正式 query manifest 读取原始目标 `bbox_xyxy`，不再使用 64 个 query 点自身的包围盒。4 条跨分量连接边的目标框归一化 2D 长度为 `0.0831--0.1143`，中位数 `0.1008`；3D 长度为 `0.1540--0.4467`，相对各视频局部 3D kNN 边中位数的倍数为 `3.40--7.35`，中位数 `5.09`。

连接边受到受控异常扰动时，4 个有连接边的视频的 component-centered error 中位数分别为 `0.01934`、`0.02575`、`0.01256` 和 `0.00416`；无连接边视频保持数值零误差。异常敏感性与连接边显著长于局部边的现象一致，说明修复 3D 图的收益依赖少量拓扑脆弱的长 bridge，后续只能作为模型侧次级候选验证。

## 图候选层级调整

### 主工程候选

2D symmetric kNN `k=3` 继续作为主工程候选。理由是 5/5 连通、无桥边、无需长距离连接边，且不依赖深度图的跨组件修复。

### 次级候选

将 `3D-k3-plus-2D-component-connectors` 提升为预先登记的次级图候选，用于之后的模型侧结构消融。它不能自动成为最终默认图；修复边数量、归一化长度、3D 关系长度和桥边敏感性必须单独报告。

### 诊断对照

原始 observed 3D `k=3` 只保留为数值诊断对照。其断连状态不适合作为最终全局点间约束图，也不应直接作为 E3 的唯一图方案。

## 流程调整

- 不再继续进行无上限的 `k` 搜索、复杂 hybrid、IRLS 或 Huber 试验；E2-G.1 已足以完成当前图结构筛选阶段。
- 增加一次轻量报告补充：把 4 条连接边的像素长度归一化到目标 bounding-box 对角线，并报告其 3D 长度、局部边长度分位数及 bridge 被污染时的组件中心误差。
- 上述补充已完成；输出位于 `experiences_results/surface-relation-formal-e2-trackcraft-bundles-2026-09-23/formal_e2_g1_diagnostic.json`，并使用 20 个固定扰动种子。
- MotionForesight `P0,H` 可用后，E3 预先比较 2D 主候选与修复 3D 次候选；不要求在 E3 之前把二者压缩成唯一默认图。
- E2-M 仍等待可变 query checkpoint 或经过验证的 adapter；E3 仍等待 observed-only `P0,H`。
- 当前数据仅覆盖 rigid 与 articulated，后续结论不得扩展到 deformable 运动。

## 当前放行结论

E2-G 与 E2-G.1 已完成当前无模型图/积分器阶段。下一工作重点转向模型接口和 MotionForesight `P0,H` 获取；在此之前不再增加图结构复杂度，也不进入 E4 或 IRLS。
