# F2 统一分辨率多视频复核与关系头判断

- 日期：2026-09-26
- 状态：F2 完成；Relation Head 未获得跨视频放行证据，F3 暂停。

## 协议修正

此前 F0 使用 `process_res=392`，而 query manifest 的坐标网格为长边 518。该尺度/网格不一致导致部分视频 query anchor 严重偏离。远端重新使用 `process_res=518` 生成同坐标 F0：

- 651：有效 query `47/48`，anchor p90 `1.12 px`
- 4d：`48/48`，p90 `1.09 px`
- 9069：`48/48`，p90 `0.96 px`
- cb78：`48/48`，p90 `1.09 px`
- 356：仍失败，p90 `75.97 px`，排除

所有纳入 F2 的样本均通过同坐标、时间和低误差 anchor 审计。

## 多视频 LOO F2

四个视频按视频留出，训练视频的前 7 个 future slots 训练、后 2 个验证；留出视频的 9 个 slots 全部测试。图为 2D symmetric kNN `k=3`，特征统计只使用训练视频。

跨视频测试的中位指标：

| 方法 | 去质心误差 | 绝对边向量误差 | 边相对误差 |
|---|---:|---:|---:|
| baseline | `0.1152` | `0.1100` | `1.2053` |
| geometry edge | `0.1697` | `0.1423` | `1.6217` |
| node centered | `0.2227` | `0.2345` | `2.8171` |
| H edge | `0.2138` | `0.1821` | `2.3776` |

结论：H edge、node residual 和 geometry-only 修正都没有跨视频稳定优于 baseline。训练视频验证段选择的修正门控在留出测试上没有恢复该差异。

## 同视频时间留出

统一分辨率后，四个视频分别进行前 2 slots 训练、中间 2 slots 验证、后 5 slots 测试。H edge 在个别视频降低了中心化误差或 raw ADE，但边相对误差没有稳定降低；预设五种子放行条件仅个别视频满足，整体不通过。

这表明关系头能够在单视频、有限时间段上拟合关系目标，跨视频迁移时失效。该结果更接近特征/目标的跨视频不变性不足，不能推出“点间相关运动不存在”。

## Oracle 上限

对四个纳入视频直接使用真实 pseudo-GT 边位移做组件 L2 积分，去质心和绝对边向量误差接近 `1e-16`。因此积分器和图结构在数学上没有成为瓶颈；当前瓶颈是从 observed-only `P0,H` 预测未来关系修正。

## 决策

- 关系建模的数学和工程链路保留。
- 不冻结 H edge 为有效改进，不启动 F3、真实视频 Relation Head 训练或更复杂 gate。
- 当前实验不足以否定“点间运动可建模”这一方法假设，但不支持“当前 Relation Head 设计具有跨视频预测收益”。
- 后续若继续研究，应先改变监督/特征不变性来源，再重新做视频级留出；继续微调当前 head 不具备优先级。

输出：

- [四视频 LOO F2](../../experiences_results/surface-relation-f2-loo-2026-09-26/f2_loo_res518_anchor5_selected_gate.json)
- [统一分辨率 F0](../../experiences_results/surface-relation-e3-formal-res518-2026-09-26)
- [多视频 F2 runner](../../scripts/surface_relation_f2_leave_one_video_out.py)
- [单视频 F2 runner](../../scripts/surface_relation_f2_redesigned_pilot.py)
