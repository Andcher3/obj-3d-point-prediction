# 正式 E2 模型接口闸门与执行边界

日期：2026-09-22

## 决策

公开 `MolmoMotion-4B-H3-F30` processor 实际固定要求 `points_3d_history=(3,8,3)`，因此不能把 8-query 结果改写为正式 E2 的 48-query 主结果或 64-query 密度分析。原有 32/48/64 query 数据协议保持不变。

正式 E2 增加两个独立闸门：

1. **E2-G：密集图与积分器辅助验证**。使用至少 5 个独立 Ego4D 视频、每个视频 64 个候选 query、48 点主子集、32/64 点密度分析，以及真实时间戳对齐的三维历史和 pseudo-GT。若暂时没有密集 observed-only 预测，只允许使用明确标注的受控扰动基线来测试图拓扑、关系积分、噪声传播和 query-density 稳定性；该结果不计入正式模型 E2，也不能声称验证 MolmoMotion 或 Relation Head 的密集预测能力。
2. **E2-M：模型密集接口验证**。只有找到可变 query checkpoint、经过验证的等价 adapter，或重新训练得到的模型后，才运行模型侧 48/64-query 推理。必须通过输入输出形状、点顺序置换一致性、无未来信息、时间协议和 32/48/64 smoke test。

E2-G 的 `future_3d` 来源必须在 manifest 中显式记录。若它不是实际的 observed-only baseline 预测，而是 TrackCraft3R 轨迹、受控扰动轨迹或 oracle-derived baseline，则结果必须按对应名称报告，不能称为 MolmoMotion baseline。禁止复制、插值或重复 8 点来伪造密集轨迹。

## 当前放行状态

- 正式 E2-M：阻塞。公开 checkpoint 不满足 query contract。
- 正式 E2-G：数据仍未就绪。新增视频缺少人工审核 polygon mask、三维历史、未来轨迹、pseudo-GT 和真实时间戳。
- E3、E4、IRLS：继续暂停。

## 代码 AI 可继续执行

- 审核并固定 6 个候选视频的人工 polygon mask；
- 生成 15 fps 时间戳、三帧历史、未来 9 步和 TrackCraft3R pseudo-GT；
- 为每个视频生成 64/48/32 query manifest，并运行正式数据审计；
- 调查官方代码中 8-query 限制的位置，制作不改变权重语义的 adapter smoke test；
- 若没有密集预测来源，只准备 E2-G 所需 bundle 和受控图积分输入，不运行并不合格的正式模型 E2。

## 本轮已作出的决定

- 接受 E2-G 作为正式模型 E2 之前的辅助图/积分器实验，但单独命名、单独报告，不替代模型预测结果。
- 任何密集 `future_3d` 的来源必须写入 manifest；TrackCraft3R pseudo-GT、受控扰动基线和 MolmoMotion prediction 不能混用或互称。
- 正式模型 E2-M 继续等待可变 query 模型接口；E3 仍等待 observed-only 的 `P0,H` 和模型侧证据。
