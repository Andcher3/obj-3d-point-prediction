# 点间关系 F2 修正版 pilot

- 日期：2026-09-26
- 状态：已完成单视频修正版 pilot；未进入 F3。

## 实验设计

- 数据：F0 同坐标链视频 `9069bd26-0139-4e88-9957-9f9489b35a91`。
- 输入：MotionForesight observed-only `p0` 和 `H_output`；TrackCraft3R query-only 轨迹仅作为伪 GT。
- 图：2D symmetric kNN，`k=3`。
- 时间划分：15 个未来槽位中，前 8 步训练，中间 2 步验证，最后 5 步测试。
- 随机种子：5 个。
- 损失：0.5 去质心坐标 MSE + 0.5 边向量 MSE。
- 修正项：head 最后一层 zero-init，使初始输出严格等于 baseline；训练特征只用训练时域拟合标准化；按验证损失选 checkpoint；各方法使用同一结构损失。
- 对照：原始 baseline、geometry+time edge adapter、`H+p0+time` node residual、`H` pair + geometry edge Relation Head。

## 测试集结果

| 方法 | raw ADE (m) | 去质心点误差 / 尺度 | 边相对误差 | 边长形状误差 |
|---|---:|---:|---:|---:|
| baseline | 0.0264 | 2.678% | 18.998% | 11.427% |
| geometry edge | 0.0443 | 4.477% | 31.046% | 14.983% |
| node residual | 0.1044 | 1.910% | 16.365% | 8.200% |
| H edge | 0.0238 | 2.251% | 21.155% | 10.603% |

## 判断

- 修正版中，`H edge` 在 5 个种子里没有一次同时满足“边相对误差优于 baseline、结构误差不增加超过 2%、raw ADE 不增加超过 2%”的放行条件。
- `H edge` 的 raw ADE 接近 baseline，但边相对误差高于 baseline；因此它没有表现出关系结构收益。
- node residual 的结构指标优于 baseline，但 raw ADE 明显变差，说明当前结构修正与整体坐标预测存在冲突，不能直接放行。
- geometry-only edge adapter 明显恶化，说明仅对现有边几何增加自由修正会破坏 baseline。
- 当前单视频时间外推结果不支持进入 F3，也不能据此宣称 Relation Head 在所有视频上失败。

## 限制与后续边界

- 只有一个完整 F0 视频，无法进行按视频划分的泛化验证。
- TrackCraft3R 不是外部真实 3D ground truth。
- 远端新增视频的 SSH 认证当前失败，尚未形成第二个同坐标 F0 bundle。
- 因此本报告完成的是“修正版 F2 单视频 pilot”，不是最终多视频 F2。

## 输出

- runner：[surface_relation_f2_redesigned_pilot.py](../../scripts/surface_relation_f2_redesigned_pilot.py)
- 结果：[f2_redesigned_pilot.json](../../experiences_results/surface-relation-f2-redesigned-2026-09-26/9069bd26-0139-4e88-9957-9f9489b35a91/f2_redesigned_pilot.json)
- 决策：F3 暂停；取得额外同坐标 F0 视频后，才进行多视频 F2。
