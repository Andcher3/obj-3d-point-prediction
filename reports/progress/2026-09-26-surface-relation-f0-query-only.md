# F0 query-only TrackCraft3R 获取

- 日期：2026-09-26
- 完成工作：在新双 V100 远程环境按完整 `7+15`、22 帧协议重建 `9069bd26-0139-4e88-9957-9f9489b35a91` 输入，使用 48 个正式 query 运行 TrackCraft3R query-only 推理，并用同一输入和同一 query 运行 observed-only MotionForesight。
- 输出：`experiences_results/surface-relation-f0-query-only-2026-09-26/9069bd26-0139-4e88-9957-9f9489b35a91/full22_res392_48.npz`、`f0_audit.json`、`motionforesight_observed_only_querymatched_arrays.npz`、`f0_same_input_interface_audit.json`；更新 `scripts/surface_relation_e3_motionforesight_preflight.py`，加入 query-axis attention 分块和精确 query 配对。
- 关键结果：TrackCraft3R 输出形状为 `(22, 48, 3)`，有限值比例 `1.0`，query 越界数 `0`；MotionForesight 输出 `P0=(22,48,3)`、`H_output=(22,48,16)`、`H_last_DiT=(22,48,1536)`，均为有限值。MotionForesight 只接收前 7 帧，未来 15 帧未输入。
- 配对闸门：将 TrackCraft3R frame-0 camera 坐标转换到最后观测相机坐标后，t0 点位差相对于两者联合点云包围盒对角线的 p90 为 `2.43%`、最大 `3.90%`；同输入坐标接口 pilot 通过。未来边向量相对差异 p90 为 `26.81%`，它反映 baseline 与 TrackCraft pseudo-GT 的预测差异，尚不构成监督标签失败或 Relation Head 收益证据。
- 限制：当前只有一个视频，TrackCraft 使用 `process_res=392`；没有外部真实 3D ground truth，因此本结果只证明同输入接口和坐标登记可用。
- 下一步：在相同协议下扩展独立视频的 F0 配对审计，再决定是否生成真实视频 Relation Head 训练缓存。
