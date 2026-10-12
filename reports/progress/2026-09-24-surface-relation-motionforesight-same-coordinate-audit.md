# MotionForesight 同坐标监督链与 Relation Head 对照审计

## 完成内容

- 核对远端 `models_pretrained/future_scene_flow/sparse_dataset.py`。
- 确认原始链路将 `Pj`、t0 anchor 和 GT tracks 一起变换到最后观测相机坐标，并共享观测 `Pj` 的 `mean/scale`。
- 确认 validity 同时依赖有限值和可见性；返回 `p0_t0_norm`、`gt_tracks_norm`、`visibility`、`pj_mean`、`pj_scale` 和 `w2c_lastobs`。
- 远端未发现原始 `*_sparse.npz` 训练标签；已有五个视频的 `p0_h_cache_aligned.npz` 仅作为审计缓存，不能视为原始监督。
- 官方 MotionForesight 仓库、发布 checkpoint zip 和 TrackCraft3R 子模块均未包含 sparse-label 生成脚本或训练标签下载入口。
- 论文公开的 A.2/3.1 只描述了生成流程：SSv2 22 帧窗口、anchor mask、DA3 深度/相机、完整片段 TrackCraft3R、最后观测相机坐标和 validity；没有公开原始 sparse 标签文件。
- 新增 `scripts/rebuild_motionforesight_sparse_npz.py`，按官方读取器要求从 dense TrackCraft `track_map`、paired user NPZ 和固定 query 生成兼容格式；该文件明确标记为 reconstructed，不宣称官方标签。
- 已在一个 9069 视频样本上生成 `T=7,N=48` 的重建 sparse 文件，字段包含 `tracks_xyz/query_uv/visibility/extrinsics_w2c/fx_fy_cx_cy`，有效率为 `1.0`。
- 从远端保留的 15 fps RGB 序列中重建了一个完整 `T=22`、`7+15` DA3 输入；观测帧为索引 `0:7`，未来帧为 `7:22`，时间和坐标字段审计通过。
- 22 帧 TrackCraft3R dense 推理在 V100 上两次退出且未生成轨迹文件；第一次有显存不足 traceback，第二次降低空间分辨率后进程级退出。该失败属于运行资源边界，不构成标签质量结论。
- 下载五个缓存到 `experiences_results/motionforesight-e3-source-cache-2026-09-24/`。
- 新增 `scripts/motionforesight_e3_same_coordinate_audit.py`，核查 decoded `P0`、伪 GT、validity、时间槽位和 2D `k=3` 边关系。
- 新增 `scripts/surface_relation_e3_head_comparison.py`，在合成 oracle 目标上比较等规模 edge Relation Head 与 node residual。

## 审计结果

五个缓存均满足形状、有限值和时间匹配检查：`P0/H` 为 9 个正式未来槽位，query 为 48 点，validity 每个 horizon 均为 `48/48`，时间误差不超过半个 15 fps 步长。缓存的坐标框架均标记为 `last_observed_camera`。

当前 `pseudo_gt_source` 仍为 `trackcraft3r_offline_future`，因此 `relation_loss_ready=false`。缓存级 edge-vector 误差和跨视频差异较大，不能据此放行真实视频训练。

合成等规模对照（10 个固定种子）：

- edge head 参数量 `1155`，node residual 参数量 `1163`；
- 两者均能优化单样本 oracle 目标；
- 最终损失比例中位数：edge `0.0188`，node `0.0245`；
- 最终绝对损失中位数：edge `0.00209`，node `0.00203`。

结果只说明两种接口和 constrained-L2 优化均可运行，未证明真实 `H_output` 含有关系信息，也未证明 edge head 优于 node residual。

## 当前闸门

- 真实视频 Relation Head：继续暂停。
- TrackCraft3R/DA3 跨上下文绝对 3D pseudo-GT：继续冻结。
- 合成/oracle 接口测试：通过工程可行性检查。
- 真实训练前仍需恢复 MotionForesight 原始同坐标训练标签或可靠的同坐标标签生成链，并从 decoded-coordinate tracks 派生 relation loss。
- 重建适配器只解决格式和字段链路，不能替代官方训练标签；在得到真实标签前不冻结 edge head，也不扩展 TrackCraft3R、IRLS 或 learned gate。
- 当前可获得的公开材料足以重建流程说明和 22 帧输入，不能获得作者原始 sparse 标签；后续应等待作者提供原始数据/生成链，或单独建立替代监督实验。
- 作者项目页只公开论文、代码仓库、基线可视化和作者联系入口，未提供 sparse 标签下载链接；已记录作者公开联系地址 `hbharad2@jhu.edu`、`yjangir@jhu.edu`，不将其视为已取得数据。

## 环境备注

本地受控脚本运行时出现 PyTorch 与 NumPy 2.x 的兼容警告；本轮计算使用纯 Torch 张量，结果文件已生成。进入模型训练前应使用兼容的远端 Conda 环境复跑。
