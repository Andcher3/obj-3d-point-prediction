# MotionForesight 官方监督来源审计与后续决策

- 日期：2026-09-24
- 状态：公开监督生成链未找到；当前重建文件仅作 schema smoke test；需先做一次官方协议兼容的标签链重建审计

## 官方资料核查

1. 官方公开仓库和 `SparseTrackDataset` 提供 sparse NPZ 的读取协议、坐标变换和 validity 处理，但没有公开训练 sparse 标签生成器。
2. 官方公开预处理脚本面向 `*_user.npz` 模型输入，负责 RGB、depth、extrinsics 和 intrinsics 的组织；它没有生成未来 GT tracks 的步骤。
3. 官方论文说明训练伪标签来自完整视频上的离线 dense tracking，但该生成过程未随公开仓库完整提供。
4. 当前本地 checkpoint 配置为 `obs_frames=7`、`total_frames=22`、`future_frames=15`；当前 16 帧/9 未来步的 formal E2 审计不能直接当作官方训练标签协议。

## 对重建适配器的判断

`rebuild_motionforesight_sparse_npz.py` 已正确完成格式级 smoke test。样本包含 `tracks_xyz/query_uv/visibility/extrinsics_w2c/fx_fy_cx_cy`，但当前样本为 `T=7` 的观察窗口，并标记为 `reconstructed_from_dense_trackcraft_output`。它不能替代官方训练标签，也不能生成未来 15 步监督。

## 决策

1. 停止继续搜索同名 `*_sparse.npz` 下载文件；公开资料不足以提供它们。
2. 保留当前重建文件作为读取器/schema 测试，不进入 Relation Head 训练。
3. 由代码 AI 只实施一次有界的“官方协议兼容标签链审计”：固定当前 checkpoint/config，使用完整 `T=22` 窗口、7 帧 observed + 15 帧 future、官方 query/reference-frame 约定和 `SparseTrackDataset` 的 last-observed-camera 变换，生成 1--2 个样本并检查未来标签质量。
4. 该审计通过后，才允许在少量样本上派生 decoded-coordinate relation loss；若失败，则正式关闭当前 vector pseudo-GT 监督路线。
5. 在审计通过前，不冻结 edge head，也不启动真实训练、IRLS、Huber 或 learned gate。

公开资料与本地 checkpoint 之间存在 protocol 版本差异，后续报告必须记录仓库 commit、checkpoint config、reference frame、总帧数和未来切片，避免把 16 帧诊断协议写成官方训练协议。
