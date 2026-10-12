# MotionForesight E3 前置接口检查

- 日期：2026-09-23
- 状态：模型接口通过；监督坐标登记阻塞；Relation Head 未训练

## 完成工作

- 在 AutoDL V100S 上建立 `motionforesight-e3` 环境：PyTorch `2.5.1+cu118`、NumPy `1.26.4`，补齐 DA3 运行依赖。
- 下载并校验 MotionForesight、TrackCraft3R、Wan2.1 与 UMT5 资产；修正 tokenizer 离线路径和 V100 query-axis attention 分块。
- 完成官方示例 observed-only smoke，并在 5 个正式 Ego4D 视频上构造连续 7 帧、15 fps 输入。
- 通过 observed-only TrackCraft3R 建立 48-query 到第一观测帧的身份映射，导出 MotionForesight `P0`、output latent `H` 和 last-DiT `H`。
- 固定 leave-one-video-out 视频级划分和缓存来源标签。
- 对剪刀视频固定输入复跑，全部缓存数组与压缩 NPZ 均逐字节一致。

## 接口结果

- 5/5 视频：`P0=(15,48,3)`、output latent `H=(15,48,16)`、last-DiT `H=(15,48,1536)`。
- 5/5 视频全部有限值，query 逆序采样误差为 `0`。
- 5/5 视频未向 MotionForesight 提供未来帧。
- 正式 9 个未来时间点与模型前 9 个槽位一一对应，最大误差不超过半个 15 fps 时间步。
- checkpoint 在 Ego4D 上的物理时间尺度校准仍未建立。

## 新发现的阻塞

E2 pseudo-GT 与 observed-only `P0,H` 来自不同的 DA3 序列上下文。共同 `t0` 相似登记后，视频级中位残差约占目标 3D bbox 对角线的 `2.8%--25.2%`；浇水壶和熨斗分别约为 `25.2%`、`16.3%`，不能直接训练 Relation Head。

浇水壶的一致 pseudo-GT 原型将中位残差从 `25.2%` 降至 `3.9%`，说明分离 observed/future DA3 并在共同 `t0` 接入坐标的方向有效；但 48 个 query 中仍有 14 个超过 bbox 对角线的 10%，最大 dense anchor 误差约 `67.7 px`。当前问题集中在 TrackCraft3R 的变长序列 dense identity，而非 MotionForesight 的 48-query readout。

## 输出

- `scripts/surface_relation_e3_prepare_formal_input.py`
- `scripts/surface_relation_e3_formal_preflight.py`
- `scripts/run_surface_relation_e3_formal_preflight_remote.sh`
- `scripts/surface_relation_e3_prepare_consistent_pseudogt_input.py`
- `scripts/build_surface_relation_e3_consistent_pseudogt_bundle.py`
- `scripts/summarize_surface_relation_e3_preflight.py`
- `configs/surface_relation_e3_video_splits_2026-09-23.json`
- `experiences_results/surface-relation-e3-preflight-2026-09-23/`

## 下一步

使用 past-only 12 帧 TrackCraft3R 窗口建立 query 身份，以训练长度 12 帧构造由 3 帧 observed 和 9 帧 offline future 组成的一致 pseudo-GT。该登记审计通过后，才生成 Relation Head 训练缓存。
