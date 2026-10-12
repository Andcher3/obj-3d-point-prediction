# 正式 E2-G 数据构建与辅助实验

日期：2026-09-23

## 已完成

- 在远端 Conda 环境 `/root/autodl-tmp/conda_envs/trackcraft-e2` 安装并验证 `torch 2.5.1+cu118`，V100S CUDA 可用。
- 核对既有 8 点实验：TrackCraft3R 先生成 `(12,480,832,3)` dense track map，8 点仅在后续 MolmoMotion 输入阶段采样，未降低 TrackCraft3R 主干显存。
- 定位本次 OOM：当前环境缺少 FlashAttention/SageAttention，PyTorch SDPA math 路径尝试申请 `62.66 GiB`。
- 将 SDPA fallback 改为 query 轴分块、保留完整 key/value；小张量完整/分块结果最大差异 `3.33e-16`。`480x832`、12 帧推理成功，运行中观察到的显存最高约 `18.2 GiB`。
- 对 5 个 Ego4D 视频生成 TrackCraft3R dense 轨迹，并在 `t0` 投影平面反向匹配 64 个正式 query。5 个视频均得到 `(3,64,3)` 历史和 `(9,64,3)` pseudo-GT，有限值与有效率均为 `1.0`。
- `t0` query 到 dense 轨迹的最大匹配误差为 `1.03--2.29 px`，各视频中位数为 `0.42--0.59 px`。
- 正式数据审计通过：5 个独立视频、真实时间字段、投影一致性、来源标签、query 数和未来 9 步形状均满足 E2-G 输入要求。
- 运行 2D/3D symmetric kNN `k=3`、32/48/64 query、7 个非 oracle 条件、每条件 20 个固定种子的 E2-G runner。跨图主 RVE 使用两种图的共享边并集，避免各自边集造成比较偏差。

## 结果

- 48-query 2D 图在 5/5 视频上连通且无桥边；3D 图在 4/5 视频上出现 2 个连通分量。
- 共享评估边集下，2D 的非 oracle 主指标宏平均为 3D 的 `1.1456` 倍，超过“不差于 3D 5%”门槛；按条件均值统计，2D 在 0/5 视频上不劣于 3D。
- 32→48 query 的 35 个“视频×噪声条件”组合均未出现主指标增加，中位比值约 `0.8003`。
- E2-G 数据与工程闸门通过；2D `k=3` 默认图保留闸门未通过。3D 的断连问题使其也不能直接替代为默认图，当前图选择保持未决。

## 边界

- 当前只有 rigid 与 articulated，共 5 个视频；没有可靠 deformable 样本。
- `future_3d` 是明确标记的 controlled perturbation baseline，`pseudo_gt` 来自 TrackCraft3R；结果不代表 MolmoMotion 密集预测或 Relation Head 可学习性。
- E2-M 继续受公开 MolmoMotion 固定 8-query 接口阻塞。E3、E4、IRLS 继续暂停。

## 输出

- `experiences_results/surface-relation-formal-e2-trackcraft-bundles-2026-09-23/`
- `formal_e2_audit.json`
- `formal_e2_g_results.json`
- `formal_e2_g_gate_summary.json`
- `scripts/build_trackcraft_formal_e2_bundle.py`
- `scripts/run_trackcraft_formal_e2_remote_batch.sh`
- `scripts/trackcraft_chunk_attention_patch.py`
- `scripts/summarize_surface_relation_formal_e2.py`

## 下一步

检查 3D 断连与 2D 噪声传播差异，设计不改变预注册 query/time 协议的共同连通图对照；在此之前不冻结正式默认图，也不进入 E3。
