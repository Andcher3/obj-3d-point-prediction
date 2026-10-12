# 正式 E2 代码准备

日期：2026-09-22

## 完成工作

- 按冻结协议实现 15 fps 时间戳网格和最近邻匹配，容差固定为半个目标时间步长 `1/(2*15)`。
- 实现 polygon mask 读取、向内腐蚀 2 像素、三帧历史有限正深度筛选和 deterministic normalized-bbox FPS。
- 生成嵌套的 32/48/64 query 子集，主结果固定为 48 点。
- 实现只接受真实时间戳字段的正式 2D/3D `k=3` runner，包含 Gaussian、outlier、逐 horizon RVE、时间 AUC 和 32 对 48/64 的共同 query 差异记录。
- 实现 `16/48/64` query sampler smoke test 及可选 MolmoMotion processor smoke test。
- 建立 6 视频 Ego4D manifest 模板，保留 2 个既有 UUID 和 4 个新增类别槽位。

## 输出

- `scripts/surface_relation_formal_e2_prepare.py`
- `scripts/surface_relation_formal_e2_runner.py`
- `scripts/surface_relation_query_smoke.py`
- `configs/surface_relation_formal_e2_video_manifest.template.json`
- `experiences_results/surface-relation-formal-e2-query-smoke-2026-09-22.json`
- `experiences_results/surface-relation-formal-e2-current-run-check-2026-09-22.json`

## 验证结果

- sampler smoke：16、48、64 三种 query 数均成功生成。
- 当前 processor smoke 未执行，因为本地没有可用的 MolmoMotion checkpoint/repository 参数；输出明确标记为 `not_run`，没有宣称模型支持可变 query。
- 正式 runner 对现有 4 个 bottle `.npz` 全部拒绝，原因是缺少 `future_timestamps_s`、`pseudo_gt_timestamps_s` 和 `t0_seconds`。这阻止了未经验证的 `future_3d[:9]` 对齐进入正式实验。
- 现有本地和新 AutoDL 数据盘仍只有 3 个 source video 组与 8 点 query，正式 E2 数据闸门未通过。

## 当前限制

- 尚未使用 AWS 凭据下载 Ego4D 数据；当前缺少新增 3--4 个视频及人工审核的 polygon mask。
- processor smoke 需要实际 checkpoint 和代码仓库路径后才能确认模型是否接受 48/64 点。
- TrackCraft3R 输出必须重新写入真实 future/pseudo-GT 时间戳和 `t0_seconds`，旧数组不能直接转换为正式 E2 bundle。

## 下一步

补齐 5--6 个独立 Ego4D 视频、每视频 polygon mask 和时间戳对齐的 64-query bundle；完成 processor smoke 后运行正式 2D/3D `k=3` E2。
