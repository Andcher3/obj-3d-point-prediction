# DAVIS frame-0 直接 query 四视频实验

源帧 14 的目标实例 mask 直接提供 48 个 query，TrackCraft3R 分别在源帧 14--20（仅观测）和 14--35（离线教师）运行；MotionForesight 只接收观测 7 帧。最后观测源帧 20 是输出相机坐标参考，评价未来源帧 21--29。DAVIS 真实采集 fps 未知；预览 MP4 的 8 fps 仅用于播放，旧预览的 24 fps 也没有用于时间网格。

`P0/H` 的 `[7:16]` 与未来源帧 21--29 的对应遵循 7+15 模型输出槽位协议；模型预检元数据的 `formal_e3_future_mapping_verified` 仍为 `false`，缺少独立采集时间戳验证。质量闸门已失败，未将该缓存用于训练。

## 质量闸门

| 视频 | 有效 query | 教师首帧 anchor p90 | t0 留出登记 p90 / max | 图组件 | 结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| car-turn | 48/48 | 44.05% | 13.23% / 21.95% | 1 | FAIL |
| drift-turn | 48/48 | 9.71% | 10.48% / 15.73% | 1 | FAIL |
| boat | 48/48 | 2.60% | 5.96% / 46.62% | 2（27/21） | FAIL |
| soccerball | 48/48 | 14.72% | 34.59% / 50.01% | 1 | FAIL |

百分比 anchor 相对源帧 14 目标 bbox 对角线；登记误差相对观测 t0 三维 bbox 对角线。`quality_audit.json` 保留全部阈值、2D 跨上下文偏差、bridge 和失败字段。四段均未通过预先固定的质量闸门，因此未启动整视频留一训练，暂无 baseline/逐点/联合解码器的每视频训练指标。

## 文件与核验

- 各视频目录包含原始 `observed7_query_tracks.npz`、`full22_query_tracks.npz`、`motionforesight_observed_only_arrays.npz`、`p0_h_cache_aligned.npz`、query manifest、质量审计和运行日志；教师伪标签与 MotionForesight 预测来源分别标明。
- `visualizations/<video>/` 包含 22 帧教师、7 帧观测轨迹、源帧 14 query 图和源帧 20 对比图。所有视频为 854×480。部分点出画面仍保留，不能视为质量通过。
- `remote_payload.tar.gz` 的 SHA-256 为 `9e7ea6deb0d0a27e0efd2ad3b9886f1470eee2fe698980a041ddfc57fbbb6e89`；58 个远端归档文件与本地逐字节一致。`sha256_manifest.json` 记录归档和预览文件，`verification_summary.json` 记录形状、帧数和失败闸门。
- `quality_summary.json` 是原始远端文件，其中 `training_gate_at_least_three` 是旧汇总代码字段，与冻结协议的四段全通过条件不一致。实际 `pass_count=0`，该不一致未造成训练；本地 runner 已修正为 `training_gate_all_four`。

旧的 `surface-relation-davis-joint-decoder-probe-2026-09-28/visualizations/` 使用错误 query 时间对应，已在其 README 标记作废，不得引用。
