# DAVIS query 时间对齐审计

- 日期：2026-09-28
- 范围：`car-turn`、`drift-turn`、`boat`、`soccerball`

## 结论

当前 query 不是从原始视频第 0 帧采样，而是从协议中的 `t0=源帧20` 目标 mask 采样。这个采样步骤本身符合当前 7+15 窗口定义。

后续接口存在时间对齐错误：`build_davis_t0_frame0_mapping.py` 使用观测序列第 7 帧（源帧20）的 dense 轨迹，把 t0 query 匹配到 dense 像素后，写入 `query_sets["48"]`；`run_trackcraft_query_inference.py` 再将该坐标传给 TrackCraft3R 的默认 query 接口。该接口没有传入 query frame index，坐标会被解释为输入序列第 0 帧（源帧14）的 query。

结果是：

- 轨迹视频第 0 帧显示源帧14；
- query 坐标却代表源帧20的位置；
- query 在第 0 帧大量落到目标外部；
- 轨迹第 7 帧才接近原本定义的 t0，但教师运行仍可能有跨上下文 anchor 偏差。

## 证据

按 `query_t0_manifest.json` 中的源图坐标，在目标 mask 内的 query 数：

| 视频 | 源帧14 | 源帧20 |
|---|---:|---:|
| `car-turn` | 31/48 | 48/48 |
| `drift-turn` | 13/48 | 48/48 |
| `boat` | 40/48 | 48/48 |
| `soccerball` | 0/48 | 48/48 |

`soccerball` 最能直接说明问题：48 个 query 全部位于源帧20足球 mask 内，但源帧14一个也不在目标内。

可视化视频当前在首帧标注 `source frame 14`，所以用户看到的车外/足球旁 query 是时间错位的直接表现。

## 影响

当前四段 DAVIS 的 query-only TrackCraft3R 轨迹、质量闸门和关系解码器可行性判断不能直接作为有效证据。此前的失败同时混入了 query 起始帧错配，需在修正 query 时间接口后重新生成轨迹并重新审计。

## 修正方向

二选一：

1. 使用 TrackCraft3R 支持的 query-frame 参数，把 query 明确设为序列第 6 帧；
2. 保持 TrackCraft3R query 固定为第 0 帧，则应从源帧14 mask 采样 query，并把源帧20作为轨迹中的第 6 帧检查点。

当前协议若要保持“t0=最后观测帧、query 位于 t0”，应采用方案 1。修正前不应继续训练或比较关系解码器。
