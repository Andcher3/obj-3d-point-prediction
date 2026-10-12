# DAVIS official interleaved Pj0 audit

## Status

- `training_approved=false`
- 四段视频质量闸门均为 `FAIL`，未启动 MotionForesight baseline、等参逐点或跨点联合解码器训练。
- Pj0 锚定仅修复 TrackCraft3R frame0 输出契约，不代表 t0 登记或未来教师轨迹可靠。

## Protocol

- 视频：`car-turn`、`drift-turn`、`boat`、`soccerball`。
- query：源帧 14 目标 mask 内确定性 48 点。
- observed：源帧 14--20，共 7 帧，仅作非官方 observed-only 诊断，未用未来帧补足 12 帧。
- offline teacher：DA3 输入源帧 14--36，共 23 帧；TrackCraft3R 两条官方 12 帧 run：`[0,2,...,22]` 与 `[0,1,3,...,21]`；输出只保留源帧 14--35 的 `(22,48,3)`。
- 每条 run 独立执行 `tracks += Pj0 - tracks[0]`，保留 raw、anchored、run indices 和锚定偏移。
- 时间依据为 DAVIS 源帧序号；原预览 MP4 的 24 fps 未用于时间计算。本目录视频以 8 fps 编码，仅用于播放。

## Quality result

| Video | t0 UV p90 / bbox | 3D registration p90 / bbox | Graph | Future mask hits, source 21--29 | Result |
| --- | ---: | ---: | --- | --- | --- |
| car-turn | 32.72% | 18.84% | connected, 2 bridges | 31,40,32,36,30,32,22,22,22 | FAIL |
| drift-turn | 43.57% | 11.71% | connected, 0 bridges | 5,0,0,0,0,0,0,0,0 | FAIL |
| boat | 3.05% | 6.71%, max 47.21% | disconnected 35/13 | 45,40,43,41,42,39,37,39,40 | FAIL |
| soccerball | 32.22% | 31.24% | disconnected 39/9 | 3,27,21,37,36,43,38,44,42 | FAIL |

四段均为 `48/48` 有效 query，observed 与 teacher 的 frame0 anchor p90 均低于 2%。未来 mask 命中是额外教师质量诊断，当前固定 gate 尚未把该字段作为独立阈值；本批仍统一禁用真实视频训练，不能因 frame0 通过而放行。

## Visual check

- 四段 frame0 的 48 点均位于指定目标。
- `drift-turn` 到 t0 和未来已明显移出车体，后续大量点跑到画面左上，与未来 mask `0/48` 的统计一致。
- `soccerball` 在 t0 被柱遮挡，教师点明显粘在柱附近，不能视为球体身份保持。
- `car-turn` 的点群逐步分散，部分点越出车体。
- `boat` 外观上较稳定，但有效图分裂为 35/13，登记最大误差为 47.21%，仍为 FAIL。

## Coordinate diagnosis

- predictor 将每帧 DA3 点图转换到共享 frame0 camera 坐标。投影到第 j 帧使用 `W2C_j @ C2W_0`。
- `load_npz_data` 与存储 W2C 的最大差异低于 `4e-17`；点图坐标往返投影最大误差低于 `2e-13 px`。
- 两条 run 的交错索引、源帧映射和相机位姿均正确。未来 mask 失败不是 W2C/C2W 方向或二次归一造成。
- Pj0 锚定在每条 run 内严格保持时间位移；float64 推导误差约 `1e-17`，保存为 float32 后最大约 `1.2e-7`。

## Main files

- `diagnostic_summary.json`：四视频紧凑汇总。
- `<video>/quality_audit.json`：固定质量 gate。
- `<video>/pj0_anchor_comparison.json`：raw/anchored 首帧及逐帧 mask 命中。
- `<video>/future_projection_diagnostic.json`：逐 run 索引、位姿方向、旧直接 22 帧对照与点图坐标往返。
- `<video>/full22_query_tracks.npz`：官方交错 raw/anchored 轨迹与 run 元数据。
- `visualizations/<video>/`：教师、observed、frame0、t0 和关键帧检查图。

