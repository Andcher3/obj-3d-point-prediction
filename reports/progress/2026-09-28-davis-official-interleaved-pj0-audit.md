# DAVIS 四视频官方交错 TrackCraft3R 与 Pj0 锚定审计

**日期：** 2026-09-28

## 完成工作

- 只读盘点远端系统盘、数据盘、GPU 和运行进程；清理残留 pip 临时目录，保留 Hugging Face 权重缓存、Conda 环境和既有实验产物。
- 为四段 DAVIS 视频生成源帧 14--36 的 DA3 23 帧输入，按官方两条 12 帧交错协议生成源帧 14--35 的 `(22,48,3)` query-only 教师轨迹。
- 保留 raw 轨迹，并对每条 run 独立执行 Pj0 frame0 常量平移锚定；生成 observed7 非官方诊断轨迹、质量审计、投影、逐帧 mask 命中与可视化。
- 核对 predictor 坐标语义、`load_npz_data`、W2C/C2W、交错索引和 pointmap 投影往返；与旧的直接 22 帧推理分开比较。
- 四段均在双 V100S 上完成，TrackCraft3R 运行时每卡约 8.1 GB，无 OOM。

## 运行问题与修复

- 首对视频完成模型推理后，对比脚本将 23 个相机位姿传给 22 帧轨迹，触发 NumPy 维度错误。模型输出未损坏；脚本修正为按轨迹长度截取位姿，仅重跑审计。
- 审计公式进一步改为显式 `W2C_j @ C2W_0`。重跑后数值结论不变。

## 结果

| Video | frame0 anchor p90 | t0 UV p90 | 3D registration p90 / max | Graph | Result |
| --- | ---: | ---: | ---: | --- | --- |
| car-turn | 1.36% | 32.72% | 18.84% / 25.37% | connected, 2 bridges | FAIL |
| drift-turn | 1.13% | 43.57% | 11.71% / 18.21% | connected, 0 bridges | FAIL |
| boat | 0.57% | 3.05% | 6.71% / 47.21% | disconnected 35/13 | FAIL |
| soccerball | 1.61% | 32.22% | 31.24% / 39.57% | disconnected 39/9 | FAIL |

- 四段均为 `48/48` 有效 query，Pj0 锚定后 frame0 为 48/48 mask 命中。
- `car-turn`、`drift-turn`、`soccerball` 的 t0 跨上下文 UV 或 3D 登记明显失败；`boat` 因单点登记最大误差和有效图断连失败。
- 未来 source 21--29 mask 命中：`car-turn` 为 `31,40,32,36,30,32,22,22,22`；`drift-turn` 为 `5,0,0,0,0,0,0,0,0`；`boat` 为 `45,40,43,41,42,39,37,39,40`；`soccerball` 为 `3,27,21,37,36,43,38,44,42`。
- 未来 mask 命中是额外教师质量诊断，当前固定 gate 未单列该阈值；本批全部标记为不可用于真实视频训练。
- predictor 输出是 frame0-camera 坐标，逐帧投影链正确；pointmap 往返误差低于 `2e-13 px`，`load_npz_data` W2C 差异低于 `4e-17`。未来偏离无法归因于索引、W2C/C2W 方向或二次归一。

## 视觉复核

- 四段 frame0 的 48 点均落在目标上。
- `drift-turn` 到 t0 和未来已移出车体，后续大量点位于画面左上，与未来 mask `0/48` 一致。
- `soccerball` 在 t0 被柱遮挡，教师点粘在柱附近，身份保持失败。
- `car-turn` 点群逐步分散，部分越出车体。
- `boat` 视觉较稳，但图断连为 35/13，登记最大误差 47.21%，不能放行。

## 磁盘记录

- 清理前：系统盘空闲约 2.43 GB，数据盘空闲约 3.33 GB。
- 仅删除约 34 KB 的残留 pip 临时文件；Conda package cache 未通过安全清理释放，不执行宽泛删除。
- 实验后：系统盘空闲约 2.43 GB，数据盘空闲约 3.28 GB。

## 输出

- `experiences_results/surface-relation-davis-official-interleaved-pj0-2026-09-28/`
- `scripts/run_trackcraft_query_interleaved23.py`
- `scripts/audit_davis_official_interleaved_pj0_bundle.py`
- `scripts/diagnose_davis_official_interleaved_future_projection.py`
- `scripts/render_davis_track_visualizations.py`

## 决定

- 四视频整视频留一训练不启动。
- MotionForesight baseline、等参逐点和跨点联合解码器没有产生训练指标。
- 当前产物作为官方交错教师失败证据和坐标链诊断保留。

