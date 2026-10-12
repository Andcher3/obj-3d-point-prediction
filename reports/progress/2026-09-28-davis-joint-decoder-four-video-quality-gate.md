# DAVIS 四视频联合关系解码器质量闸门

- 日期：2026-09-28
- 协议：`configs/surface_relation_davis_joint_decoder_probe_2026-09-28.json`

## 完成工作

- 独立处理 `car-turn`、`drift-turn`、`boat`、`soccerball`。
- 每段完成调色板 mask 身份核对、48 query 采样、7 帧 observed-only TrackCraft3R、22 帧离线 query-only TrackCraft3R、7 帧 MotionForesight observed-only 推理和 t0/登记/有效性/图连通审计。
- 观测输入均为 `[1,7,3,320,576]`；预测未提供未来帧。模型输出包含 `[15,48,3]` 的 P0 和 `[15,48,16]` 的 H_output，峰值显存约 9.78 GB。
- 预览 MP4 的 24 fps 未用于时间协议；时间轴使用连续 DAVIS 源帧索引 `14--35`，评价未来为 `21--29`。

## 质量结果

| 视频 | 状态 | 有效 query | anchor p90 / max（2D bbox） | 登记 p90 / max（3D bbox） | 图组件 | bridge |
|---|---:|---:|---:|---:|---:|---:|
| `car-turn` | FAIL | 48/48 | 30.38% / 37.52% | 3.40% / 9.44% | 1 | 0 |
| `drift-turn` | FAIL | 48/48 | 55.37% / 62.17% | 25.27% / 29.42% | 1 | 0 |
| `boat` | FAIL | 48/48 | 2.34% / 4.37% | 8.44% / 41.63% | 1 | 1 |
| `soccerball` | FAIL | 48/48 | 33.56% / 43.81% | 15.81% / 22.67% | 3 | 0 |

所有视频的 2D observed mapping、有效 query 数和有限值检查通过。失败原因分别为：`car-turn` 教师 t0 anchor 超阈值；`drift-turn` 教师 anchor 与 3D 登记均超阈值；`boat` 单个 query 的登记最大误差超 `20%`；`soccerball` 教师 anchor、3D 登记和 2D 图连通性失败。

## 训练闸门

- 通过数：`0/4`；委托规则要求至少 `3/4` 才能训练。
- 未启动整视频留一训练、联合关系解码器、等参逐点对照或任何训练结果生成。
- 该结果只说明当前 DAVIS 的 TrackCraft3R/DA3 教师标签质量不足以进入训练，不能判定联合关系解码器方法本身失败。

## 磁盘与复现

- 清理前：系统盘剩余约 `1.4G`，数据盘剩余约 `3.5G`。
- 仅清理可再生 pip/base-conda 缓存后：系统盘约 `2.3G`，数据盘约 `3.5G`。
- 四视频完成后：系统盘约 `2.3G`，数据盘约 `3.2G`。
- 未删除模型权重、TrackCraft3R、MotionForesight 资产或实验结果。
- 关键 JSON 与日志已同步到 `experiences_results/surface-relation-davis-joint-decoder-probe-2026-09-28/`；关键文件本地/远端 SHA-256 一致，记录见 `sha256_manifest.json`。

## 下一步

停止本轮真实视频训练。保留四段失败审计和全部 7 帧/22 帧产物，后续需先取得可靠同坐标监督或采用独立替代监督来源。
