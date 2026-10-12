# DAVIS 修正版视频首帧点位偏移诊断

- 日期：2026-09-28
- 完成工作：对比源帧14的原始目标 mask、`query_frame0_manifest.json`、两次 TrackCraft3R 的 `query_uv_original`、`visualization_projection.npz` 和绘制脚本。
- 文件：`experiences_results/surface-relation-davis-frame0-direct-2026-09-28/{car-turn,drift-turn,boat,soccerball}/`；`scripts/build_davis_frame0_direct_projection.py`；`scripts/render_davis_frame0_direct_visualizations.py`。

## 结果

输入两次 TrackCraft3R 的 48 个 query 与源帧14的 manifest 完全一致。以原始 854×480 mask 核验，`car-turn` 的 48/48 个采样点位于目标内部。视频绘制的却是模型输出的三维首帧轨迹，经相机参数投影后的像素位置，并非输入 query 本身。

在处理图像 392×224 像素上，`car-turn` 的输出首帧相对原始 query：

| 轨迹 | 像素误差 p50 / p90 / max | 投影回源帧14后在车 mask 内 |
|---|---:|---:|
| observed-only 7 帧 | 16.82 / 26.29 / 34.53 | 28/48 |
| full 22 帧教师 | 24.33 / 33.78 / 44.75 | 24/48 |

其余序列首帧并非同幅度的偏移：`drift-turn` 的仅观测输出 p90 为 1.72 处理像素、48/48 在 mask 内；`boat` 为 4.44、42/48；`soccerball` 为 4.45、46/48。这排除了单一全局画布缩放或帧序错位作为四段共同解释。

当前可确认：旧版“从源帧20取 query 后传给第0帧”的错误已修正；本次视频首帧仍未对齐，偏差存在于三维轨迹输出与相机参数投影的组合链路中，视频编码和播放速率不会造成该偏移。现有证据无法单独区分 TrackCraft3R 首帧回归偏差与 DA3 内参/深度等投影接口不一致；视频首帧标签将“预测轨迹点”称作“query”也容易误导。

## 下一项核验

用同一 DA3 输入的深度、内参和源帧14 query 构造未经过轨迹模型的三维起点，再投影回源图；若该恒等回投影准确，比较模型输出首帧与该起点的三维残差，明确偏移发生在轨迹预测还是输入/可视化相机链。核验前不把该教师轨迹作为可靠训练标签。
