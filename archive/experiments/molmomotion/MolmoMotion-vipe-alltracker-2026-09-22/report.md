# MolmoMotion ViPE+AllTracker 官方管线 A/B 实验

## 实验目的

在同一饮料瓶视频、同一 `MolmoMotion-4B-H3-F30`、同一 8 个查询点和同一四条指令下，将输入管线从 DA3+TrackCraft3R 替换为官方 ViPE+AllTracker，检查近静止预测是否由坐标、尺度或跟踪分布差异造成。

## 输入与实现

- 视频：`test_video.mp4` 的 stride-2 版本，1280×720，15 fps，25 帧。
- 查询点：沿用此前实验的 8 个原图像素点。
- ViPE：官方 ViPE；输出相机位姿、metric depth、intrinsics。
- 2D 跟踪：官方 AllTracker，`max_side=512`，轨迹恢复到原视频坐标。
- 3D：官方 `vipe_to_colmap_general.py`，将 2D 轨迹回投影到 ViPE 世界坐标。
- MolmoMotion 输入：将世界坐标变换到第 3 个观测帧的相机坐标，使用 H3 的前三帧历史；未来帧只用于离线诊断，不进入推理。
- 为避免额外实例分割变量，本轮使用 `pipeline=no_vda` 和手工查询点；没有运行 SAM/VDA 后处理。

## 管线有效性

- ViPE 成功处理 25 帧，25 帧深度均可读取。
- AllTracker 输出 `(25, 8, 2)`，8 个点在全部 25 帧均可见。
- 2D 轨迹从第 0 帧到第 24 帧的平均位移为约 161.7 px，说明跟踪器没有输出静止轨迹。
- 观测 H3 历史的平均 3D 点间位移为 `0.00749 m`、`0.00670 m`；输入并非完全恒定。

## MolmoMotion 结果

| 指令 | 输出形状 | 预测路径长度 | 预测段内部最大变化 | 输出哈希 |
|---|---:|---:|---:|---|
| 放在鼠标旁 | (8,30,3) | 0.000832 m | 0 | 与其余三条相同 |
| 放在黑色水杯旁 | (8,30,3) | 0.000832 m | 0 | 与其余三条相同 |
| 放在电脑旁 | (8,30,3) | 0.000832 m | 0 | 与其余三条相同 |
| 继续拿着 | (8,30,3) | 0.000832 m | 0 | 与其余三条相同 |

四条输入的 `future_3d` SHA256 均为
`4c5f5f1c23aa98a5b9932b631a3d78a918e953e53c9c33fd940da86baa6b28e6`，生成文本 SHA256 均为
`4cc3cd262261bccfc7d1f8848aeffa4d42c2aa43519b0809880ef632495c792e`。预测段内部完全恒定，路径长度来自最后历史点到首个预测点的量化偏移。

## 与 DA3+TrackCraft3R 对照

此前 DA3+TrackCraft3R 的四条指令路径长度为 `0.000420 m`，同样四条输出完全相同、预测段内部恒定。ViPE+AllTracker 将首步偏移改为约 `0.000832 m`，约为前者两倍，但仍属于近静止，且没有恢复文本差异。

## 结论

在当前协议下，DA3+TrackCraft3R 与官方 ViPE+AllTracker 的坐标、尺度和跟踪分布差异不能单独解释饮料瓶样本的近静止预测或四条指令完全相同。官方管线替换后，跟踪输入有效、历史有可见运动，MolmoMotion 仍产生相同的零运动 token 序列。

这支持将主要问题转向公开 H3 自回归检查点对该开放桌面场景/输入分布的条件退化、H3 仅观察动作早期三帧，以及模型对目标关系文本未形成有效条件控制。由于本轮关闭 VDA/SAM，结论限定为“ViPE+AllTracker 几何与跟踪组件已完成 A/B 排除”，不等同于完整官方数据生成管线的端到端复现。

## 输出

- 远程结果：`experiences_results/MolmoMotion-vipe-alltracker-2026-09-22/output/`
- 预测可视化：`experiences_results/MolmoMotion-vipe-alltracker-2026-09-22/visualizations/comparison_2x2.mp4`
- 最后一帧对比：`experiences_results/MolmoMotion-vipe-alltracker-2026-09-22/visualizations/comparison_2x2_final.png`
- 结果指标：`experiences_results/MolmoMotion-vipe-alltracker-2026-09-22/output/molmomotion_greedy/metrics.json`
- 本地运行器：`scripts/run_molmomotion_vipe_alltracker_probe.py`
