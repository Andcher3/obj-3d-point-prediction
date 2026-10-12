# DAVIS 修正后 frame-0 query 预览

- 日期：2026-09-28
- 范围：`car-turn`、`drift-turn`、`boat`、`soccerball`

## 处理

- 按 MotionForesight/TrackCraft3R 接口，从当前 22 帧窗口的源帧14（模型输入 frame 0）目标 mask 采样 48 个 query。
- 采用与实验脚本一致的处理分辨率 `392×220`、形态学腐蚀和确定性 FPS 采样。
- 将 query 映射回原始 `854×480` RGB 帧，绘制目标边界、query 点和稳定编号。
- 仅生成静态预览，没有运行 TrackCraft3R、MotionForesight 或修改远端实验数据。

## 输出

目录：`experiences_results/surface-relation-davis-joint-decoder-probe-2026-09-28/corrected_frame0_query_previews/`

所有四段图像均为 `854×480`，48 个点均来自源帧14的目标 mask 内部。待用户确认后，再用这些 query 重建远端输入并重新执行轨迹和质量审计。
