# DAVIS TrackCraft3R 首帧坐标链诊断

- 日期：2026-09-28
- 范围：`car-turn`、`drift-turn`；只读检查现有 7 帧与 22 帧产物，未重新推理、训练或覆盖文件。
- 数值记录：`experiences_results/surface-relation-davis-frame0-direct-2026-09-28/diagnostics/coordinate_chain_car_drift.json`

## 核对结果

- frame-0 mask query 均为 `48/48` 在目标内，DA3 处理分辨率为 `392x224`，TrackCraft3R 模型分辨率为 `416x240`。保存的 `query_uv_model` 与拉伸比例计算值最大差异小于 `1.5e-5 px`。
- 原始深度结合原 K 反投影后再投影，最大误差为 `0` 或 `2.84e-14 px`。`load_npz_data` 直接返回 NPZ 的 K，不调整内参；输入 `w2c[0]` 距单位阵的最大误差低于 `8e-17`。
- 按 TrackCraft3R 实际实现，用 nearest-resized depth、缩放后的 `K_model` 和 `astype(int)` query 像素构造 `Pj(0)`。用 `K_model` 投影精确返回整数 query；换回输入分辨率后，与原 query 的 p90 仅 `1.00--1.05 px`，48 点仍全部在源 RGB mask 内。
- TrackCraft3R 解码输出 frame0 已偏离 `Pj(0)`：`car-turn` observed/full 的输入分辨率投影 p90 分别为 `26.29/33.78 px`，mask 内仅 `28/48`、`24/48`；`drift-turn` 为 `1.72/8.60 px`，mask 内为 `48/48`、`40/48`。直接使用原 K 投影，与使用 `K_model` 后按分辨率缩放的结果数值一致。

## 根因

当前偏移来自 TrackCraft3R 的解码轨迹，而非 query 选点、深度/K 反投影、内参缩放或可视化。预测器把解码结果解释为 `pixel_delta + Pj(0)`，但没有把 frame0 delta 强制为零；VAE/DiT 解码残差因此会移动首帧三维点。输出三维坐标位于 frame-0 camera 公共坐标；归一化 `w2c[t]` 将该坐标转换到各帧相机，可视化的坐标语义没有造成首帧偏移。

## 最小修复

在 query-only 输出采样后，按同一次运行内部实际整数像素的 `Pj(0)` 对每个 query 做常量三维平移：

```python
tracks[:, q] += pj0_at_integer_query[q] - tracks[0, q]
```

该操作令 frame0 精确回到 Pj 起点，保持每个 query 的全部时间位移 `tracks[t]-tracks[0]` 不变。反事实检查中两段视频均恢复 `48/48` mask 内起点，预计只消除解码首帧偏置。实施时应新增显式字段或独立版本目录，同时保留 raw tracks 和 `(48,3)` anchor delta，禁止静默覆盖旧产物。修复后仍须重新执行 observed/full 跨上下文登记、未来关系质量和图闸门；它不能解决 DA3 两种上下文之间的深度、尺度或局部几何差异。

双线性 query 采样最多消除约 1 px 的整数取样误差，不能解释或修复 `car-turn` 的 26--34 px 偏移，不作为主修复。
