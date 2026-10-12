# Query-only TrackCraft3R 12/16/22 Smoke

## Protocol

- 使用正式 t0 query manifest 的 48 点集合，不重新采样。
- 使用已有 observed-only dense TrackCraft 输出计算并固定 t0→frame-0 的 query 映射，再按各输入分辨率转换为 query-only 输入坐标。
- 通过 query-only runner 输出轨迹，不构造 dense 全图 query。
- 目标完整协议为 `T=22 = 7 observed + 15 future`，目标输出 `(22,48,3)`。

## 结果

| 阶段 | 结果 |
|---|---|
| 12 帧 smoke | PASS，映射版输出 `(12,48,3)`，finite fraction `1.0` |
| 16 帧 smoke | PASS，映射版输出 `(16,48,3)`，finite fraction `1.0`，OOB `0` |
| 22 帧 query-only | FAIL，未生成输出 |

22 帧失败发生在 `predictor.predict()` 的 `model_fn_wan_video` / DiT self-attention，CUDA 尝试额外分配 `13.16 GiB`，当时剩余约 `10.67 GiB`。日志中没有 query 映射错误、输入读取错误或坐标字段错误。

## 结论

- query-only 路径和固定 t0→frame-0 query 映射在 12/16 帧上通过。
- 当前 V100 资源下尚未得到完整 `(22,48,3)` TrackCraft3R 输出。
- 不能把 12/16 帧结果扩写为 22 帧，也不能用分段结果替代完整 22 帧标签。
- `relation_loss_ready=false` 和真实视频训练暂停状态保持不变。

结果文件：

- `experiences_results/motionforesight-query-only-protocol-2026-09-24/protocol_status_final.json`
- `experiences_results/motionforesight-query-only-protocol-2026-09-24/smoke12.npz`
- `experiences_results/motionforesight-query-only-protocol-2026-09-24/smoke16.npz`
- `experiences_results/motionforesight-query-only-protocol-2026-09-24/full22.log`
