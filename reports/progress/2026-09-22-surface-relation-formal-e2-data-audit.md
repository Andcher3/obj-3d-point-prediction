# 正式 E2 数据扩展审计

日期：2026-09-22

## 完成工作

- 核对 E2-lite readiness 记录与本地固定种子输出。
- 检查新 AutoDL 数据盘目录，确认仍只有 21 个 `.npz`、3 个源视频组和 8 点 query。
- 实现正式 E2 输入审计脚本，明确 5--10 个独立视频、32--64 query、时间元数据和 2D/3D 投影一致性要求。

## 输出

- `scripts/surface_relation_formal_e2_audit.py`
- `experiences_results/surface-relation-formal-e2-audit-2026-09-22.json`

## 审计结果

- `formal_e2_ready=false`；
- 当前独立源视频数：`3`，低于最低 `5`；
- 当前每个样本 query 数：`8`，低于最低 `32`；
- `points_2d_at_t0` 与 observed 3D 投影一致性可检查，但未来预测与 pseudo-GT 时间对应仍缺少显式元数据；
- 新服务器数据盘未发现额外原视频、query 文件或 TrackCraft 输出。

## 协议决定

- 正式 E2 主候选：2D symmetric kNN，`k=3`；
- 最小对照：observed 3D symmetric kNN，`k=3`；
- hybrid 暂不进入主线，等待真实 mask/depth 边界数据；
- 每个源视频保留 32--64 query、观测历史、future/pseudo-GT 时间索引，并报告 query subset、query density、桥边、谱间隙、条件数与历史拓扑变化。

## 下一步

补齐至少 2 个独立源视频，并从每个视频生成 32--64 个目标表面 query 及其观测历史、未来 TrackCraft pseudo-GT 和时间索引；完成后直接运行正式 E2，而不把当前 21 个文件扩写成伪独立样本。
