# 点间表面关系 E0--E1 可行性实验

日期：2026-09-22

## 完成工作

- 实现 component-aware uniform-gauge L2 graph integration CPU 原型。
- 完成单/多分量、零修正、节点置换、gauge 和反向梯度检查。
- 对 21 个 MolmoMotion + TrackCraft3R `.npz` 运行 oracle、Gaussian 边噪声、异常边、随机删边和人工断连实验。
- 在本地 Windows/PyTorch 1.12.1 与 AutoDL Linux/PyTorch 2.14.0+cpu 独立复跑。

## 输出

- `scripts/surface_relation_feasibility.py`
- `experiences_results/surface-relation-feasibility-2026-09-22/e0_solver_unit_tests.json`
- `experiences_results/surface-relation-feasibility-2026-09-22/e1_oracle_noise.json`
- `experiences_results/surface-relation-feasibility-2026-09-22/remote/`
- `reports/progress/2026-09-22-surface-relation-feasibility-experiments.md`

## 关键结果

- E0 全部通过；服务器最大 centered RMSE 约 `5.2e-16`，最大梯度相对误差约 `3.7e-9`。
- E1 按 3 个底层视频宏平均，oracle 恢复 100% RVE，5% Gaussian 边噪声下恢复约 75.1%，达到预设门槛。
- 10% 异常边下恢复率降至约 28.7%，人工断连时约 59.7%；错误边和连通性是后续重点风险。
- 积分没有引入共同平移漂移，观测到的最大量级约为 `1e-16 m`。
- 本地与服务器的 E1 差异只在约 `1e-17` 浮点量级。

## 证据边界

当前数据只有 bottle、scissors、watering-can 三个底层视频，每个样本 8 个点。实验只支持求解器正确、现有轨迹中存在可恢复内部结构误差，以及 L2 对异常边较敏感；不支持 Relation Head 有效或完整方法优于 node residual 的结论。

## 下一步

实现 E2-lite：对 `k=2/3/4` 的 2D 与 observed 3D kNN 比较连通性、条件数、历史边稳定性和 query 子集敏感性。取得 32--64 query 的更多视频后，再运行正式 E2。
