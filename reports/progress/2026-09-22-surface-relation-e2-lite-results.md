# 点间表面关系 E2-lite 图比较实验

日期：2026-09-22

## 完成工作

- 比较 2D kNN、observed 3D kNN、hybrid-lite，分别使用 `k=2/3/4`。
- 对 5 个去重图配置和 12 个去重轨迹配置运行 oracle、4 档 Gaussian 噪声和 3 档 outlier 污染。
- 每个非 oracle 条件运行 20 个固定随机种子，共 15,228 条记录。
- 按源视频先聚合去重轨迹变体，再报告种子中位数、IQR、失败比例和最差结果。
- 增加逐 horizon RVE 与归一化时间轴上的梯形 AUC。
- 在本地与 AutoDL V100 实例的 CPU 环境独立复跑。

## 输出

- `scripts/surface_relation_e2_lite.py`
- `experiences_results/surface-relation-e2-lite-2026-09-22/manifest.json`
- `experiences_results/surface-relation-e2-lite-2026-09-22/graph_diagnostics.json`
- `experiences_results/surface-relation-e2-lite-2026-09-22/noise_robustness_summary.json`
- `experiences_results/surface-relation-e2-lite-2026-09-22/e2_lite_runs.jsonl`
- `experiences_results/surface-relation-e2-lite-2026-09-22/remote/`

## 图诊断

2D kNN 所有设置均连通。3D `k=2` 在一个 scissors 配置断连。hybrid-lite 在 scissors 和 watering-can 中断连，`k=3` 时 2/5 配置断连。

`k=3` 对比：

| 图 | 连通配置 | 边数范围 | `lambda_2` 范围 | KKT 条件数范围 |
|---|---:|---:|---:|---:|
| 2D kNN | 5/5 | 13--15 | 0.764--1.044 | 5.39--7.09 |
| 3D kNN | 5/5 | 15--17 | 0.838--1.975 | 4.05--8.55 |
| hybrid-lite | 3/5 | 12--14 | 0--0.676 | 2.89--10.41 |

2D `k=3` 的 5 个配置均无桥边。现有历史只有 3 帧且运动较小，2D 图边集合完全不变，无法充分检验长时图稳定性。

## 噪声结果

三视频宏平均下，2D 与 3D `k=3` 的 Gaussian 结果接近：

| 条件 | 2D k=3 | 3D k=3 | hybrid k=3 |
|---|---:|---:|---:|
| 5% Gaussian | 87.0% | 87.3% | 80.7% |
| 10% Gaussian | 74.1% | 74.2% | 61.1% |
| 20% Gaussian | 48.3% | 48.7% | 22.3% |
| 10% outlier | 71.4% | 62.9% | 64.2% |

宏平均仍掩盖饮料瓶失败。2D `k=3` 在该视频上的中位恢复率为：5% Gaussian `53.7%`、10% Gaussian `8.1%`、20% Gaussian `-84.8%`、10% outlier `12.8%`、20% outlier `-91.9%`。

## 判断

首版接口暂用 2D symmetric kNN、`k=3`。它满足当前连通性要求，复杂度最低，Gaussian 鲁棒性与 3D 图相当，并在饮料瓶 10% outlier 条件下优于 3D `k=3`。hybrid-lite 没有证据支持保留。

当前结果不支持正式选定图结构，也不支持引入 IRLS。8 点图中 outlier 比例会离散为少量边，图密度变化同时改变被污染边数，需在 32--64 query 的正式 E2 中重新评估。

## 复现

新服务器配置显示 10 CPU、72 GB 内存、V100S 32 GB；实验仅使用 CPU，运行约 25.6 秒。跨平台噪声汇总最大数值差约 `1.33e-15`，图谱条件数最大差约 `1.07e-13`。

## 下一步

保持 E3/E4 暂缓。扩充正式 E2 数据，固定预测与 pseudo-GT 时间对应，并采样每视频 32--64 个 query 后重新比较 2D、3D 与具有真实 mask/depth 边界信息的 hybrid graph。
