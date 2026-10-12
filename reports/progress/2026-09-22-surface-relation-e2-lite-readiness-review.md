# E2-lite 结果复核与后续阶段判断

日期：2026-09-22

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-22
- Verification Status: VERIFIED
- Version Label: validation_v1

## 判断

E2-lite 的实现和结果可以接受，足以结束当前的求解器/小图原型筛选阶段，并进入正式 E2 数据扩展。2D symmetric kNN、`k=3` 可以作为当前原型图；该选择依据是连通性、无桥边、低复杂度以及与 3D kNN 近似的噪声恢复，不是统计意义上的最优性结论。

E3 仍需等待 MotionForesight 的 observed-only `P0,H`。E4 继续后置。IRLS 仍不应加入主方案。

## 复现核对

- 使用本地脚本、相同数据、`base_seed=20260922`、20 个噪声种子重新运行。
- `manifest.json`、`graph_diagnostics.json`、`noise_robustness_summary.json` 和 `e2_lite_runs.jsonl` 与原始本地结果逐字节一致。
- 本地与新 AutoDL 输出的数值差异最大为 `1.33e-15`；图谱条件数最大差异为 `1.07e-13`。两端差异属于浮点实现差异。
- 总运行数 `15,228` 的计算与脚本配置一致：12 个去重轨迹配置 × 3 种图 × 3 个 k ×（1 个 oracle + 7 个噪声条件 × 20 个种子）。

## 结果解释

### 图结构

- 2D `k=3`：5 个去重 graph configuration 全部连通，均无桥边。
- 3D `k=3`：在 Gaussian 条件下与 2D 基本相当。
- hybrid-lite `k=3`：5 个 configuration 中 2 个断连；其 Gaussian 恢复低于 2D/3D。
- 2D `k=3` 与 3D `k=3` 的主要差异很小：5% Gaussian 分别约 `87.0%` 与 `87.3%`；10% Gaussian 分别约 `74.1%` 与 `74.2%`；10% outlier 分别约 `71.4%` 与 `62.9%`。

因此，2D `k=3` 的优势主要是实现简单、拓扑稳定、没有额外深度阈值；不能据此声称 2D 在预测质量上显著优于 3D。

### 视频间异质性

2D `k=3` 的中位恢复率如下：

| 视频组 | 5% Gaussian | 10% Gaussian | 20% Gaussian | 10% outlier | 20% outlier |
|---|---:|---:|---:|---:|---:|
| drink bottle | 53.7% | 8.1% | -84.8% | 12.8% | -91.9% |
| sewing scissors | 92.9% | 85.7% | 71.7% | 78.8% | 72.4% |
| watering can | 91.9% | 83.9% | 67.6% | 83.7% | 66.0% |
| 三组宏平均 | 87.0% | 74.1% | 48.3% | 71.4% | 约 46.9% |

20 个种子只降低了噪声采样方差，没有增加独立视频数量。饮料瓶在强噪声条件下持续失败，说明当前系统的风险来自数据/轨迹条件和关系误差尺度，不能通过换图自动消除。

## 输入与指标核对

- `points_2d_at_t0` 与 `points_3d_history[-1]` 的相机投影在 21 个样本中一致，最大绝对误差约 `2.3e-5` 像素，因此 2D/3D 图比较没有发现观测时刻错配。
- `temporal_rve_auc` 已改为逐 horizon RVE 在归一化 `[0,1]` 时间轴上的梯形积分，比 E1 的平均 RVE 更合适。
- `future_3d[:9]` 与 pseudo-GT 的时间戳对应仍未验证，正式实验必须固定该协议。
- 当前只有 3 个源视频、每个样本 8 个 query；图稳定性只覆盖 3 个历史帧，且多数图的历史 Jaccard 为 1，尚不能说明长时动态下的拓扑稳定性。

## E2-lite 结论的边界

1. hybrid-lite 失败只说明当前深度跳变和局部 3D 阈值过滤过于激进，不否定使用真实 mask/depth 边界的 hybrid graph。
2. 2D/3D 的差距没有置信区间或独立视频级检验，不能使用“显著优于”。
3. 8 点图使 outlier 比例离散为 1--数条边；20% outlier 不能直接对应大规模图上的 20% 错误边。
4. 2D `k=3` 无桥边是当前 5 个小图上的观测，不是对任意 query 数量的保证。
5. 当前结果仍来自 MolmoMotion/TrackCraft3R 轨迹，尚未验证 MotionForesight 的 Relation Head 是否能学习这些关系修正。

## 建议的下一阶段

### 正式 E2（可开展）

- 扩展到 5--10 个独立视频，每个视频 32--64 query；
- 固定未来预测与 pseudo-GT 的时间映射；
- 保留 2D `k=3` 为主候选，保留 3D `k=3` 为最小对照，暂时移除 hybrid-lite 主线；
- 评估 query subset/density、图连通性、桥边比例、谱间隙、条件数和历史拓扑变化；
- 按视频和运动类型报告结果，不只报告三视频宏平均。

### E3（条件性可开展）

获得 MotionForesight observed-only `P0,H` 后，比较 zero correction、geometry-only 和 query-feature Relation Head。只有 query feature 在 held-out 视频上稳定改善，才进入 E4。

### 暂缓项目

- E4 最小训练消融：等待 E3 的可学习性证据；
- IRLS/Huber：先从真实 Relation Head 残差估计错误边比例和幅度，再决定是否测试；
- learned gate、anchor graph、global motion head：当前没有证据证明必要。

## Fallacy Scan

- Coverage: 11/11 checked
- 主要风险：宏平均掩盖视频异质性（Simpson 聚合风险）、便利样本选择偏差、20 个种子被误作独立样本、当前多种图和噪声条件带来的研究者自由度。
- 未发现因果推断、反向因果或基于显著性筛选的结论。

## 输出

- `experiences_results/surface-relation-e2-lite-2026-09-22/`
- `experiences_results/surface-relation-e2-lite-validation-2026-09-22/`
- `reports/progress/2026-09-22-surface-relation-e2-lite-readiness-review.md`
- `reports/progress_log.md`

## 下一步

进入正式 E2 数据扩展；保留 2D `k=3` 作为原型默认图，继续暂缓 E4 与 IRLS，待 MotionForesight `P0,H` 可用后开展 E3。
