# 点间表面关系方案的小型可行性实验计划

日期：2026-09-22

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-22
- Verification Status: PARTIALLY_VERIFIED_E0_E1
- Version Label: code_plan_v1

## 完成工作

基于当前精简方案，设计由无训练数值检查、现有轨迹离线实验、表示可学习性 probe 和小规模训练消融组成的可行性验证顺序。计划优先淘汰数学实现、图构造或 Relation Head 学习信号不足的方案，暂不加入面积 gauge、Huber/IRLS、anchor graph 和 learned gate。

## 当前条件

- 本地尚无 MotionForesight 训练复现、checkpoint 或可直接读取的预测 latent。
- 本地已有 MolmoMotion + TrackCraft3R 数值结果，单个样本包含：
  - `future_3d: (8,30,3)`；
  - `pseudo_gt: (8,9,3)`；
  - `points_3d_history: (3,8,3)`；
  - `points_2d_at_t0: (8,2)`。
- 这些结果可立即用于 solver、graph 和 oracle correction 检查，但不能替代 MotionForesight 上的最终学习实验。

## 核心研究问题

1. 受约束 L2 graph integration 是否按理论工作并可稳定反传？
2. 在真实轨迹尺度和图拓扑下，关系修正是否存在足够大的可恢复结构误差？
3. observed-only query graph 是否比简单 2D/3D 邻接更稳定？
4. MotionForesight query readout 是否包含预测 relation correction 所需的信息？
5. Relation Head + graph integration 是否优于 direct relation loss 和参数量匹配的 node residual？

## 实验 E0：求解器单元测试

### 目的

验证 component-aware uniform-gauge L2 integration 的数学实现，排除训练前的线性代数与梯度错误。

### 设置

- CPU；随机生成 `N=8/32/128` 的连通图和多分量图；
- 构造随机 `P0`、`Pgt`，令 `deltaR*=B(Pgt-P0)`；
- 测试 perfect relation、zero correction、边噪声、节点重排和断连分量；
- 用 KKT 或 null-space basis 求解，不显式求逆。

### 指标与通过条件

- perfect relation 的 component-centered RMSE `<1e-5`；
- zero correction 的 `max|P_hat-P0|<1e-7`；
- query permutation 后还原顺序的输出误差 `<1e-6`；
- 多分量解均满足 `C_comp^T U≈0`；
- backward 梯度有限，finite-difference 相对误差 `<1e-3`。

### 决策

任何一项失败都先修 solver，不进入学习实验。

## 实验 E1：现有真实轨迹上的 oracle 与噪声上限

### 目的

判断 graph integration 能修复多少内部结构误差，以及错误 edge relation 会怎样传播。

### 数据

使用本地 bottle、scissors、watering-can 等已有 `(8,30,3)` 预测与 `(8,9,3)` pseudo-GT，先对齐共同的 9 个未来时刻。prompt 变体不能当作独立样本，按视频聚合。

### 条件

1. `P0` 原预测；
2. oracle correction：`deltaR*=B(Pgt-P0)`；
3. oracle + Gaussian edge noise：GT 边长中位数的 `1%/5%/10%/20%`；
4. oracle + outlier edges：随机污染 `5%/10%/20%`；
5. 随机删边和人工断桥。

### 指标

- component-centered point error；
- relative-vector error：`mean_e ||(BP)_e-(BPgt)_e||`；
- structural-drift AUC；
- ADE/FDE；
- 条件数、连通分量数和投影残差。

### 通过条件

- oracle 条件恢复 `>95%` 的可恢复 centered structural error；
- `5%` relation noise 下仍减少 `>50%` relative-vector error；
- 无共同平移漂移；
- 没有数值爆炸或跨分量伪修正。

### 解释边界

该实验验证 solver 与误差传播，不验证 Relation Head 能否学到 correction。

## 实验 E2：query graph 小型比较

### 目的

选择首版图构造，避免在完整 mesh、测地距离和复杂置信度上提前投入。

### 图方案

1. 2D query kNN；
2. observed 3D kNN；
3. hybrid graph：2D 局部候选 + mask/depth jump 过滤 + 3D 距离过滤。

固定 `K=2/3/4`，不使用 learned gate、Voronoi area 或 future frame。

### 数据

- 立即使用现有 8-point 样本做接口检查；
- 随后从 5--10 个视频各采样 32--64 个 query，并用 TrackCraft3R 生成 pseudo-GT；
- 至少包含 rigid、articulated 和 deformable 三类运动。

### 指标

- 最大连通分量节点覆盖率；
- 度分布、桥边比例和图条件数；
- observed-history edge stability；
- E1 噪声条件下的结构恢复率；
- query density/subset 重采样后共同点结果方差。

### 决策

若 hybrid graph 未稳定优于简单 2D 或 3D kNN，则首版采用更简单的图；不为 novelty 保留复杂 graph builder。

## 实验 E3：Relation Head 可学习性 probe

### 前提

能够运行 MotionForesight 或等价的 observed-only forecasting backbone，并缓存 `P0,H`。

### 目的

验证 `Readout(H,q_i)` 是否含有超出 baseline relation 的 correction 信息。

### 数据与模型

- 50--200 个 clips，按视频划分 train/validation；
- 冻结 backbone；
- 训练三个相同规模的小 MLP：
  1. zero correction；
  2. geometry-only：`r0_t,r_init,c,time`；
  3. query-feature：在 2 的输入上加入 `f_i,t,f_j,t`。
- 目标为 `delta r*=r_gt-r0`。

### 指标与通过条件

- held-out correction MAE / Huber error；
- relation proposal RVE；
- 第 3 项相对 geometry-only 至少改善约 `10%`，且稳定超过 zero correction；
- 报告 3 个随机种子，不进行小样本显著性夸大。

若 query features 没有提供额外收益，应停止复杂 Relation Head，回到 decoded-coordinate relation loss。

## 实验 E4：最小训练消融

### 目的

回答显式 edge representation 与 graph integration 是否值得保留。

### 对照

1. frozen MotionForesight baseline；
2. query node residual head；
3. node residual + final decoded-coordinate relation loss；
4. Relation Head + component-aware L2 integration。

2--4 使用相同缓存特征、相近参数量、相同数据和训练预算。首轮不使用 Huber/IRLS、area gauge、anchor graph、learned gate 或 global motion head。

### 数据规模

- pilot：100--500 clips 或完整训练集的 1%--5%；
- 固定 3 个随机种子；
- rigid、articulated、deformable 分层报告。

### 主指标

- relative-vector error 和 structural-drift AUC；
- component-centered shape error；
- ADE/FDE 作为安全指标；
- 远期 horizon 的 outlier-point rate。

### 暂定通过条件

- 第 4 项相对 baseline 在主要结构指标上改善 `>=10%`；
- 相对参数量匹配 node residual 仍有稳定优势；
- ADE/FDE 不恶化超过约 `2%`；
- 至少两个物体运动类别和多数随机种子方向一致。

若第 3 项与第 4 项相当，采用更简单的 decoded-coordinate relation loss 或 node residual。

## 实验 E5：是否需要 Huber/IRLS

仅在 E4 通过后开展。对 held-out graph 注入 `0/5/10/20%` 错误边，比较 constrained L2 与 2--3 步 IRLS。

只有在接近真实图错误率的条件下，IRLS 明显降低结构误差且不导致桥边退化，才将其纳入正式方法。否则保留 L2。

## 推荐执行顺序

```text
E0 solver unit tests
    ↓
E1 real-track oracle/noise test
    ↓
E2 graph comparison
    ↓
获取或复现 MotionForesight P0,H
    ↓
E3 readout/learnability probe
    ↓
E4 tiny ablation
    ↓
E5 robust solver decision（可选）
```

E0--E2 可以在没有 MotionForesight checkpoint 的情况下立即开展。E3 是投入完整训练前最重要的停止条件。

## 本地输出

- `reports/progress/2026-09-22-surface-relation-feasibility-experiments.md`
- `reports/progress_log.md`

## 关键结果

最小可行性验证不应直接训练完整模型。优先证明求解器正确、真实轨迹中存在可恢复结构误差、简单 observed-only 图足够稳定，再检查 query features 是否真的能预测 correction。Relation Head 若不能超过 geometry-only 或 node residual，应及时删除。

## 下一步

实现并运行 E0；随后直接读取现有 `.npz` 轨迹完成 E1，无需等待 MotionForesight 复现。

## 2026-09-22 执行结果

已完成 E0 和 E1 的本地与 AutoDL CPU 复跑，代码为：

- `scripts/surface_relation_feasibility.py`
- `experiences_results/surface-relation-feasibility-2026-09-22/`

### E0

- `N=8/32/128`，覆盖单连通分量和多连通分量；
- component-centered RMSE 最大约 `5.2e-16`；
- zero correction 最大绝对误差为 `0`；
- 节点置换误差最大约 `1.1e-14`；
- component gauge 均值误差约 `1e-16`；
- finite-difference 梯度相对误差最大约 `3.7e-9`。

全部达到 E0 预设条件。

### E1

数据共 21 个 `.npz`，按底层视频聚合为 bottle、scissors、watering-can 三组。使用最后一个观测时刻的 3D query 构造 symmetric `k=3` kNN 图；预测取前 9 步，与 9 步 pseudo-GT 对齐。

按视频宏平均：

| 条件 | RVE | 可恢复 RVE 比例 | centered point error |
|---|---:|---:|---:|
| baseline | 0.0313 m | 0% | 0.0291 m |
| oracle | 约 0 | 100% | 约 0 |
| 5% Gaussian edge noise | 0.00418 m | 75.1% | 0.00319 m |
| 10% Gaussian edge noise | 0.00788 m | 52.6% | 0.00612 m |
| 10% outlier edges | 0.0122 m | 28.7% | 0.0101 m |
| 人工分成两半 | 0.0151 m | 59.7% | 0.0218 m |

oracle 与 5% Gaussian 条件达到 E1 预设门槛。所有积分条件的共同平移改变量均处于约 `1e-16 m` 的数值误差范围。异常边和人工断连明显降低恢复率，支持后续测试 IRLS 与图连通性诊断，但尚不能证明其优于 L2。

### 服务器环境

- 远端目录：`/root/autodl-tmp/surface-relation-feasibility/`；
- Python `3.10.8`，CPU PyTorch `2.14.0+cpu`，NumPy `2.2.6`；
- 当前无卡资源档实际显示约 `0.5 CPU / 2 GB RAM`，仍可在数秒内完成 E0--E1；
- 该地区 `/etc/network_turbo` 提示暂不支持学术加速，PyTorch CPU wheel 通过官方 CPU index 安装，其他 Python 包使用实例默认镜像。

### 解释限制

- 只有 3 个底层视频和每个视频 8 个 query，不能支撑方法有效性结论；
- prompt 变体只用于逐文件稳定性检查，统计时按底层视频聚合；
- 当前 30→9 对齐沿用已有指标行为，正式实验前仍需固定时间戳对应协议；
- E1 使用 oracle relation correction，只验证可恢复误差、求解器和噪声传播，不验证 Relation Head 可学习性。

## 2026-09-22 E2-lite 执行结果

已完成 2D kNN、observed 3D kNN 与 hybrid-lite 的 `k=2/3/4` 比较。每个 Gaussian/outlier 条件运行 20 个固定随机种子，共生成 15,228 条去重轨迹配置运行记录。

输出：

- `scripts/surface_relation_e2_lite.py`
- `experiences_results/surface-relation-e2-lite-2026-09-22/`
- `reports/progress/2026-09-22-surface-relation-e2-lite-results.md`

E2-lite 将时域指标改为：先计算每个预测时刻的 mean edge RVE，再在归一化 horizon `[0,1]` 上做梯形积分。该 `temporal_rve_auc` 与简单全时域平均不再代数相同；本次运行中二者最大绝对差约 `0.00220 m`。

图诊断结果：

- 2D kNN 的 `k=2/3/4` 在 5 个图配置中均连通；
- 3D `k=2` 在一个 scissors 配置中断连，`k=3/4` 均连通；
- hybrid-lite 在 scissors 和 watering-can 配置中断连，`k=3` 时有 2/5 个配置断连；
- 2D `k=3` 的 5 个配置均无桥边，`lambda_2` 为 `0.764--1.044`，KKT 条件数为 `5.39--7.09`；
- 2D 历史图拓扑在现有 3 帧历史中未变化；3D/hybrid 在部分配置出现边集合变化。

按源视频内去重轨迹配置求均值，再在 20 个种子上统计，2D `k=3` 的 AUC 恢复率为：

| 视频 | 5% Gaussian | 10% Gaussian | 20% Gaussian | 10% outlier | 20% outlier |
|---|---:|---:|---:|---:|---:|
| drink bottle | 53.7% | 8.1% | -84.8% | 12.8% | -91.9% |
| sewing scissors | 92.9% | 85.7% | 71.7% | 78.8% | 72.4% |
| watering can | 91.9% | 83.9% | 67.6% | 83.7% | 66.0% |

饮料瓶在强噪声下持续失败，不能用三视频宏平均替代视频级判断。2D `k=3` 在饮料瓶 10% Gaussian 的 20 个种子中保持正恢复，但 20% Gaussian 和 20% outlier 的失败率均为 100%。

当前暂定第一版使用 2D symmetric kNN、`k=3`：

- 相比 `k=2`，谱稳定性和噪声恢复更好；
- 相比 `k=4`，边数更少，异常边数量的离散增长较小；
- 与 3D `k=3` 的 Gaussian 结果接近，在饮料瓶 10% outlier 下更稳定；
- hybrid-lite 没有稳定收益，并引入断连和额外阈值。

该选择只用于后续接口原型。正式 E2 仍需 5--10 个视频、每个视频 32--64 个 query，并验证时间对应、query density/subset 和真实 mask/depth 边界过滤。
