# 点间表面关系方案从第一版到当前版的演化与复杂度审计

日期：2026-09-22

## 审查材料

- 初始讨论：`C:/Users/Andcher/Downloads/预测问题分析.md`
- 四轮修正版附件：
  - `D:/codex/.codex/attachments/72875ddc-05e1-4e43-8309-6082a217b857/已粘贴的文本.txt`
  - `D:/codex/.codex/attachments/d5aa28a7-e775-40f5-b8ef-fd85443e5511/已粘贴的文本.txt`
  - `D:/codex/.codex/attachments/be700083-7783-40d4-9ee6-b86bc6b577a4/已粘贴的文本.txt`
  - `D:/codex/.codex/attachments/aae885ef-8b42-49a5-8171-92d57d84e071/已粘贴的文本.txt`
- 前序方法审查与 MotionForesight 官方论文、代码说明。

## 总体判断

当前版本修复了第一版三个结构性缺口：query 与 latent token 的错误对应、关系特征无法进入冻结 decoder、独立边关系无法形成统一坐标。核心数据流因此比第一版更短、更可实现。

后续审查又加入面积加权 gauge、Huber/IRLS、anchor graph、谱稳定、物理能力边界等防御内容。其中只有组件级 gauge、显式 baseline relation、有效性 mask 和 observed-only 图构造应进入第一版正式方法；其余多数应作为消融、诊断或未来扩展，避免继续增加主方法复杂度。

## 第一版与当前版

### 第一版

```text
reference pointmap + mask
    -> visible surface graph
latent grid node h_i,t
    -> edge state z_ij,t
    -> relation head predicts Delta r_ij,t
    -> graph-attention message passing
    -> h'_i,t
    -> existing head + frozen dense decoder
    -> point trajectories

L = L_track + lambda_rel L_rel + lambda_cons L_cons
```

关键假设是 surface graph 可以降采样成 latent graph，一个 query 或表面区域与一个 latent token 对齐。

### 当前版

```text
Frozen MotionForesight -> P0,H
H + query coordinates  -> query readout f_i,t
observed surface       -> query graph E,c
f_i,f_j,r0             -> relation correction delta_r_hat
delta_r_hat            -> constrained graph integration U*
P_hat                  = P0 + U*
```

核心变化是把关系作用点从 latent feature message passing 移到 decoded query coordinate correction。

## 修改项与必要性

| 修改 | 第一版的问题 | 当前处理 | 必要性 | 审计结论 |
|---|---|---|---|---|
| latent graph 改为 query readout | 压缩 token 与任意 query 不一一对应 | `f_i,t=Readout(H_t,q_i)` | 必须 | 保留 |
| latent adapter 改为输出空间 refinement | 更新后的稀疏 `f'_i` 无法输入冻结 dense decoder | 先解码 `P0`，再修正 query 坐标 | 必须 | 保留；原 adapter、`gamma` 和 latent writeback 移除 |
| `Delta r` 主监督改为未来关系状态/相对 baseline correction | 差分标签噪声大，不能直接约束远期形状 | 预测 `delta r=r_gt-r0`，速度项降为辅助 | 必须 | 保留 |
| Relation Head 从辅助头改为影响最终坐标 | 原关系头可能只是坐标损失的重表达 | graph integration 将 edge proposal 变成 node correction | 条件必须 | 保留为待验证核心假设；必须与 direct relation loss、node residual 对照 |
| 独立 edge prediction 加全局整合 | 边可能违反反对称和环一致性 | 用 incidence matrix `B` 投影为统一坐标 | 必须 | 保留 |
| `mu||P-P0||^2` 改为 correction-space gauge | `mu I` 会收缩正确低频结构修正 | `P=P0+U`，只固定平移零空间 | 必须 | 保留 |
| 单一全局 gauge 改为组件级 gauge | 断连图每个分量都有独立平移零空间 | `C_comp^T U=0` | 条件必须 | 保留；若第一版只用最大可靠连通分量，可退化为单一 gauge |
| query mean 改为面积加权 gauge | query 密度影响所保持的平均量 | `C_comp^T diag(a)U=0` | 非必须 | 第一版删除；它只是规范选择，且依赖噪声 mesh/Voronoi 面积 |
| L2 整合改为 Huber + IRLS | 高置信错误边会传播到多个节点 | robust edge integration | 非必须 | 第一版删除；先用带固定可靠边的 constrained L2 |
| Relation Head 显式输入 `r0_ij,t` | 网络需从 latent 隐式恢复 baseline 已预测关系 | 将 `r0` 作为输入 | 高价值、低成本 | 保留 |
| learned gate 改为固定 `c_ij` | gate 可全部缩到零，退化回 baseline | 固定观测几何可信度 | 必须的简化 | 保留固定权重，不在第一版学习 gate |
| `L_cons` 删除 | 坐标与关系靠软损失耦合，且增加重复目标 | integration 已将关系用于坐标生成 | 应删除 | 当前处理正确 |
| surface anchor graph | query 子集变化会改变图 | 固定 anchors 后向 query 插值 | 非必须 | 移出第一版；固定 query protocol 并做稳定性测试即可 |
| global motion head | pairwise relation 无法修正整体平移 | 另预测统一位移 | 与当前问题无关 | 删除；由 MotionForesight 保留宏观运动 |
| 刚体、关节、接触、无穿透限制 | 可积性容易被夸大为物理正确 | 作为能力边界声明 | 声明必要，模块不必要 | 保留文字限定，不加入物理模块 |
| rotation equivariance / canonicalization | xyz MLP 不自动满足坐标旋转等变性 | 候选后续设计 | 非必须 | 当前统一相机坐标下先做评测，不加入主方法 |
| 冻结 baseline、stop-gradient | 联合训练会使 correction target 移动 | 第一阶段冻结 MotionForesight | 实验隔离必要 | 作为训练阶段设置，不写成方法贡献 |
| 三类权重 `a,c,m` | 面积、边可信度和标签有效性曾被混用 | 分别定义 | 记账必要 | 第一版只保留 `c` 和训练 mask `m`；删除 `a` 后更清楚 |
| 谱条件数、`lambda_2`、弱先验 | 无正则解在弱连接图上可能放大噪声 | 监测或增加稳定机制 | 诊断必要，模块未必必要 | 记录条件数和断连率；无证据前不加先验 |

## 应移出第一版的防御项

### 1. 面积加权 gauge

它不能恢复真实物体质心，只决定 edge-only correction 中无法辨识的平移规范。面积来自单目可见表面估计时还会引入新的深度和三角化误差。第一版使用每个连通分量的 uniform zero-mean gauge 即可：

```text
C_comp^T U = 0
```

### 2. Huber + IRLS

它增加迭代求解、反向传播、gauge KKT 和有效连通性处理，只针对异常边。它不能解决弱连接图对普通小噪声的谱放大，甚至可能继续降低唯一桥边的有效权重。

第一版采用：

```text
min_U sum_e c_e ||(B U)_e-delta_r_hat_e||^2
subject to C_comp^T U=0
```

先通过观测几何过滤明显错误边。只有错误边注入实验显示 L2 明显失稳时，再加入 robust solver。

### 3. surface anchor graph

它需要 anchor feature、anchor relation、query--anchor 插值和新的采样超参数。当前任务已有固定 query sampling protocol，方法只需声明依赖该协议，并报告 query density/subset sensitivity。

### 4. global motion head

当前模块针对内部结构漂移，整体平移继续由 MotionForesight 负责。加入 global branch 会混合宏观预测错误和结构修正，削弱问题归因。

### 5. learned projection gate、时空联合 projector、物理约束模块

这些都对应尚未由实验确认的次级失效模式。固定 edge confidence、逐时刻线性整合和清楚的能力边界已经足够。

## 更干净的第一版方法

### 图构造

不要求完整 mesh 或 Voronoi 面积。只从参考/最后观测 pointmap、目标 mask 和固定 queries 构造稀疏可靠图：

1. 在图像或 mask 内选择局部 query 候选；
2. 删除跨 mask 边界、深度跳变过大或三维距离异常的边；
3. 对称化并保留固定 K 个邻居；
4. 计算连通分量。

### 前向

```text
P0,H = FrozenMotionForesight(V_obs)
f_i,t = Readout(H_t,q_i)
r0_ij,t = p0_j,t-p0_i,t

delta_r_hat_ij,t = phi(
    f_i,t, f_j,t,
    r0_ij,t, r_ij,0,
    c_ij
)

U*_t = argmin_U sum_(i,j) c_ij
       ||(U_j-U_i)-delta_r_hat_ij,t||^2
       subject to C_comp^T U=0

P_hat_t = P0_t+U*_t
```

`d_G` 不是首版必需输入；邻接关系已经编码局部表面接近性，可在消融中再加入。

### 损失

```text
L = L_track(P_hat,P_gt)
  + alpha L_relation-proposal(r0+delta_r_hat, r_gt)
  + beta L_final-relative-velocity(BP_hat, BP_gt)
```

- `L_relation-proposal` 使用 edge confidence 和 GT validity mask，直接训练 Relation Head；
- 时间损失作用于最终 `B P_hat`，避免混淆 raw proposal 与投影后的实际关系；
- 不使用独立 `L_cons`；
- GT future validity 只用于训练损失，不能进入推理时求解器。

## 必要的最小对照

1. 原始 MotionForesight；
2. 只增加 decoded-coordinate relation loss；
3. 参数量匹配的 direct node residual；
4. Relation Head + constrained L2 integration；
5. 若第 4 项受错误边明显影响，再测试 Huber/IRLS。

前四项足以回答显式 edge representation 和 graph integration 是否有必要。面积 gauge、anchor graph、learned gate、global head 与物理约束无需进入首轮实验。

## 结论

从第一版到当前版，真正必要的进步集中在 query readout、输出空间关系整合、correction-space gauge 和显式 baseline relation。面积加权、robust IRLS、anchor graph 及额外运动/物理分支属于过早防御。

建议将正式第一版冻结为“固定观测图 + Relation Head + 组件级 uniform-gauge L2 integration”，停止继续增加模块，先验证它是否优于直接关系损失和普通 node residual。

本次完成方法演化与复杂度审计，未实现或运行模型。
