# 关系约束图投影方案三次复审

日期：2026-09-21

## 审查对象

- `D:/codex/.codex/attachments/d5aa28a7-e775-40f5-b8ef-fd85443e5511/已粘贴的文本.txt`
- 最新方案以 MotionForesight 原始 query 预测 `P0` 为锚，Relation Head 预测 edge relation correction，再通过可微分加权图最小二乘得到最终坐标。

## 总体判断

该版本已经补全上一版最关键的 `f'_i -> p_hat_i` 接口。关系预测通过 graph projection 直接影响最终坐标，不再需要将 sparse query feature 写回 frozen dense decoder。闭式方程、反对称构造和零初始化严格退化到原始预测的推导成立。

当前版本可以作为一个完整、可实现的候选方法进入原型验证，但仍不宜直接确定为最终主方法。其最重要的限制是：图投影只能修正点集内部相对结构，严格保留每个连通分量的原始质心；同时，锚定项会收缩低频结构修正。直接 node residual baseline 仍必须保留。

## 数学核查

投影目标：

```text
min_P  mu ||P-P0||_F^2
     + lambda ||W^(1/2)(B P-R_hat)||_F^2
```

其正规方程为：

```text
(mu I + lambda B^T W B) P
= mu P0 + lambda B^T W R_hat
```

当 `mu>0`、`W>=0` 时矩阵为正定，解唯一。实现时应使用 batched Cholesky 或共轭梯度求解，不能显式求逆。

若：

```text
R_hat = B P0 + delta R
```

则最终修正满足：

```text
delta P
= lambda (mu I + lambda L)^(-1) B^T W delta R
```

其中 `L=B^T W B`。因此该模块本质上是把 edge correction 投影成一组全局一致、零均值的 node correction。

### 零初始化成立

当 `delta R=0` 时，`R_hat=B P0`，正规方程两边同时等于 `(mu I+lambda L)P0`，所以 `P_hat=P0`。该性质成立。

### 最终坐标可积，但 Relation Head 输出本身未必可积

`B P_hat` 必然来自统一坐标，因此最终输出满足反对称和环一致性。独立预测的 `R_hat` 仍可能存在 cycle component；投影只是寻找它的加权可积近似。论文中应将其称为 edge relation proposals，避免声称 Relation Head 自身已经输出全局一致结构。

## 已解决的问题

1. **输出接口闭合**：关系模块在 decoder 后修正 query coordinates，无需 sparse-to-dense latent writeback。
2. **关系直接参与结果形成**：`R_hat` 位于坐标求解目标中，不再只是辅助头。
3. **反对称性**：每条无向边只预测一个规范方向，反向取负。
4. **最终坐标全局可实现**：输出来自统一 `P_hat`。
5. **有效性加权**：relation 与 relative-velocity loss 已加入 edge confidence、两端点有效性和归一化。
6. **监督权重与作用权重分离**：`w_rel` 与投影强度分开是合理的。
7. **旋转问题表述修正**：区分物体运动旋转与坐标变换等变性。

## 仍需修正的主要问题

### 1. 投影不能修正整体平移

因为 `B 1=0`，对正规方程按节点求和可得：

```text
sum_i P_hat_i = sum_i P0_i
```

对每个图连通分量也分别成立。因此关系投影只能重整点群形状，不能修正所有点共同的平移、整体终点或质心偏差。该性质适合当前“点散开”问题，但不能据此声称同时改善宏观运动预测。

### 2. 完美 relation 也会被锚定项收缩

若 relation correction 等于真实值：

```text
delta R* = B(P_gt-P0)
```

则：

```text
delta P_hat
= lambda (mu I+lambda L)^(-1)L(P_gt-P0)
```

在 Laplacian 特征值 `ell` 上，修正比例为：

```text
lambda ell / (mu + lambda ell) < 1
```

低频、缓慢变化的结构误差会被更强地抑制。`mu/lambda` 不是普通超参数，应报告敏感性，并区分高频散点、低频弯曲和刚体旋转的效果。

### 3. 不能删掉 generic node residual baseline

graph projection 也是一种受 Laplacian 约束的 node residual：

```text
delta P = K delta R
```

它是否优于直接预测 `delta P_i` 必须由参数量相近的消融证明。将 generic residual head 从开发顺序中删除，会使“edge parameterization 是否必要”无法回答。

### 4. `w_msg` 应改为 `w_proj`

当前版本没有 latent message passing；该权重控制的是 edge relation 对坐标投影的作用强度。若后续增加 feature message passing，应另设 `w_msg`，避免再次混淆监督、坐标投影和特征传播。

### 5. 学习式投影权重可能塌缩

训练早期 relation correction 不可靠时，最容易降低最终误差的策略可能是令 `w_proj` 接近零，使输出退化为 `P0`。建议：

- 先固定 `w_proj=c_ij` 训练 relation head；
- 再开放动态 gate；
- 报告 gate 分布、与结构误差的相关性及去 gate/固定 gate 消融。

### 6. L2 投影仍会受错误边影响

Relation Head 可用 Huber loss，但投影内部仍是平方残差。一条高权错误 relation 会直接拉动多个节点。首版可保留 L2 闭式解，但必须做错误边注入和 correction clipping；若不稳，再比较 IRLS/Huber graph projection。

### 7. query density dependence 未解决

图投影对节点度数、边数和连通结构敏感。增加 query 数量会改变 Laplacian 规模与相同 `lambda` 的实际作用强度。数学结构保证 query permutation equivariance，但不保证 query subset 或 density invariance。应做 degree normalization，或按有效边数/平均度归一化，并测试同一点在不同 query 集中的输出。

### 8. 基线与 Relation Head 的梯度边界需明确

关系修正监督应写为：

```text
delta r*_ij,t
= r*_ij,t - stopgrad(p0_j,t-p0_i,t)
```

第一版建议冻结 MotionForesight，并对 `P0` stop-gradient，只训练 readout、Relation Head 与投影权重。之后再比较联合微调。否则 relation target 会随 baseline 更新而变化，且梯度可能通过预测构造的“标签”形成非预期路径。

### 9. 时间一致性仍只通过 loss 间接控制

每个未来时刻独立求解投影；若 `delta R_t` 或动态 `W_t` 抖动，最终坐标仍会产生时间 jitter。除低权 relative-velocity loss 外，应检查 gate 的时间平滑性，并报告结构误差随时域的曲线。

## 推荐的正式最小方法

### 固定基线

```text
MotionForesight -> P0, H
stopgrad(P0), freeze baseline
```

### 关系修正

```text
f_i,t = Readout(H_t, q_i)
delta R_hat = RelationHead(f_i,t, f_j,t, g_ij)
R_hat = B P0 + delta R_hat
```

### 坐标投影

```text
(mu I + lambda B^T W_proj B) P_hat
= mu P0 + lambda B^T W_proj R_hat
```

### 训练目标

```text
L = L_track(P_hat, P_gt)
  + alpha L_rel(delta R_hat, B(P_gt-P0))
  + beta L_rel-vel
```

旧版独立 `L_cons` 首轮可以删除，因为投影层已经在前向过程中耦合坐标与 relation；投影残差保留为诊断指标。

## 必须保留的消融

1. MotionForesight。
2. 只加 decoded-coordinate relation loss。
3. 参数量匹配的 generic node residual head。
4. Relation Head，只做辅助监督。
5. Relation Head + fixed-weight graph projection。
6. Relation Head + learned-weight graph projection。
7. 2D adjacency、3D kNN、visible-surface graph。
8. 冻结 baseline 与联合微调。
9. `mu/lambda`、query density、错误边比例和图断连敏感性。

## 当前结论

最新版已经具备完整输入、可微分关系模块和明确输出，适合进入小规模实现。它应被定位为“保持原 MotionForesight 质心运动的结构残差修正层”。其价值是否超过直接关系损失或普通 node residual，仍需由消融决定。

在当前项目中，该模块继续属于单条轨迹内部结构一致性的局部组件，不承担多未来概率或全局语义预测。
