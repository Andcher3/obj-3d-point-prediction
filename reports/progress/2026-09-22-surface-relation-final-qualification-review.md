# 点间关系整合方案合格性复核

日期：2026-09-22

## 审查对象

- `D:/codex/.codex/attachments/aae885ef-8b42-49a5-8171-92d57d84e071/已粘贴的文本.txt`
- 前序审查记录：`reports/progress/2026-09-22-surface-relation-latest-review.md`
- MotionForesight 论文与官方代码说明。

## 判断

当前核心设计有条件合格，可以停止继续扩展概念模块并进入原型实现。更细的判断为：

- 带组件级 gauge 的 L2 graph integration：数学接口合格，可先实现。
- Huber + IRLS 版本：方向合理，但求解器、时间损失和数值边界仍需正式定义。
- 作为论文最终方法或已证明优于普通 node residual：尚未合格，需要实验。

最新回应正确吸收了组件级零空间、面积权重、显式 baseline relation、断连 patch 能力边界和 query-set dependence 等意见。“已经没有明显数学接口漏洞”仍然略强。

## 已成立的部分

1. `C_comp^T M U=0` 能对每个连通分量固定独立平移自由度。
2. `r0_ij,t` 显式输入 Relation Head，使 correction target 与输入语义一致。
3. 不再将 graph-integrability 表述成刚体、关节、接触或无穿透保证。
4. 承认固定 K 邻居与面积加权不能给出 query-subset invariance；anchor graph 可作为后续选项。
5. 承认 robust integration 只能处理异常边和重尾误差，不能修复系统性 relation bias 或错误拓扑。
6. 删除独立 `L_cons` 合理，投影残差可作为诊断量。

## 仍需正式修正

### 1. IRLS 更新仍未写出带 gauge 的实际求解器

文本中的 IRLS 更新仍写成无约束正规方程。每次迭代应解：

```text
[B^T W(k) B   M C_comp] [U]  = [B^T W(k) deltaR]
[C_comp^T M       0   ] [nu]   [0]
```

若采用 vector-norm Huber，需明确：

```text
s_e = ||(B U)_e-deltaR_e||_2
w_e = c_e rho'(s_e)/max(s_e, eps)
```

并说明是固定 2--3 步的 unrolled approximate solver，还是求到收敛。标准 Huber 在全线性区间不保证严格凸；若需要唯一、平滑的层，可比较 pseudo-Huber 或在约束子空间内加入极小数值稳定项。

### 2. robust loss 与弱连接噪声放大不是同一个问题

Huber 能抑制少数大残差边，但不能消除小 `lambda_2` 对普遍小噪声的放大。若唯一桥边被 IRLS 下调，图的有效连通性还会进一步变弱。需要另外记录：

- 有效图的连通分量和 `lambda_2`/条件数；
- 最低可信边权或断连判定规则；
- L2、Huber/IRLS、弱结构先验三者的对照。

因此不能把 robust integration 表述为弱连接稳定性的完整替代方案。

### 3. 时间损失的“严格等价”只对 raw relation proposal 成立

若定义：

```text
r_prop = r0 + delta_r_hat
```

且 baseline、mask 和权重固定，则 correction-space 与 `r_prop` 上的 relative-velocity loss 代数等价。

最终坐标关系为：

```text
r_final = B(P0+U*) = r0 + B U*
```

一般有 `B U* != delta_r_hat`。采用 robust integration 时差异更明显。因此必须区分：

- proposal relation loss：训练 Relation Head；
- final projected relation/velocity loss：约束实际输出及投影引入的时间抖动。

最新文本中的 `hat r` 需要明确指向哪一个量。

### 4. 面积加权 gauge 是规范选择，不是物体质心估计

`C_comp^T diag(a) U=0` 表示每个分量在所选面积测度下保留 baseline 的平均运动。它不恢复完整物体质心，也不天然优于 uniform query mean；二者分别保留不同统计量。应根据希望由 MotionForesight 保留的宏观运动定义选择，并报告 uniform/area-weighted 对照。

### 5. visible mesh 与面积权重仍是新增预处理接口

MotionForesight 的公开接口是观测视频、目标 mask/query，以及由观测得到的 pointmap；并不直接提供可靠的表面 mesh。正式方法应增加：

```text
G, a, c = GraphBuilder(observed pointmaps, target mask, queries)
```

并明确 mesh、Voronoi 面积、测地边和置信度全部只由观测段生成。未来帧或未来伪标签只能参与训练监督，不能参与图构造或预测时权重。

### 6. 三类权重及推理可用性仍需写入公式

- `a_i`：节点面积权重，只用于 gauge；
- `c_ij`：由观测表面估计的边可信度，可进入推理时积分器；
- `m_i,t^gt`：未来伪标签有效性，只用于训练损失和归一化。

若积分器需要时间相关有效性，只能使用由观测条件预测的 `m_hat_i,t`，不能把 GT future validity 放进前向求解器。

## 推荐的第一版实现边界

```text
P0,H = FrozenMotionForesight(V_obs)
G,a,c = GraphBuilder(observed pointmaps, M_target, Q)
f_i,t = Readout(H_t,q_i)
r0_ij,t = p0_j,t-p0_i,t
deltaR_hat_t = RelationHead(f_i,t,f_j,t,r0_ij,t,r_ij,0,d_G,c_ij)

U*_t = constrained_L2_solve(deltaR_hat_t,c,a,C_comp)
P_hat_t = P0_t+U*_t
```

先使用受约束 L2 求解器，确认梯度、退化到 baseline、断连分量和 query 协议均正确，再加入 Huber/IRLS。时间损失优先定义在最终 `B P_hat` 上，同时保留 proposal relation loss训练 Relation Head。

必要对照仍包括原始 MotionForesight、decoded-coordinate relation loss、参数量匹配的 direct node residual、L2 integration、Huber integration、uniform/area gauge，以及 query density/subset 和错误边敏感性。

## 结论

方案的主逻辑已经成立，适合进入原型实现。当前剩余问题集中在求解器定义、图预处理输入、最终关系与 proposal 关系的区分，以及稳定性取舍。它仍只修正单条未来轨迹的内部结构，不能处理整体平移误差、多未来概率或全局事件选择。

本次仅完成方法审查，未实现或运行模型。
