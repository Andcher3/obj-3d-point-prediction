# 固定规范图关系整合方案复审

日期：2026-09-21

## 审查对象

- D:/codex/.codex/attachments/be700083-7783-40d4-9ee6-b86bc6b577a4/已粘贴的文本.txt
- 最新方案将带 mu I 的图投影改为零质心约束下的关系修正整合。

## 总体结论

这版在方法结构上进一步收敛，解决了上一版最重要的偏置：mu ||P-P0||^2 会把所有非零结构模式都收缩，即使关系预测正确也无法完全恢复结构误差。改用 gauge-fixed integration 后，若图连通、边权有效且关系修正准确，所有零均值的 query-set 内部结构修正可以被精确恢复。

当前方案可作为小规模原型的正式候选，但还不能直接宣称最终方法。新的主要风险是：

1. 单一 1^T U=0 约束只适用于有效图连通；
2. 去掉 mu I 后变成无正则最小二乘，可能放大 relation proposal 和伪真值噪声；
3. 保持的是采样 query 集的均值，不自动等于物体质心；
4. B U 可积只表示代数一致性，不表示物理运动正确。

## 已成立的修正

### 1. mu I 的批评成立

旧投影的结构修正为：

delta P = lambda (mu I + lambda L)^(-1) L delta

其每个非零 Laplacian 模式都被乘以小于 1 的比例，因此会压制正确的低频形变、整体旋转结构和其他非局部结构变化。不能仅把这看作普通超参数问题。

### 2. 固定规范整合的数学结论成立，但需加前提

若有效加权图连通，且 delta R_hat = B delta_gt，则带 1^T U=0 的最小二乘解为：

U* = delta_gt - 1 mean(delta_gt)

因此可以精确恢复零均值的内部结构修正。若图有多个有效连通分量，单一全局约束不够；每个分量都有独立平移零空间。应使用组件指示矩阵 C_comp 施加 C_comp^T U=0，或者显式保证有效图连通。

### 3. 关系输出与最终坐标的接口已经闭合

delta R_hat -> U* -> P_hat 直接作用于最终 query 坐标，不需要把 sparse feature 写回 frozen dense decoder。

### 4. 取消独立 L_cons 可以成立

最终坐标由 P_hat 统一生成，因此 B P_hat 自动具有反对称和环一致性。L_cons 不再是实现全局可积性的必要条件；投影残差更适合作为诊断量。

## 仍需修正或限定的地方

### 1. “质心保持”应改称 query-set mean 保持

约束 1^T U=0 保持的是被采样 query points 的算术均值，不一定是物体的几何质心、质量中心或完整表面质心。query 只覆盖可见表面，采样可能不均匀，遮挡也会改变可见区域。论文应写成“保持 baseline 的 query-set mean motion”，除非引入面积或可见性权重并证明其代表物体中心。

### 2. 去掉 mu I 不等于统计上更优

旧 mu I 有偏，但具有去噪和稳定作用。新方案是无正则最小二乘：

U* = L^+ B^T W delta R_hat

伪标签噪声、错误边和 Relation Head 误差可能被放大。正确结论是：mu I 不应无条件压制所有结构模式，但是否完全移除正则化需要验证。可比较固定规范整合、保留小幅结构先验、edge-uncertainty 正则化和 robust/IRLS integration。

### 3. 固定边权不能解决错误拓扑

W_proj=C_surface 能防止 learned gate 退化为零，但不能识别深度断裂假边、错误表面三角形、关节边界连接或 mask 污染。高置信错误边仍会被 L2 投影传播。固定权重可以作为首版，但不能被表述为已经解决拓扑错误。

### 4. Relation Head 应显式接收 baseline edge relation

当前修正标签是：

delta r*_ij = r*_ij - r0_ij
r0_ij = p0_j - p0_i

若 Relation Head 输入只有 f_i,f_j,g_ij，baseline 只存在于标签定义中。建议显式输入 r0_ij、端点 p0_i,p0_j 或其编码；否则必须验证 H 是否足以恢复 P0。

### 5. “全局一致”不等于“物理正确”

B U* 来自统一坐标，只保证 graph-integrability。它不保证刚体距离、关节轴、真实形变、无自交或接触动力学。方法目标应写成结构一致性/可积性修正，不能表述为完整物理约束。

### 6. query density 只被部分缓解

整体缩放边权不会改变无正则最小二乘解，但改变 query 数量仍会改变图拓扑、节点度数、连通分量、条件数和邻居关系。固定 K 与权重归一化只能减轻问题，不能保证 query subset/density invariance。

### 7. 时间抖动风险仍存在

固定 W_proj 只消除了 gate 随时间抖动。若 delta R_hat,t 抖动，线性整合会把抖动传递到 U_t。保留 relative-velocity loss 合理，但不能称为严格时间一致性保证。

## 推荐的正式表述

P0,H = FrozenMotionForesight(V)
f_i,t = Readout(H_t, q_i)
r0_ij,t = p0_j,t - p0_i,t
delta R_hat_t = RelationHead(f_i,t, f_j,t, g_ij, r0_ij,t)

U*_t = argmin_U sum_(i,j) c_ij ||(U_j-U_i)-delta r_hat_ij,t||^2
       subject to C_comp^T U = 0

P_hat_t = P0_t + U*_t

训练时使用带有效性 mask 的：

L = L_track(P_hat,P_gt) + alpha L_rel(delta R_hat, B(P_gt-P0)) + beta L_rel-vel

第一版冻结 baseline，P0 和 relation correction target 使用 stop-gradient。

## 必须保留的对照

1. 原始 MotionForesight。
2. 仅 decoded-coordinate relation loss。
3. 参数量匹配的 generic node residual。
4. Relation Head 只做辅助监督。
5. 固定权重 gauge-fixed graph integration。
6. 带结构正则的 graph integration。
7. 2D adjacency、3D kNN、visible-surface graph。
8. 连通图与断连图。
9. query density/subset 和错误边敏感性。

## 最终判断

这版已经达到“数学上自洽、接口闭合、可进入原型实现”的程度。它比上一版更好地分离了 baseline 的 query-set mean motion、relation module 的内部结构修正和 graph integrator 的全局可积性。

但还不能称为已验证的最终方法。最需要补上的边界是：

1. 明确有效图连通和组件级 gauge；
2. 明确保持的是 query-set mean，不是自动的 object centroid；
3. 在无正则精确恢复与噪声鲁棒性之间做出实验依据；
4. 将 baseline edge relation 显式提供给 Relation Head。

该方法仍然只解决单条轨迹内的结构一致性，不解决多未来概率或全局事件模式。
