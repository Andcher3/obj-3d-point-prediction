# 固定规范关系整合方案最新复审

日期：2026-09-22

## 总体判断

这版基本通过方法学审查，已经把组件级自由度、面积加权、baseline relation 和鲁棒整合纳入候选方法。它从“图投影接口可行”推进到了“核心方程自洽、可以进入原型实现”。

“已经没有明显数学接口漏洞”的表述略强。仍需在正式方法中写清实现边界，并由实验决定无正则整合、鲁棒整合和 query 表示的取舍。

## 已正确解决的部分

1. 对多个有效连通分量使用 C_comp^T M U=0，比单一全局约束正确。
2. 将“保持质心”改为 query-set mean；面积权重可以减少像素采样密度造成的偏差，但不等同于完整物体质心或质量质心。
3. 移除全量 mu I，避免无条件压制真实低频结构模式，优先考虑 edge-level robust integration 是合理取舍。
4. 将 r0_ij,t=p0_j,t-p0_i,t 显式输入 Relation Head，使 correction target 与模型输入语义一致。
5. 区分 graph-integrability 与刚体、关节、接触、无穿透等物理正确性。
6. 在 P0 固定、权重和有效性 mask 相同的条件下，完整 relation 与 correction relation 的 relative-velocity loss 代数等价。

## 仍需修正的地方

### 1. IRLS 必须保留 gauge 约束

普通的 B^T W B U=B^T W delta R 在图拉普拉斯存在零空间时不够完整。每次 IRLS 迭代都应使用带约束的 KKT 系统：

    [B^T W B   M C_comp] [U]  = [B^T W delta R]
    [C_comp^T M  0     ] [nu] = [0]

或使用满足约束的 null-space basis。若使用 Huber-on-vector-norm，IRLS 权重还应明确为 rho'(s_e)/max(s_e,eps)。

### 2. 断连分量是能力边界

组件级 gauge 能使解唯一，但不能恢复不同可见 patch 之间的相对平移。若同一物体被图分成多个分量，方法只能分别修正分量内部结构；不能为了得到单连通图而强行添加低置信跨遮挡边。

### 3. 三类权重必须分开

- a_i：节点代表的可见表面积，用于 gauge；
- c_ij：边关系或表面邻接可信度，用于关系目标；
- m_i,t：端点时序有效性，用于监督 mask。

正式公式应区分 C_comp^T diag(a)U=0 与 c_ij m_i,t m_j,t 的 objective weight。

### 4. 固定 K 不保证 query-subset invariance

固定 K geodesic neighbors 和面积加权只能减轻 query density 影响，不能保证改变 query 子集后同一点输出不变。若采用固定 query sampling protocol，应明确声明方法依赖该协议；若要声称采样稳定性，应引入 surface anchor graph 或连续表面算子。

### 5. robust integration 不能解决系统性偏差

Huber 或 IRLS 主要降低异常边和重尾误差的影响。整片表面的 relation proposal 若存在一致方向偏差，鲁棒整合仍会保留该偏差；错误拓扑也不能仅靠鲁棒损失修复。

### 6. r0 输入仍可能出现零修正退化

Relation Head 可能输出接近零的 correction，使系统退化为 baseline。需要保留 zero-correction、frozen baseline 和 graph integration disabled 对照。

## 推荐核心形式

    P0,H = FrozenMotionForesight(V)
    f_i,t = Readout(H_t,q_i)
    r0_ij,t = p0_j,t-p0_i,t
    delta R_hat_t = RelationHead(f_i,t,f_j,t,r0_ij,t,r_ij,0,d_G,c_ij)

    U*_t = argmin_U sum_e c_e m_e,t rho(||(BU)_e-delta R_hat_e,t||)
           subject to C_comp^T diag(a)U=0

    P_hat_t = P0_t+U*_t

L2 版本是闭式受约束最小二乘；Huber/IRLS 版本必须在每次迭代保留同一组件级 gauge。

## 当前结论

该方案可以判定为：核心设计合格，可以进入原型实现；方法优势、鲁棒整合形式和 query 表示是否值得采用，仍需实验确认。

它解决的是单条轨迹内部的结构一致性和关系整合，不解决整体运动误差、多未来概率或全局事件模式。
