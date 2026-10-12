# 修正版点间表面关系方案复审

日期：2026-09-21

## 复审结论

修正版明显优于上一版，已经正确修复两个核心问题：

1. 不再把 surface graph 节点与压缩 latent token 假设为天然一一对应，而是引入 query--latent readout。
2. 不再只监督 `Delta r`，改为以未来相对位置 `r_ij,t` 为主、相对速度为低权重辅助。

但方案仍不能称为“实现细节已经闭合”。最大剩余问题是 query feature graph 与原 MotionForesight 冻结 track decoder 的接口没有定义；此外，独立边关系预测还缺少图一致性约束，局部边和 learned gate 也不能自动保证长期整体形状稳定。

## 已被正确修正的部分

### Query--latent 对齐

`f_i,t = Readout(H_t, q_i)` 是合理的概念修正。它承认一个 query 坐标由多个 latent token 共同影响，并为后续图操作建立显式 query 节点。

但这只是解决了“如何读取特征”，还没有解决“如何把更新后的 query 特征送回预测器”。

### 关系监督

未来相对位置监督比只监督时间差分稳定。关系速度项降为辅助项也更符合伪标签噪声特点。

### 运动边界

将表面图解释为“邻接关系”，而非“邻点必须共享同一刚体运动”，这一点是正确的。铰链两侧保留边并不必然错误；关键是消息函数是否能表达相对运动变化。

## 仍未解决的关键问题

### 1. Query graph 与冻结 decoder 的接口断裂

原始路径大致是：

```text
track latent grid -> frozen decoder -> dense pointmap -> query readout
```

修正版变成：

```text
track latent grid -> query readout -> graph message passing -> point prediction
```

这里的 `point prediction` 没有明确实现方式：

- 如果直接由 query feature 预测坐标，就绕过了原来的冻结 decoder，需要新增 query decoder/head；
- 如果要继续使用冻结 decoder，就必须把更新后的 query feature scatter/fuse 回 latent grid，但多个 query 重叠、未采样区域和空洞的处理都未定义；
- 如果在 decoded query coordinates 上做残差修正，则不需要假设 latent 与 query 一一对应，反而是第一版最稳妥的路径。

因此，`query--latent bridge` 尚未完全解决架构问题，只解决了特征读取问题。

### 2. 独立边关系不保证全局可积性

若 Relation Head 独立预测每条边的 `r_hat_ij`，需要至少明确：

- 反对称约束：`r_hat_ji = -r_hat_ij`；
- 环路一致性：`r_ij + r_jk + r_ki = 0`；
- 边关系如何与节点坐标共享尺度和坐标系。

当前的 `L_cons` 只约束关系头与点头的对应边一致，不能自动消除不同边之间的环路冲突。关系头可能同时预测一组互相矛盾的边，而点头通过平均或折中满足坐标损失。

更安全的做法是让关系由节点位置差分得到，或者采用可积的节点势函数参数化；若保留独立 relation head，则增加 cycle/antisymmetry 检查或损失。

### 3. 局部图不一定抑制全局形状漂移

表面邻接图主要约束局部边。即使每条局部边误差不大，也可能出现：

- 长距离剪切；
- 连通区域之间的整体相对漂移；
- 局部边长保持但整体形状改变。

建议至少使用多尺度边、稀疏长程锚边或 patch/supernode 关系，并单独报告随时间和图距离增长的结构误差。

### 4. learned gate 可能学会关闭关系

`c_ij * g_ij,t` 可以降低错误边的影响，但 gate 可能退化为接近零，从而绕过消息传递；也可能趋近全一，重新变成无条件平滑。需要报告 gate 分布、边保留率和 gate 消融，不能把 learned gate 自动视为运动类型识别器。

### 5. 图节点、边与采样策略尚未闭合

如果图来自密集像素表面，而 graph node 是稀疏 query points，需要定义像素边如何映射到 query 边。如果只在 query 点之间建图，则应说明 query 采样密度、随机采样稳定性、孤立点和连通分量处理。否则每次 query 重采样可能改变图结构，导致训练和推理不一致。

### 6. 伪标签问题不能只靠 Huber 解决

Huber 可以降低异常点梯度，但不能消除系统性深度偏差、相机对齐误差和遮挡错误。边置信度应同时考虑轨迹可见性、历史跟踪稳定性和未来标签质量；仅由初始表面几何得到的 `c_ij` 不够。

### 7. Joint-Relation Transformer 的类比需要收窄

它支持“由坐标派生的关系监督也可能提供有用归纳偏置”，但其人体骨架具有已知节点语义和可靠拓扑。任意物体的可见表面图是噪声更高、拓扑不完整的 2.5D 图。因此该文献是机制先例，不是当前表面关系方案有效性的直接证据。citeturn0academia14

## 当前建议的实现顺序

先采用不破坏原 decoder 的残差方案：

```text
H -> frozen track decoder -> base query coordinates p_base
H -> query readout -> graph adapter -> delta p
p_hat = p_base + delta p
```

先比较原始模型与 decoded-coordinate relation loss。只有当结构损失稳定改善，再增加 query feature graph；独立 relation head 应作为后续消融，而不是默认最终模块。

如果一定要让 graph feature 直接影响冻结 decoder，则需要正式定义 query-to-grid 的 scatter/fusion 算子，并验证它不会改变 decoder 输入分布。

## 复审后的方法定位

当前方案已经从“概念上混淆 latent graph 与 surface graph”进步到“有合理研究假设的 query-level relational correction”。它适合进入小规模消融实验，尚不适合直接作为论文最终方法或宣称能普遍解决点群发散。

最重要的可证伪问题是：在相同参数预算和相同伪标签下，关系消息传递是否比仅加入 decoded-coordinate relation loss 带来额外结构收益，并且是否不会损害关节和可形变物体的真实运动。

