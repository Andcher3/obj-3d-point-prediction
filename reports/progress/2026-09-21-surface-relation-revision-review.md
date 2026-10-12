# 点间表面关系方案二次复审

日期：2026-09-21

## 审查对象

- `D:/codex/.codex/attachments/72875ddc-05e1-4e43-8309-6082a217b857/已粘贴的文本.txt`
- 内容为另一模型对首次审查意见的回应及修正版 `Query–Latent Readout + Surface Relation Graph + Relation Head + Message Passing`。

## 总体判断

修正版已经解决或承认三项关键问题：

1. 不再把 pixel/query point 与压缩 latent token 当作天然一一对应；
2. 将主要关系目标从逐帧差分 `Delta r` 改为未来关系状态 `r_ij,t`，差分项降为辅助监督；
3. 接受必须比较直接关系损失、独立关系头和关系消息传递。

当前方案仍未达到可直接实现状态。最重要的新缺口是 query–latent bridge 只有 readout，没有 writeback 或明确的 query decoder：更新后的稀疏 query feature `f'_i,t` 无法直接进入原有冻结的 dense track decoder。

## 对修正意见的逐项判断

### 已基本解决

- `Delta r` 不再承担主要监督，改用 `r_ij,t` 主项与低权重相对速度项。
- 承认 relation 标签不含独立观测，方法价值应来自关系归纳偏置并通过消融证明。
- 不强制对所有对象使用固定距离或相同速度约束。
- 将 last-observed camera frame 保持为当前统一坐标系是合理的。

### 只解决了一半

#### Query–latent bridge

从 `H_t` 在参考 query 坐标附近读取 `f_i,t` 是可行候选，但它不是天然的“点专属特征”。冻结 decoder 中一个输出像素受邻近多个 latent 的非线性影响；必须先验证局部 readout 是否足以重建原 decoder 的 query 输出。

更关键的是，方案没有定义 `f'_i,t -> p_hat_i,t`：

- 若直接增加新的 sparse point head，会绕过原 TrackCraft3R frozen decoder，属于新的轨迹解码器；
- 若把 `f'_i,t` 写回 `H_t`，需要 scatter/aggregation、重叠 query 冲突、背景 token 和未采样区域处理；
- 若在 decoder 后做 query residual refinement，则架构最清楚，但它是输出空间 refinement，不是 latent adapter。

#### Relation Head 的耦合

当前 message passing 使用的是共享 edge state `e_ij,t`，并未显式使用 Relation Head 输出 `r_hat_ij,t`。因此更准确的说法是“relation-supervised edge feature 参与点预测”，还不能说“预测出来的 relation 本身参与坐标形成”。

#### 软边和 learned gate

门控可以降低错误消息传播，但不能自动证明模型学到了关节、形变或真实拓扑。由于原始 MotionForesight 路径已经可完成任务，gate 可能全部收缩到接近零。需要报告 gate 分布、图边干预和去图测试。

### 尚未解决

#### 关系图的全局可实现性

若每条边独立预测向量关系，需要满足：

```text
r_ji = -r_ij
r_ij + r_jk + r_ki = 0
```

否则不同边给出的关系可能不存在任何统一点坐标解。`L_cons` 可以缓解，但在有限权重下不能保证反对称性和环一致性。应只预测规范方向的无向边，或显式加入 cycle consistency，或让关系由共享坐标候选派生。

#### 图边用于监督与用于消息传递被混为一体

一条跨关节边可能适合预测相对运动，却未必适合作为强消息传播通道。relation graph 和 message-passing graph 可以共享候选边，但应允许不同权重或不同 gating。

#### 可靠度没有进入最终损失

修正版在消息中使用 `c_ij`，但最终 `L_rel-pos`、`L_rel-vel` 和 `L_cons` 仍是无权求和。需要同时加入：

- edge confidence；
- 两端点的 validity/visibility mask；
- 相对速度项所需的相邻时刻有效性；
- 按有效边数和时间数归一化。

#### query-set dependence

若预测结果通过 sparse query graph 形成，同一个点的输出可能随其余 query 的采样数量、位置和邻接变化。必须明确这是期望行为还是副作用，并测试 query permutation、query density 和 query subset 稳定性。

#### 旋转问题被部分误解

当前不必做 object canonicalization，也不应只保留边长。但原批评关注的是坐标系变化下的等变性：整个坐标系旋转 `R` 后，理想模型应满足 `r_hat -> R r_hat`。普通 MLP 处理 xyz 分量并不自动具有该性质。该问题可以降为后续消融，但不能用“需要预测物体旋转”来反驳。

## 对引用先例的边界

- Joint-Relation Transformer 证明派生关系监督和关系—节点双流在已知人体骨架、干净关节坐标任务中可能有效；它不能证明任意表面 query、单目伪标签和压缩 VAE latent 上同样成立。
- TrajLoom 支持直接在坐标输出上加入有效性掩码、归一化后的时间速度与空间邻居关系损失；它更直接支持最小关系损失基线，不能直接证明独立 Relation Head 和 query graph adapter 的必要性。

## 推荐的可实现架构顺序

### A. 最小关系损失基线

```text
H_t -> frozen track decoder -> dense pointmap P0_t
                              -> query coordinates p0_i,t
                              -> direct edge relation loss
```

先验证结构监督本身是否减少散点和撕裂。

### B. 稀疏 query residual refinement

```text
H_t -> frozen decoder -> p0_i,t
  \-> local readout -> f_i,t -> graph/message passing -> delta p_i,t

p_hat_i,t = p0_i,t + delta p_i,t
```

这是当前最清楚的高级版本：保留原 decoder，以零初始化 residual point head 修正指定 query。Relation Head 可以从共享 edge state 输出 `r_hat`，但必须做独立消融。

### C. latent writeback adapter

```text
H_t -> query readout -> graph update -> normalized scatter -> H'_t -> frozen decoder
```

该版本最复杂，需要解决 query 集依赖、scatter 冲突和 dense field 完整性，在 A、B 有效后再考虑。

## 必要的新增验证

1. `readout probe`：冻结 MotionForesight，检查局部 readout + 小 head 能否复现 frozen decoder 在 query 上的输出。
2. `query stability`：改变 query 顺序、密度和子集，测同一点预测变化。
3. `gate usage`：报告 gate 分布，并进行图边打乱、删除和错误边注入。
4. `edge realizability`：测试反对称误差与三角环误差。
5. `weighted validity`：所有 relation loss 使用 edge confidence 与时空有效性 mask。
6. 原始 MotionForesight、直接 relation loss、query refinement、Relation Head、message passing 的逐级消融。

## 当前结论

修正版已从“存在接口错误的概念图”推进为“有明确研究假设的候选架构”。当前仍不宜称为最终方案。最适合立即实现的是 A；若 A 有效，再实现 B。现有的最终框图必须先补全 `f'_i,t` 到 `p_hat_i,t` 的接口，并决定是否保留 frozen decoder。

该方向仍只处理单条轨迹内部的局部结构一致性，属于概率多未来与全局语义方案的正交组件。
