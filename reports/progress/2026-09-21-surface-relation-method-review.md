# 点间表面关系预测方案审查

日期：2026-09-21

## 审查材料

- 用户提供的 `C:/Users/Andcher/Downloads/预测问题分析.md`。
- MotionForesight、TrackCraft3R、TrajLoom、MolmoMotion 的论文与官方项目说明。
- 本地 `reports/progress/2026-09-18-motionforesight-demo-failure-analysis.md`。

## 总体判断

显式建模同一目标上邻近点的相对运动，能够直接针对长时域中的点群撕裂、剪切和离群点。当前讨论形成的 `Visible Surface Graph + relation supervision + message passing` 可作为研究假设，但尚不是自洽、可直接实现的完整方法。最严重的问题是把压缩后的 track latent token 与参考帧表面节点视为天然逐点对应；另外，单独 Relation Head 的标签完全由坐标标签计算，可能只是在复杂形式下重复直接关系损失。

## 主要问题

### 1. query、像素与 latent token 的对应被过度简化

MotionForesight 在采样 query points 上计算解码坐标损失，但内部预测的是空间压缩后的 residual-track latent，随后由冻结 decoder 恢复完整 tracking pointmap。TrackCraft3R 的 latent 具有参考帧锚定的空间布局，却处于约 `1/16` 空间分辨率；一个 token 对应一个带重叠感受野的区域，不对应任意采样 query point。

因此不能只把像素级表面图“降采样”后认定每个图节点对应一个 track token。若要在 latent 上传播关系，需要明确：

- query-to-token 的可微分赋值矩阵或 ROI/bilinear sampling；
- 一个 token 覆盖多个前景点、背景点或深度层时的聚合规则；
- 图边合并、前景占比、无有效表面点 token 的处理；
- adapter 输出怎样经过冻结 decoder 后仍保持指定 query 的关系。

### 2. Relation Head 没有增加独立监督信息

讨论中的关系标签

```text
Δr_ij,t = (p_j,t - p_i,t) - (p_j,t-1 - p_i,t-1)
```

完全由点坐标标签计算。`L_rel` 是坐标误差的线性组合，`L_cons` 又要求 Relation Head 与坐标输出隐含的关系一致。若使用平方损失并对 relation 输出消元，二者近似退化为一个加权的直接坐标关系损失；它可以改变优化偏置，但没有加入新的物理观测。

Relation Head 只有在其 edge feature 确实反向参与点预测时，才提供额外的结构归纳偏置。即便如此，也必须与不增加独立 head 的 decoded-coordinate relation loss 做消融。

### 3. 当前 `Δr` 目标不直接约束长期形状

该量表示邻点相对速度，适合抑制局部运动不一致，但逐帧小误差会累积。只匹配 `Δr` 仍可能在远期产生较大的相对位置漂移。更直接的主项应比较每个未来时刻的相对向量：

```text
(p_hat_j,t - p_hat_i,t) - (p_gt_j,t - p_gt_i,t)
```

或比较相对参考帧的累计形变。逐帧相对速度可以作为较小权重的辅助项。

### 4. 差分关系会放大伪标签噪声

`Δr` 同时包含四个坐标项，容易放大深度抖动、跟踪漂移和相机位姿误差。MotionForesight 的未来轨迹本身来自离线伪标签；MolmoMotion 使用点间关系主要是过滤和加权不可靠轨迹，这并不能直接证明同类差分标签适合成为 forecasting loss。训练前应先测量边关系标签的噪声随预测时域和深度方向的增长。

### 5. 固定表面图可能跨越错误运动边界

参考 mask 与单帧 pointmap 只能提供初始可见的 2.5D 表面。深度断裂、薄物体、自遮挡、折叠表面和相邻但不同部件可能造成错误三角形。固定局部邻接也无法自动区分刚体内部、关节边界和可形变区域。

表面图应使用软边权和边可靠度，至少结合：mask 边界、深度跳变、法向差异、观测期相对运动一致性和几何置信度。关节或形变对象不能统一施加强刚体距离保持。

### 6. 坐标性质没有完全说明

相对向量消除了全局平移，但会随坐标轴旋转；它不是旋转不变量。MotionForesight 默认使用最后一个观测帧的相机坐标系，因此同一样本内可以一致监督，但跨视角泛化不能仅靠“平移不变”论证。可比较全局向量、局部参考坐标和仅使用边长/角度的表示。

### 7. `gamma=0` 初始化有优化细节

若 `h' = h + gamma * m_theta` 且 `gamma` 精确为零，第一步中 `m_theta` 参数梯度也为零，先只有 `gamma` 更新。该设计可以保持初始网络不变，但会延迟 adapter 学习。可比较小正值 `gamma` 与零初始化 adapter 输出投影。

### 8. 现象成因尚未隔离

点群散开可能来自缺少关系归纳偏置，也可能来自未来伪标签噪声、冻结 decoder、空间压缩、深度方向误差、单未来回归和真实未来歧义。仅凭定性视频不能把问题归因于缺少 point relation。应先渲染并统计训练伪标签本身的结构漂移。

## 建议的最小可行版本

第一阶段保留原 MotionForesight 架构，不增加 Relation Head 或 latent adapter，只在解码后的 query coordinates 上加入可靠边的直接关系损失：

```text
L = L_track
  + lambda_abs * L_edge-relative-position
  + lambda_vel * L_edge-relative-velocity
```

其中 `lambda_vel` 较小，边使用 robust loss、有效性 mask 和置信度权重。这个版本能直接验证“结构损失是否减少点群撕裂”，且不需要假设 query 与 latent token 一一对应。

若该版本有效，再依次测试：

1. 独立 Relation Head；
2. Relation Head + consistency；
3. query feature sampling 后的 graph/message-passing adapter；
4. latent supernode graph。

## 必要消融与指标

- 原始 MotionForesight。
- 直接 decoded-coordinate relation loss。
- Relation Head，不回传关系特征。
- Relation Head + consistency。
- graph adapter。
- 2D 邻接、3D kNN、表面测地邻接。
- GT graph、估计图、随机扰动图。
- rigid、articulated、deformable 分层。
- ADE/FDE 之外报告：随时域变化的边相对向量误差、边长畸变、局部角度畸变、刚体对齐后形状误差、离群点率和结构 drift 曲线。

## 研究定位

该方向解决的是单条预测内部的空间一致性，不解决多未来概率、模式概率或全局语义选择。它更适合作为 MotionForesight 基线的局部结构增强。图消息传递、关系监督和一致性损失本身已有广泛先例，贡献强度应来自 MotionForesight 场景中的失效证据、可靠的表面关系构造、对不同物体运动类型的适配，以及充分消融。

## 本次产出

- 完成用户对话记录中最终结构关系方案的理论、接口和实验审查。
- 给出不依赖 latent 对齐假设的最小可行验证版本。
- 更新 `reports/progress_log.md`。
