# E3 定位实验复核与监督路线决策

- 日期：2026-09-24
- 状态：当前 TrackCraft3R/DA3 vector relation pseudo-GT 路线冻结；Relation Head 仅保留受控工程测试

## 证据判断

### t0 首帧教师实验

- `9069...`：固定 query 作为 TrackCraft3R 第一帧 anchor 后，留出登记 p90 为 `14.03%`，最大误差 `39.25%`，7/48 点超过 10%。
- 浇水壶：留出登记 p90 为 `40.02%`，最大误差 `47.26%`，37/48 点超过 10%。
- 两个结果都未通过既定登记门槛。首帧固定消除了从早期窗口回溯到 t0 的身份匹配因素，但不能消除不同坐标/深度上下文之间的几何不一致。

### 分辨率 A/B

- 浇水壶高分辨率结果：有效 anchor 44/48，p90 `5.27%`，未来两两距离漂移 p90 `78.63%`。
- `process_res=392`：有效 anchor 24/48，p90 `35.42%`，未来两两距离漂移 p90 `91.07%`。
- 降分辨率会恶化结果，因此不能把高分辨率失败主要归因于 392 分辨率；该 A/B 不能唯一确定失败原因。

## 方法假设边界

仍保留：

1. 同一目标点之间可能存在可利用的相关运动；
2. 时变软关系修正可能减少独立点预测的散开；
3. 组件级 gauge-fixed constrained-L2 图积分在数值上可用。

当前不成立或尚未成立：

1. TrackCraft3R/DA3 生成的独立三维轨迹可以与 observed-only MotionForesight `P0,H` 直接配对；
2. 当前单上下文 vector relation pseudo-GT 足够可靠，可直接用于真实视频监督；
3. `H_output` 含有可学习的关系修正信号；
4. Relation Head 优于同等规模的 node residual。

固定刚体边长或硬约束也不应恢复。浇水壶正式 16 帧结果的未来两两距离漂移 p90 为 `78.63%`，说明当前估计链不能把大形状变化直接解释成真实形变。

## 决策

1. 停止继续为当前 TrackCraft3R/DA3 vector pseudo-GT 路线增加登记器、历史窗口、鲁棒积分器或 learned gate。
2. 不生成真实视频 Relation Head 训练缓存。
3. 允许开展 `E3-controlled`：zero correction、梯度/有限差分、constrained-L2、query permutation、单 oracle 样本过拟合，以及 Relation Head 与等规模 node residual 的接口对照。
4. 受控实验通过后，优先复现 MotionForesight 官方同坐标训练数据链，直接从其原始 `future_3d`/`L_track` 标签派生 decoded-coordinate relation loss；在此之前不恢复真实视频训练。

## 解释边界

本轮结果证明的是当前监督来源和坐标接口不具备放行条件，不证明关系建模方向无效，也不证明 MotionForesight 的 hidden state 没有关系信息。若未来只能获得跨上下文伪标签，应将关系目标降级为诊断量，不能作为训练监督。
