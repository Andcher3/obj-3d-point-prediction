# 2026-09-15 MolmoMotion 校时多视频复测

## 完成内容

- 将 30 fps 视频按步长 2 采样为 15 fps，使用 `H=3` 历史帧。
- 选取剪刀拿起和手持浇水壶行走两段包含实际物体运动的 Ego4D 视频。
- 用 DA3-BASE 与 TrackCraft3R 构造历史三维点，并以之后 9 帧作为离线轨迹伪标签。
- 对每段视频测试真实动作、反事实动作和多步骤动作，共完成 12 次预测。
- 完成 6 个 greedy 结果的精确复跑、11 项自动检查和可逆代码事务验证。

## 结果

校正后不同文字会改变确定性轨迹，剪刀的静止文字将路径长度压到 `0.0022 m`，拿起文字为 `0.0593 m`。模型的动作遵循与精度仍不稳定：剪刀 observed-action 不比静止文本更接近伪标签；浇水壶携带文本过度预测到 `1.2008 m`，三种文本方向仍高度相似。

当前证据将 MolmoMotion 定位为“文本条件接口已验证、可靠条件执行未验证”的第二阶段候选。

## 产物

- 报告：`reports/analysis/2026-09-15-molmomotion-tracker3d-retest/report.md`
- 代码与结果：`archive/experiments/molmomotion/MolmoMotion-tracker3d-retest/final/`


