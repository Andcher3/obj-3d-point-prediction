# E3 受控 Relation Head 测试复核

- 日期：2026-09-24
- 状态：工程闸门通过；真实视频监督与方法收益仍未验证

## 结果判断

- zero correction 最大误差为 `0`，说明积分器在零修正时能退化到 baseline。
- 梯度为有限值，单样本损失比例降至 `0.0124`，说明反向传播和最小优化路径可运行。
- query permutation 最大误差 `1.11e-16`，支持当前图构造、边特征和积分接口在同步置换下保持等变。
- edge head 与 node residual 参数量分别为 `1155` 和 `1163`，参数量对照基本匹配。

这些结果没有测试真实 `H_output` 是否含有关系信息，也没有产生 edge head 优于 node residual 的性能证据。置换测试也只覆盖同步置换输入、query 和图边的实现性质，不等同于训练后的泛化不变性。

## 决策

1. E3-controlled 工程阶段可以判定通过，不再为当前接口增加复杂模块。
2. 不启动真实视频 Relation Head 训练，也不进入 E4、IRLS、Huber 或 learned gate。
3. 下一阶段优先复现 MotionForesight 官方同坐标监督链，确认原始 `future_3d`、`L_track`、有效性 mask、时间窗口与 `P0` 的坐标关系。
4. 监督链成立后，先做 decoded-coordinate relation loss 与等规模 node residual 的最小对照；Relation Head 暂作为候选实现，不能写成已验证改进。

E0 已包含数值梯度检查，因此本轮“梯度有限性”不必单独升级为阻塞条件；若后续修改积分器或 head，再补有限差分对照。
