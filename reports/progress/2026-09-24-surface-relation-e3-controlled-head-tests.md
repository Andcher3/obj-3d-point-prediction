# E3 受控 Relation Head 工程测试

- 日期：2026-09-24
- 范围：合成 oracle 数据；不涉及真实视频监督，不生成真实训练缓存

## 测试

- constrained-L2 零修正退化；
- 梯度有限性；
- 单样本过拟合；
- query permutation；
- 与参数量匹配的 node residual 对照。

## 结果

- zero correction 最大绝对误差：`0`；
- 梯度：有限值，通过；
- edge Relation Head 参数量：`1155`；
- node residual 参数量：`1163`，比例 `0.993`；
- 单样本损失比：`0.0124`，过拟合测试通过；
- query permutation 最大绝对差：`1.11e-16`，通过。

该结果只证明受控接口、积分器反传和置换实现可运行，不证明 `H_output` 含有真实关系信息，也不证明 Relation Head 优于 node residual。真实视频监督路线仍冻结。

结果：`experiences_results/surface-relation-e3-controlled-head-2026-09-24.json`。
