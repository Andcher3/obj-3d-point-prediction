# F2 多轮修正版与留出实验复核

- 日期：2026-09-26
- 状态：F2 迭代阶段完成；未放行 F3。

## 认证与数据边界

- 按用户给出的交互式命令再次连接 `region-9.autodl.pro:49356`。
- 服务器提示密码后输入用户给出的密码，仍返回 `Permission denied`。
- 当前仅有一个完整同坐标 F0 bundle，因此所有新增结果仍是单视频 pilot。

## 已运行的 F2 变体

### 1. 修正版基础 F2

- zero-init residual；
- 统一去质心坐标损失和边向量损失；
- 训练时域拟合特征标准化；
- 验证集选 checkpoint。

最后 5 个未来槽位测试中：

- baseline 边相对误差：`18.998%`
- H edge：`21.155%`
- node residual：`16.365%`，但 raw ADE `0.1044 m`

H edge 未通过。

### 2. 时间切分变体

- `8/2/5`：node centered 通过 5/5；dynamic edge 通过 0/5。
- `10/2/3`：node centered 和 dynamic edge 均通过 5/5。
- `6/4/5`：node centered 和 dynamic edge 均通过 0/5。

这说明正结果依赖训练未来时域覆盖范围，不能视为稳定的长时域泛化。

### 3. query 留出

- 每次使用 32 个 query 监督、16 个 query 留出；
- 全部 15 个 future slot 用于训练；
- 测试边包含至少一个留出 query；
- 5 个随机 query 划分。

结果：

- baseline 边相对误差中位数：`17.63%`
- node centered：`20.78%`
- dynamic edge：`22.86%`
- 两个变体均为 `0/5` 通过安全闸门。

## 解释

- head 可以在已见未来时域和已见 query 上拟合关系目标；
- 这种拟合没有稳定转移到更远时域或未见 query；
- H edge 没有显示出比 baseline 更可靠的关系信息；
- node centered 解决了此前的整体平移破坏，但没有解决空间 query 泛化；
- 目前的正结果更接近“受限数据上的拟合能力”，不构成 Relation Head 或 node residual 的泛化证据。

## 当前方法判断

- “baseline 存在结构误差”：在单视频伪 GT 上仍有支持。
- “冻结 H 后的 edge Relation Head 能提取可泛化的点间关系”：当前证据不支持。
- “组件级约束和 node centered residual 可以稳定改善真实预测”：当前证据不支持。
- “关系建模在数学和工程上可运行”：已支持。

这并不构成数学意义上的不可能性证明。它说明在当前数据、伪 GT、特征和训练接口下，继续微调 head 不应作为下一步主线。若要重新开启关系学习，必须先取得更多独立同坐标 F0 bundle，并按视频划分训练/测试；当前远端认证失败使该扩展暂时无法进行。

## 输出

- 基础 F2：[f2_redesigned_pilot.json](../../experiences_results/surface-relation-f2-redesigned-2026-09-26/9069bd26-0139-4e88-9957-9f9489b35a91/f2_redesigned_pilot.json)
- 时间变体（8/2/5）：[默认变体结果](../../experiences_results/surface-relation-f2-variants-2026-09-26/9069bd26-0139-4e88-9957-9f9489b35a91/f2_variant_probe.json)
- 时间变体（10/2/3）：[split10_2_3.json](../../experiences_results/surface-relation-f2-variants-2026-09-26/9069bd26-0139-4e88-9957-9f9489b35a91/split10_2_3.json)
- 时间变体（6/4/5）：[split6_4_5.json](../../experiences_results/surface-relation-f2-variants-2026-09-26/9069bd26-0139-4e88-9957-9f9489b35a91/split6_4_5.json)
- query 留出：[f2_query_split_probe.json](../../experiences_results/surface-relation-f2-query-split-2026-09-26/9069bd26-0139-4e88-9957-9f9489b35a91/f2_query_split_probe.json)
- 代码：[surface_relation_f2_variant_probe.py](../../scripts/surface_relation_f2_variant_probe.py)
- 代码：[surface_relation_f2_query_split_probe.py](../../scripts/surface_relation_f2_query_split_probe.py)
