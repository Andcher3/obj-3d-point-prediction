# 点间关系 F1/F2 pilot 与结果复核

- 日期：2026-09-26
- 范围：单视频 F1 baseline 结构漂移审计与 F2 冻结主干可学习性 probe；F2 结果因比较与评估问题标记为未验证。

## F1

- 对象：`9069bd26-0139-4e88-9957-9f9489b35a91`，22 帧、7 帧观测、15 帧未来、48 个 query。
- 使用同一 F0 输入下的 MotionForesight observed-only 输出和 TrackCraft3R query-only 轨迹；后者转换到最后观测相机坐标。
- 2D symmetric kNN `k=3` 共 86 条边。
- 15 个未来步上，去质心点误差中位数归一化值由 `1.11%` 增至 `2.63%`；边向量相对误差中位数由 `10.60%` 增至 `16.86%`；边长形状相对误差中位数由 `4.39%` 增至 `10.42%`。
- 三项误差序列与 horizon 的相关系数分别为 `0.954`、`0.955`、`0.934`。这是一个单视频、伪标签下的描述性漂移信号，不是统计显著性或真实精度结论。
- 判据问题：F1 脚本的 `pilot_structural_drift_signal` 要求去质心误差至少 5%，因而结果为 false；随后依据边关系误差增长继续执行 F2，实际采用的放行逻辑与脚本布尔门槛不一致。该阶段只能记作“指标呈增长、总体 gate 未清晰通过”，不能声称严格按统一门槛放行。

## F2

- 使用前 9 个 future slot 训练、接着 3 个验证、最后 3 个测试；比较未训练 MotionForesight baseline、geometry-only edge MLP、`H+p0` node residual、`H+geometry` edge head；5 次运行仅改变初始化种子。
- 当前输出文件记录的测试边相对误差中位数：baseline `18.82%`，geometry-only edge MLP `19.79%`，node residual `19.35%`，`H+geometry` edge head `19.19%`。edge head 未优于原始 baseline；相对 node residual 也没有稳定优势。结构指标不支持放行 F3。
- F2 判据实现存在缺陷：结果生成时的自动 gate 只比较 edge head 与两个训练模块，没有要求优于未训练的 geometry-only baseline，故结果中的 `advantage_gate=true` 不能作为放行依据。运行后已修改本地 runner，将 baseline 和 node residual 纳入优势门槛；当前 JSON 仍对应旧脚本版本，未重新运行。
- node residual 的训练损失使用去质心坐标误差，但测试 raw ADE 保留整体平移，node head 又没有零均值/平移规范约束。因此其 raw ADE 与其他方法不可直接比较；去质心及边指标仍可描述，但 node 与 edge 的训练目标/约束并未完全匹配。
- 五个种子共享同一视频、同一 query 集和同一伪标签；时间留出不能替代跨视频或未见 query 泛化。验证集没有用于 checkpoint 选择，报告值是固定训练步数后的结果。以上结果不足以证明方法可泛化，也不构成 Relation Head 收益证据。

## 当前判断

- F1 观察到随时域增加的结构差异，但单视频和伪标签限制较大，且 gate 定义不一致。
- F2 的 exploratory 结果没有显示 `H+geometry` edge head 优于 baseline；现有自动通过标记无效。
- F3 未启动，Relation Head 仍未获得可验证的真实视频收益证据。
- 在进入下一阶段前，应先固定 F1 放行规则，并修正 F2 的公平比较与平移处理，再重新评估 F2。当前未重跑 F2，也未启动 F3。

## 输出

- 脚本：[surface_relation_f1_baseline_structure_audit.py](../../scripts/surface_relation_f1_baseline_structure_audit.py)
- 脚本：[surface_relation_f2_learnability_probe.py](../../scripts/surface_relation_f2_learnability_probe.py)
- F1 结果：[f1_baseline_structure_audit.json](../../experiences_results/surface-relation-f1-baseline-structure-2026-09-26/9069bd26-0139-4e88-9957-9f9489b35a91/f1_baseline_structure_audit.json)
- F2 exploratory 结果：[f2_learnability_probe.json](../../experiences_results/surface-relation-f2-learnability-2026-09-26/9069bd26-0139-4e88-9957-9f9489b35a91/f2_learnability_probe.json)
- 下一步：暂停 F3；先解决 F1/F2 的判据和比较设计问题。
