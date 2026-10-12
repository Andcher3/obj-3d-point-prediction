# F2 锚定监督下的修正方向与幅度诊断

- 日期：2026-09-27
- 状态：实验完成；事后最佳门控仅用于诊断。

## 完成工作与复现

沿用 `process_res=518` 的四视频同坐标 bundle、`anchored_displacement` 监督、四折整视频留出、三种 head（history/shared/safe）、每种五个种子和 400 步训练。每折用其余三视频的未来槽位 `0:7` 训练、`7:9` 选择 checkpoint/门控，留出视频九个槽位仅用于最终诊断。冻结 MotionForesight baseline；未重新运行 TrackCraft3R。

复跑并导出每个点、每个未来槽位的实际预测修正；逐项计算该修正与「锚定伪标签减 baseline」的方向相似程度。然后**事后使用留出视频的伪标签**，在 `0≤g≤1` 中求出使去质心坐标和三维边向量的等权平方误差最小的标量修正幅度。该幅度只说明现有修正方向的潜力，不能在推理时获得，也不能计作模型成绩。

复现命令（PowerShell）：

```powershell
python scripts/surface_relation_f2_direction_gate_diagnostic.py `
  --bundle-root experiences_results/surface-relation-e3-formal-res518-2026-09-26 `
  --reference experiences_results/surface-relation-f2-observed-relations-2026-09-26/f2_anchored_displacement_loo.json `
  --output experiences_results/surface-relation-f2-direction-gate-2026-09-27/direction_gate_diagnostic.json `
  --arrays-output experiences_results/surface-relation-f2-direction-gate-2026-09-27/corrections.npz `
  --video-ids 65165142-b15e-4bba-a9de-d5174ec46597 4d053322-1eef-4a69-8201-932c7b82fb03 9069bd26-0139-4e88-9957-9f9489b35a91 cb78dba8-6647-4ec7-875c-6ba1548f8312
```

60 组的原选门控与原 F2 一致；对应重跑指标最大绝对差 `0`。修正数组形状 `(4,3,5,9,48,3)`，均为有限数值。

## 主要结果

下表为每视频五个种子的**事后最佳修正幅度中位数**；原 F2 训练视频验证对全部 60 组均选择 `g=1`。

| 留出视频 | history | shared | safe |
|---|---:|---:|---:|
| `6516` | 1.000 | 1.000 | 1.000 |
| `4d05` | 0.233 | 0.269 | 0.213 |
| `9069` | 0.088 | 0.079 | 0.059 |
| `cb78` | 0.100 | 0.154 | 0.239 |

60/60 组在整视频汇总后的最佳幅度均大于零，表示修正方向总体带有一些与所需变化一致的信息；按单个未来槽位统计，540 组中 16 组最佳幅度为零。方向相似程度并不高：例如 `9069/history` 的综合方向余弦（两种误差向量的方向相似度）中位数仅 `0.229`。因此只凭最佳幅度为正，不能认为方向预测已可靠。

统一按有效 query 的物体尺度归一化，`4d05/history` 的 baseline 三维边向量误差为 `0.0421`，全量修正后 `0.0677`，事后最佳幅度下 `0.0392`；`9069/history` 分别为 `0.0156/0.0345/0.0152`。前者存在有限的潜在改善，后者即使用留出标签选幅度，边误差收益仍很小。`6516` 则本来适合较大修正；一个全局固定幅度不能同时匹配这些视频。

## 解释与决策

当前 head 的主要可定位问题是：训练视频选出的全量门控在三个留出视频上明显过强；现有修正方向并非全无信息，但不同视频的方向质量与所需幅度差别很大。`anchored_displacement` 已逐点消除两次 TrackCraft3R 在 `t0` 的静态位置差异，本次退化不应再归咎于该静态差异。

事后最佳门控读取了测试伪标签，不是可部署预测器，也不证明简单调小固定系数就能获益。四个视频是本轮反复研究过的探索性样本，五个种子不是五个独立视频；教师未来位移仍可能含伪标签误差。未做正式显著性推断，也不能据此认定关系方法或冻结输出接口有效。统计解释检查了常见的 11 类推断风险，当前主要风险是反复探索、小独立样本、伪标签和事后选择。

下一步研究判断：若继续这条路线，应先提出**仅使用观测输入与 baseline 预测**来预估修正可靠性和幅度的简单门控，并在事先锁定的独立视频/更可信监督上与 `g=0` baseline 比较；保留拒绝修正的能力。当前不冻结 head，不将 oracle 最佳门控作为正式结果，也不启动更复杂的积分器或 IRLS。

## 产物

- [诊断脚本](../../scripts/surface_relation_f2_direction_gate_diagnostic.py)
- [逐组指标与诊断](../../experiences_results/surface-relation-f2-direction-gate-2026-09-27/direction_gate_diagnostic.json)
- [逐点修正数组](../../experiences_results/surface-relation-f2-direction-gate-2026-09-27/corrections.npz)
