# 2026-09-14 MolmoMotion 文本类型探测

## 已完成

- 归档旧 FUTURE-L1 扰动实验的本地和远程结果。
- 在 AutoDL RTX 5090 环境部署官方 `allenai/molmo-motion`，使用 `MolmoMotion-4B-H3-F30` 权重跑通基线。
- 对三组官方完整样本执行 5 类动作文本，共 15 次推理。
- 完成一次官方文本确定性复跑；输出逐元素完全一致。
- 通过 Ego4D CLI 下载 3 个 FHO 短片，并核对哈希与时长。
- 创建修改脚本、补丁、验证记录和可执行回滚脚本；在独立副本上验证回滚。

## 关键结果

- 自由句子和多步骤计划均能得到 `(8,30,3)` 有限轨迹，接口层可以接收这些文本形式。
- 文本条件对轨迹的作用随样本变化。清洁和车辆样本主要沿视觉历史延续，冲突文本没有可靠改变运动模式；球体放置样本对对象—目标关系较敏感。
- 当前权重适合继续作为第二阶段候选进行条件遵循测试，尚不能视为通用长计划或稀有模式轨迹执行器。
- 原始 Ego4D 视频缺少 MolmoMotion 所需的目标查询像素和三维点历史，本轮只作为已下载数据样本保留，没有混入文本变量对照。

## 输出

- `reports/analysis/2026-09-14-molmomotion-text-type-probe/report.md`
- `experiences_results/MolmoMotion-text-types/run_text_type_probe.py`
- `experiences_results/MolmoMotion-text-types/remote/results/text-type-probe/metrics.json`
- `experiences_results/MolmoMotion-text-types/remote/results/text-type-probe/metrics.csv`
- `experiences_results/MolmoMotion-text-types/remote/VERIFICATION.txt`
- `experiences_results/MolmoMotion-text-types/ego4d-samples/`

## 下一步

将“接口可接受自由文本”与“轨迹按文本模式执行”分开评估；重点检查稀有模式和与视觉惯性冲突的模式条件遵循能力。
