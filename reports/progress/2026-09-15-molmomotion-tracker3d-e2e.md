# 2026-09-15 MolmoMotion Tracker3D 真实输入测试

## 完成工作

- 使用 DA3 与 TrackCraft3R 从 Ego4D sewing clip 构造查询像素和三维历史。
- 将 8 个剪刀查询点、3 帧三维历史和 4 类动作文本送入 MolmoMotion。
- 保存原始 `<tracks>` 文本、三维数组、指标、查询点与预测可视化。
- 完成相同配置复跑、哈希比较和独立副本回滚测试。

## 输出

- `reports/analysis/2026-09-15-molmomotion-tracker3d-e2e/report.md`
- `archive/experiments/molmomotion/MolmoMotion-tracker3d-e2e/final/`

## 关键结果

端到端输入链路已跑通，四条文本形成不同 token 输入；模型产生的原始轨迹文本和解析轨迹全部相同，预测段内部路径长度为 0。当前真实输入样本没有表现出可用的自由文本条件控制，因此未经适配的 MolmoMotion 暂不能作为已验证的第二阶段模型。

## 下一步

将本结果作为 MolmoMotion 失败样例；后续模型选择可比较 TrajLoom，或用更多分布内/真实 Tracker3D 样本区分模型文本能力与输入分布差异。

