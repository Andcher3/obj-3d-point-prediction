# Obj 3D Point Prediction

围绕目标物体三维点未来轨迹预测开展创新点探索的研究工作区。当前输入为已观测 RGB 视频、目标物体说明（例如掩码）及目标上的查询点，输出为未来三维点轨迹，并探索多种可能未来的条件概率建模。

当前方案与验证状态以最新研究记录为准。仓库包含研究笔记、实验脚本、配置、汇报材料和可视化；候选方法不应直接视为已验证成果。

## 阅读入口

建议 AI 助手和研究协作者按以下顺序读取：

1. [AGENTS.md](AGENTS.md)：研究范围、任务接口、证据要求与历史方向边界。
2. [research_roadmap.md](research_roadmap.md)：研究路线与待解决问题。
3. [reports/progress_log.md](reports/progress_log.md)：按时间索引的研究与实验进展。
4. 对应的 `reports/ideas/`、`reports/analysis/`、`configs/` 和 `scripts/` 文件。

引用结论时区分论文事实、已确认设计、研究假设和实验结果。未来帧可用于离线生成监督标签，预测时只使用已观测信息。`archive/imu/` 为历史归档，未纳入当前视频研究方向。

## 数据与本地文件

数据集、权重、缓存和达到 10 MiB 的文件使用目录中的 `GITHUB_PLACEHOLDERS.md` 代替，原路径和大小记录于 [github_upload_manifest.json](github_upload_manifest.json)。占位说明不含原始数据，也不能恢复文件内容。

上传范围详见 [GITHUB_UPLOAD.md](GITHUB_UPLOAD.md)。
