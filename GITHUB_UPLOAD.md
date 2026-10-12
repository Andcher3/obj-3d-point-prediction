# GitHub 上传说明

日期：2026-10-12。仓库设为私有。

代码、研究文档、配置、汇报材料与适量实验可视化按原目录上传。数据集、数组、模型权重、下载压缩包、缓存及达到 10 MiB 的文件通过各目录的 `GITHUB_PLACEHOLDERS.md` 表示；包含疑似凭据的文件也只上传占位说明。原文件保留在本地。

完整原路径、大小与原因记录于 `github_upload_manifest.json`，占位说明不能用于恢复文件内容。运行相关实验时，需要按项目记录另行取得数据和权重。

活动研究范围以 `AGENTS.md`、`research_roadmap.md` 及最新研究记录为准。`archive/imu/` 仅为历史归档，本次上传没有执行其中的程序。

可用 `python scripts/prepare_github_snapshot.py` 重新检查目录；`--write` 更新占位文件及排除清单。
