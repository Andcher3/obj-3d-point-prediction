# 仓库改名并公开

日期：2026-10-12

## 已完成

- 根据用户确认，将仓库从 `post_graduate_project` 改名为 `obj-3d-point-prediction`，可见性设为 Public。
- 当前地址：https://github.com/Andcher3/obj-3d-point-prediction 。本地 `origin` 已同步，默认分支仍为 `main`。
- 新增 `README.md`，介绍目标物体三维点预测的研究范围，并列出 `AGENTS.md`、研究路线与最新进展等 AI 阅读入口。
- 更新 `GITHUB_UPLOAD.md` 及占位清单脚本中的仓库名称和可见性说明，保持原有数据、权重、缓存及大文件占位规则。

## 验证与边界

- GitHub API 返回新名称及 `visibility=public`；不携带登录凭据也能读取仓库元数据和 README。
- 重新扫描上传范围内的文本和 Office 文档 XML，未发现已知凭据模式匹配。
- 本次未改变研究方法、运行实验或上传被占位替代的数据文件。Public 内容可匿名访问，其他 AI 的实际读取能力仍取决于其工具和网络权限。

## 输出

`README.md`、`GITHUB_UPLOAD.md`、`scripts/prepare_github_snapshot.py`、`github_upload_manifest.json`、本次进展记录与进展索引。
