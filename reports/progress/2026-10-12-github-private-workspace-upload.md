# 项目上传到 GitHub 私有仓库

日期：2026-10-12

当前仓库已按用户要求更名并公开：https://github.com/Andcher3/obj-3d-point-prediction 。以下保留首次以私有仓库上传的记录；后续变更见 [仓库改名并公开](2026-10-12-public-obj-3d-point-prediction-repository.md)。

## 已完成

- 创建私有仓库：https://github.com/Andcher3/post_graduate_project ，默认分支为 `main`。
- 将代码、研究记录、配置、汇报材料和图片/视频按原有目录上传；用户明确选择保留未超过大文件阈值的图片、视频并等待传输。
- 数据数组、模型权重、下载压缩包、数据集目录、缓存及单文件达到 10 MiB 的内容通过目录占位说明代替。全部原文件保留在本地。
- 初始化本地 Git，并将 `origin` 指向新仓库；修正本仓库使用的本机代理端口，启用 TLS 证书校验。

## 输出与验证

- `scripts/prepare_github_snapshot.py`：生成排除规则、原文件清单及目录占位说明；不删除或覆盖本地数据。
- `.gitignore`、`GITHUB_UPLOAD.md`、`github_upload_manifest.json`，以及 218 个目录的 `GITHUB_PLACEHOLDERS.md`。
- 首次提交包含 3,663 个文件；3,531 个待占位的原始文件均未被 Git 跟踪，所有预定上传文件和占位说明均在提交中。
- 已检查文本及 Office 文档 XML 中的已知凭据模式，未发现匹配；首次推送成功，本地与远端 `main` 的提交哈希一致。
- 未运行历史 IMU 命令，未运行训练、推理或研究实验。

## 使用边界

占位文件只记录原路径、大小与排除原因，不能恢复原始数据。继续实验时，需要另行取得对应数据集与权重。当前研究范围仍以现有项目文档为准，本次操作没有改变方法设计。
