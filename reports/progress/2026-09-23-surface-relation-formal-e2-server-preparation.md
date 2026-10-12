# 正式 E2 服务器环境与数据准备续接

日期：2026-09-23

> 状态更新：本页记录的是环境准备阶段。后续 TrackCraft3R pseudo-GT、5 视频正式 bundle、数据审计和 E2-G runner 已完成，见 [正式 E2-G 数据构建与辅助实验](2026-09-23-surface-relation-formal-e2-g-execution.md)。

## 已完成

- 重新连接当前 AutoDL V100S 服务器，确认 GPU 为 `Tesla V100S-PCIE-32GB`，数据盘路径为 `/root/autodl-tmp`。
- 确认 TrackCraft3R 仓库和 checkpoint 仍在远端；checkpoint 文件存在且大小约 17.8 GB。
- 通过 PyPI 安装 `depth-anything-3==0.1.1`，补齐其运行时导入所需的 `moviepy`、`proglog`、`decorator`、`addict`、`roma`、`plyfile`、`pycolmap`、`evo` 等依赖。
- `from depth_anything_3.api import DepthAnything3` 已通过导入检查。导入时仅提示可选 `gsplat` 缺失，不影响点图推理入口。
- 本地正式 E2 准备脚本、候选 manifest 和 6 个 t0 帧仍保持协议约束：15 fps、三帧历史、未来 9 步、32/48/64 query、E2-G/E2-M 来源分离。

## 当前阻塞

- DA3-BASE checkpoint 已通过 `DepthAnything3.from_pretrained` 下载并加载；六个候选视频的三帧 pointmap 已生成到远端 `ego4d-formal-e2/pointmaps_history/`，并同步到本地 `experiences_results/remote-pointmaps-history/`。
- 六个视频仍没有经过坐标记录的 t0 polygon mask；当前 t0 图像只能作为人工审核输入，不能写成正式 E2 数据。
- TrackCraft3R pseudo-GT 必须在 pointmap、真实 source timestamp 和目标 query 坐标确定后生成，当前不能把旧 `.npz` 直接复用为正式数据。
- 公开 MolmoMotion checkpoint 仍固定要求 `(3, 8, 3)`，E2-M 不得运行或扩展为 48/64 query。

## 结论

当前服务器环境已具备 DA3 API 和 TrackCraft3R 运行条件，六个视频的三帧 pointmap 已完成；正式 E2-G 仍未通过数据审计。下一步需要完成 polygon 坐标、query manifest 和 TrackCraft3R 时间对齐 pseudo-GT，之后再运行正式审计与 E2-G runner。
