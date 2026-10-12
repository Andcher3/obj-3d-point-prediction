# 正式 E2 数据与模型预检

## 已完成

- 服务器 `/root/autodl-tmp/ego4d-formal-e2/` 保留 Ego4D `ego4d.json` 与 `manifest.csv`。
- 中断的全量 annotations 下载进程已终止，临时 AWS 凭据文件已删除。
- 已下载两个可复用 clip，以及按固定 metadata 顺序选择的 4 个候选 clip；候选清单见 `configs/surface_relation_formal_e2_candidate_manifest_2026-09-22.json`。
- 候选覆盖标签为 2 rigid、2 articulated、2 non-rigid/deformable，但新增 4 个视频仍需人工审核 `t0` polygon mask，并需补齐 pointmap、真实时间戳和 pseudo-GT。
- sampler 对 16/48/64 点均通过。
- 官方仓库 `allenai/molmo-motion` 与 `MolmoMotion-4B-H3-F30` checkpoint 已下载到服务器。

## 关键阻塞

公开 checkpoint 的 processor 已成功加载，但接口直接要求：

```text
points_3d_history must be (3, 8, 3); got (3, 16, 3)
```

因此当前公开 AR checkpoint 只接受固定 8 点输入，不能支持正式 E2 冻结的 48 点主结果和 64 点密度分析。8 点不替代正式协议；需要支持可变 query 数的模型/适配器后才能进行模型侧 E2 推理。该限制不影响独立的图求解器、时间网格、mask 和 query sampler 代码预检。

## 当前状态

正式 E2 仍未就绪。数据侧缺少人工 mask、三维历史、未来 pseudo-GT 和真实时间戳；模型侧还缺少可变 query 接口。旧 21 个 `.npz` 审计结果仍为不合格：3 个 source video、每组 8 点、缺少 `t0_seconds`、`future_timestamps_s` 与 `pseudo_gt_timestamps_s`。

