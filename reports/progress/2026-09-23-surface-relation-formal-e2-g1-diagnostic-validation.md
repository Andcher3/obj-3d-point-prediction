# E2-G.1 轻量诊断复核

日期：2026-09-23

## 已完成

- 核对诊断脚本使用正式 query manifest 的目标 `bbox_xyxy`，没有继续使用 query 点自身包围盒；
- 核对 4 条连接边的归一化 2D 长度、3D 长度和局部 3D kNN 边尺度比；
- 在当前本地环境重新运行诊断脚本，输出 SHA256 与保存结果完全一致；
- 检查 PyTorch 与 NumPy 兼容警告对本轮计算路径的影响。

## 结果

- 4 条连接边的 `length_2d_over_bbox_diag` 为 `0.0831--0.1143`，中位数 `0.1008`；
- 连接边 3D 长度为局部 3D kNN 边中位数的 `3.40--7.35` 倍，中位数 `5.09`；
- 连接边受控异常扰动会产生非零 component-centered error，支持将修复 3D 图保留为次级候选；
- 当前实验只扰动连接边，没有等幅度局部边匹配对照，因此只能确认 bridge 异常会传播，不能量化声称其相对普通局部边具有更强的误差放大作用。

## 环境边界

本地 `NumPy 2.2.6` 与按 NumPy 1.x 编译的 PyTorch 组件产生兼容警告，但诊断脚本通过 Python list 转换进入 Torch，未依赖失效的 Tensor--NumPy bridge；复跑结果与原 JSON 逐字节一致。本轮结果可保留。进入模型侧实验前应使用兼容环境，避免未来代码调用 `.numpy()` 时失败。

## 决策

接受“E2-G.1 轻量诊断完成、无模型图结构阶段结束”。候选层级保持：2D `k=3` 为主工程候选，修复 3D `k=3` 为次级候选，原始 3D `k=3` 为数值对照，正式默认图等待模型侧 E3 决定。

E2-M 不构成 E3 的强制前置条件。下一阶段优先进行 MotionForesight observed-only `P0,H` 接口与缓存预检；E2-M 作为独立的 MolmoMotion 可变 query 接口问题继续阻塞。

## 输出

- `experiences_results/surface-relation-formal-e2-trackcraft-bundles-2026-09-23/formal_e2_g1_diagnostic.json`
- `reports/progress/2026-09-23-surface-relation-formal-e2-g1-diagnostic-validation.md`

## 下一步

开展 E3 前置接口检查：确认 observed-only 推理边界、`P0` 和 query-aligned `H` 的可获取性、48-query 支持、时间网格以及缓存格式。未通过这些检查前不训练 Relation Head。

