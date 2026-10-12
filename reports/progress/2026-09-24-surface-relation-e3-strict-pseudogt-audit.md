# E3 严格 pseudo-GT 身份审计

## 当前状态

已实现并部署严格审计 runner：

- past-only 身份窗口 `[t0-11, ..., t0]`；
- 离线监督窗口 `[t0-2, ..., t0+9]`；
- 两次 TrackCraft3R 在共同 `t0` 对 48 个 query 做全局一对一匹配；
- 四折 query cross-fitting，每折 36 点估计相似变换、12 点留出评估；
- 3D 登记误差、2D anchor 误差、有效 query、2D `k=3` 连通性和 bridge 字段输出。

远端环境：`region-9.autodl.pro:25869`，Conda 环境 `motionforesight-e3`，输出目录为 `/root/autodl-tmp/projects/motionforesight/outputs/e3_strict_2026-09-24`。

## 已完成证据

视频 `cb78dba8-6647-4ec7-875c-6ba1548f8312` 已完成两次 12 帧 TrackCraft3R 推理和审计：

- 48/48 query 有效；
- 2D anchor `p90=0.718%`、最大 `1.138%`，2D 图连通；
- 留出 3D 登记误差 `p90=34.52%`、最大 `59.15%` 的 3D bbox 对角线；
- 10/48 query 超过 10%；
- 视频 gate：`FAIL`。

视频 `356ed369-7023-4660-904b-be73823a21d3` 也已完成：

- 48/48 query 有效，2D 图连通；
- 留出 3D 登记误差 `p90=26.23%`、最大 `47.77%`；
- 36/48 query 超过 10%；
- 视频 gate：`FAIL`。

该结果说明旧的整体相似变换残差不能代表跨序列监督登记已可靠。当前失败来自严格留出登记误差，不是 TrackCraft3R OOM 或时间窗口缺失；误差来源仍可能包括 DA3 跨上下文几何差异、不同深度层近邻选择和 query identity 错配。

## 运行边界

其余三个视频的严格推理已完成。五个视频均未通过监督登记 gate。补充诊断已输出真实 polygon bbox、直接 UV 对照、query 最近/次近邻距离、两两距离误差和自一致性基线。Relation Head、E3-lite 训练、E2-M、E4、IRLS、Huber 和 learned gate 均未启动。

本轮使用的脚本：

- `scripts/surface_relation_e3_strict_prepare.py`
- `scripts/surface_relation_e3_strict_audit.py`

远端批处理曾因首版审计脚本未支持逐帧外参而中止；已修正批量 `4x4` 变换并复用已生成 dense 输出，未重跑已完成的模型推理。
