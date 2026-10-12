# DAVIS 四视频联合关系解码器教师审计

- 日期：2026-09-29
- 类型：替代监督获取与质量闸门实验
- 状态：工程链通过，质量闸门失败，训练未启动

## Material Passport

- Source scope: DAVIS `car-turn`、`drift-turn`、`boat`、`soccerball`
- Runtime: 远端 NVIDIA RTX 6000D；PyTorch `2.8.0+cu128`；TrackCraft3R/DA3 query-only
- Time basis: DAVIS 连续源帧索引；预览 MP4 的人为 24 fps 未用于协议
- Verification status: VERIFIED（远端运行结果与本地同步文件 SHA256 一致）

## 完成工作

- 只读检查远端磁盘后，删除已确认可再生且未在用的 `/root/autodl-tmp/wheelhouse-cu128`，释放约 2.7 GiB；可用空间由 9.9 GiB 增至 13 GiB，实验结束后约 12 GiB。
- 在独立 Conda 环境 `/root/autodl-tmp/conda_envs/trackcraft-sm120` 安装 GPU PyTorch、DA3 与 TrackCraft3R 依赖，完成 CUDA `sm_120` 实算检查。
- 下载和实验命令均启用 AutoDL 学术加速：`source /etc/network_turbo`。
- 每段视频生成 48 个固定 query、7 帧 observed-only MotionForesight 输入、23 帧 DA3 教师输入和两次 12 帧交错 TrackCraft3R query-only 输出，最终伪轨迹形状均为 `(22,48,3)`，有限值比例均为 1.0。
- TrackCraft3R 峰值已分配显存约 27.42 GiB，四视频均未发生 CUDA OOM 或进程崩溃。
- 完成 Route A（固定 t0 query/Pj0 锚定）与 Route B（完整上下文身份匹配）质量审计。

## 质量结果

| 视频 | Route A | Route B | 共同 t0 登记 p90 | 登记最大值 | Route A anchor p90 | Route B identity p90 | 2D k=3 图 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| car-turn | FAIL | FAIL | 28.55% | 75.11% | 1.01% | 40.99% | 连通，无桥边 |
| drift-turn | FAIL | FAIL | 34.52% | 40.83% | 0.90% | 106.19% | 连通，无桥边 |
| boat | FAIL | FAIL | 10.68% | 51.68% | 0.60% | 6.40% | 连通，1 条桥边 |
| soccerball | FAIL | FAIL | 30.63% | 53.28% | 2.02% | 57.78% | 不连通，分量 24/15/9 |

- Route A 的直接 anchor 误差较小，说明 query-only 和 frame-0/Pj0 锚定接口能够工作。
- 四视频共同 t0 三维登记均未通过预设门槛；Route B 还存在明显身份匹配误差。
- `soccerball` 的有效 query 图不连通，不能作为完整目标关系监督。
- Route A/B 均为 `0/4 PASS`，因此 `training_started=false`。未运行冻结 MotionForesight baseline、等参逐点或跨点联合解码器的整视频留一训练。

## 异常与限制

- 未来九帧的 mask 命中率几乎为零，但对应目标 mask 实际非空。现有结果不能区分教师轨迹错误与未来投影/mask 坐标接口错误；该字段只作为异常记录，不用于放宽质量闸门。
- 本轮验证的是 DAVIS 替代 TrackCraft3R 教师链，不能称为 MotionForesight 官方监督复现或可靠真实视频训练标签。
- 工程可行性通过不代表替代监督质量通过；当前禁止启动联合解码器和逐点基线训练。

## 输出与校验

- 本地结果：`experiences_results/surface-relation-davis-ab-teacher-2026-09-29/`
- 核心汇总：`files/final_run_summary.json`、`files/quality_summary.json`
- 每视频审计：`files/route_a/*/quality_audit.json`、`files/route_b/*/quality_audit.json`
- 运行日志：`files/common/*/pipeline.log`、`files/common/*/route_a_trackcraft.log`
- 同步归档 SHA256：`25ffab037c10d4d5204d2f64ba652374bf8c27b0058d5a9a2a45ac39db2ee290`
- 归档哈希远端/本地一致；53 个 JSON/日志文件逐文件 SHA256 不一致数为 0。

## 当前决定

- 停止本轮训练阶段，不降低 48 query、22 帧、时间窗口或质量门槛。
- 保留输出作为替代监督失败证据与工程链审计材料。
- 后续若做定位，只检查未来投影与 mask 坐标接口；不据此重新训练或放宽门槛。
