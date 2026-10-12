# 远程服务器迁移与 Future-L1 round3 实验状态审计

## Material Passport

- `type`: experiment-status-audit
- `status`: VERIFIED_READ_ONLY_AUDIT
- `date`: 2026-09-11
- `scope`: VIDEO_MOTION_FORECASTING / Future-L1 latent branch proposal
- `source`: 本地项目记录与新 AutoDL 端点只读检查
- `remote_endpoint`: `connect.bjb1.seetacloud.com:19874`
- `credentials_persisted_in_project`: false

## 审计范围

核对本地 `reports/progress_log.md`、9 月 10 日实验复核记录、round3 代码快照，并通过新端点检查远程环境、GPU、进程、代码状态、结果目录、报告、哈希和测试。没有启动模型推理、没有改动远程代码或结果。

本轮实验的直接目的来自项目对话记录：验证冻结 Future-L1 的 active latent 扰动、事件前瞻和反馈筛选能否产生多个**高层未来事件模式**。文本只承担模式区分；方向、速度、三维几何和物理可行性属于后续阶段，本轮不纳入实验门槛。

## 当前项目状态

- 研究任务仍是视频输入、指定目标物体、未来三维点轨迹及其多未来条件分布。
- 当前实验候选实现是冻结 `Eurayka/Future-L1-SFT` 的 active latent 扰动、候选生成和事件级反馈筛选。
- 本轮只验证第一阶段的高层事件候选；MotionForesight 轨迹复现、三维 decoder 和模式内物理细化属于后续阶段，不构成本轮缺陷。
- 本轮也没有把 retained 数量解释为概率；模式概率和校准仍是后续的独立问题。
- MotionMap 模式图是备选路线；TrajLoom 适配仍是接口可行性讨论，未形成三维实验。

## 迁移后远程环境

- 主机：Ubuntu 22.04，`NVIDIA GeForce RTX 5090`，驱动 `580.105.08`，CUDA build `12.8`。
- Conda 环境：`/root/autodl-tmp/conda_envs/future-l1-latent`，Python `3.10.21`，PyTorch `2.9.1+cu128`，Transformers `4.57.1`，Accelerate `1.10.1`。
- 模型：`/root/autodl-tmp/models/future-l1`，约 17G；NLI 模型 `nli-deberta-v3-xsmall`，约 281M。
- 数据：`cut_0.mp4`、`cut_1.mp4`、`cut_2.mp4` 及此前 public-short 样本；数据目录约 57M（不含模型）。
- 结果：`/root/autodl-tmp/results/future-l1-latent`，约 58M。
- 当前 GPU 显存使用为 0 MiB，未发现 `run_complete_search`、Future-L1 或其他实验进程；迁移后的任务均已结束。
- 远程 Git 分支为 `experiment/l1-latent-perturbation`，工作树有 `src/model/future_l1.py` 修改及 `experiments/` 未跟踪内容；这不是已提交的干净复现实验版本。

## 已完成实验

| 结果目录 | 设置 | 实际产物 | 观察 |
|---|---|---:|---|
| `public-short` | 3 个公开视频，baseline + 5 档 sigma × 2 seed | 33 次 | Sintel 3、Bunny 4 次答案改变，Flower 0 次；只是 latent/文本敏感性测试 |
| `complete-search` | 3 个视频，各 22 个候选、1 轮反馈 | 66 个候选 | 旧版事件选择路径可运行；本轮不要求轨迹解码 |
| `active-round3` | 各视频 45 个候选，sigma `{0.03,0.1,0.2}`，2 seed，1 轮反馈，branch cap 3 | 135 个候选 | retained 分别为 2、2、1；目标主体约束和 NLI 去重路径可运行 |
| `round3-event-aware-v2` | 各视频 200 个候选，sigma `{0.01,0.03,0.1,0.2}`，3 seed，2 轮反馈，branch cap 5 | 600 个候选 | 三个命令均 `EXIT=0`，耗时 1048/1022/1029 秒；retained 分别为 5、1、1 |
| `round3-high-sigma-stress` | 各视频 30 个候选，sigma `{0.3,0.5,0.8,1.0}`，seed 101，无反馈 | 90 个候选 | 三个命令均 `EXIT=0`，耗时 155/154/152 秒；retained 分别为 1、0、1 |

round3-event-aware-v2 的每个视频都满足：报告与 raw/latent 文件哈希一致、latent 非空且有限、两次零噪声 baseline 的 raw 与 latent 一致。直接 CPU/CUDA 单元检查也通过：零噪声精确 no-op、相同 seed 一致、不同 seed 改变 active vector、inactive vector 不变、数值有限。

## 结果解释边界

- round3 输出文本事件候选和保存的 latent；`trajectory_decoder` 为 `null` 是符合本轮第一阶段范围的预期状态，不作为失败证据。
- `retained_count` 是筛选后的候选数，不是概率质量；NLI 相似度也不是校准概率。
- cut_0 虽得到 7 个 exact answer，最终 5 个 retained 中多条只是“移动/放置在不同位置/准备使用”的近邻表述；cut_1 的 4 个 exact answer 最终只留 1 条，cut_2 的 3 个 exact answer 最终只留 1 条。
- 高 sigma 压力测试没有显示稳定的事件覆盖提升，cut_0 全部同一答案，cut_1 无候选通过结构筛选，cut_2 仍由 baseline 事件占主导。
- 这批样本不是 Ego4D/SSv2 的正式协议，不能推出数据集级事件模式覆盖、模式概率质量或跨场景泛化。

## 代码与测试状态

round3 运行时记录了 `active_latent_generation=true`、`shared_prefill=true`、`branch_cache_batching=true`，并保存了代码 SHA-256。代码级检查 `test_complete_scheme_v2.py`、`test_multibranch_generation.py` 和 `test_latent_noise_unit.py` 通过；环境未安装 `pytest`，因此没有 pytest 测试套件结果。

## 当前阻塞与下一实验门槛

1. 先重新定义高层事件等价与粒度，区分对象、动作、关系和事件顺序，避免把语言改写计作独立模式。
2. 保留重复 baseline、反馈零扰动和相同反馈提示对照，量化扰动与反馈各自带来的新增有效事件数。
3. 使用有明确目标物体和未来事件标注的数据协议，报告事件覆盖、近义重复率、目标一致性和跨视频稳定性。
4. 轨迹 decoder、轨迹监督和物理合理性评价移到第二阶段；本轮不以这些内容作为实验门槛。
