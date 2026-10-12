# 正式 E2 协议冻结与参数决策

日期：2026-09-22

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent / research-progress-logger
- Origin Mode: plan
- Origin Date: 2026-09-22
- Verification Status: ANALYZED
- Version Label: formal_e2_protocol_v1

## 总体决定

正式 E2 采用 Ego4D 作为唯一主数据源，目标收集 6 个真正独立的视频，最低接受 5 个。SSv2 暂不混入主实验，因为当前没有与 Ego4D 等价的相机、深度和 TrackCraft3R 三维伪标签链；之后可作为独立外部验证集。

主图为 2D symmetric kNN，`k=3`；最小对照为 observed 3D symmetric kNN，`k=3`；hybrid 暂缓。

## 视频选择

### 纳入数量

- 目标：6 个 source video；
- 最低：5 个 source video；
- 每个 source video 只取 1 个正式窗口；同一视频的 prompt、stride、起始窗口和重复推理不增加独立样本数。

### 数据来源

- 使用 Ego4D 中连续的人—物交互片段；
- 保留已有的 sewing-scissors 和 watering-can 视频，重新按本协议截取窗口；
- 排除夜间低照度 car 视频和当前 1.64 秒的本地 bottle 测试视频，它们不满足目标可见性或时间窗口条件；
- 新增视频按 Ego4D 元数据排序后，以固定规则选择，不根据 E2 结果挑选。

已有视频的 UUID 为 `cb78dba8-6647-4ec7-875c-6ba1548f8312`（sewing-scissors）和 `356ed369-7023-4660-904b-be73823a21d3`（watering-can）。新增候选按以下预筛选顺序处理：片段长度至少 2.0 秒、无切镜、目标在首尾帧可见、目标面积约占图像 `3%--60%`、三帧历史的有效深度比例至少 90%。在每个运动类别配额内按 `(source_video_uuid, segment_start)` 排序，取最先满足条件的候选；不得根据已计算的 E2 误差回选视频。

### 目标运动覆盖

优先构成：2 个 rigid、2 个 articulated、2 个 non-rigid/deformable 物体或运动片段。若 Ego4D 中无法得到足够可靠的 deformable 三维轨迹，最低接受 5 个视频，但必须至少覆盖两个运动类别，并在报告中声明类别限制。

### 片段条件

- 原视频在目标窗口内连续、无切镜；
- 目标在窗口首尾均可见，且不是完全被手遮挡；
- 窗口长度固定为 2.0 秒；
- 处理帧率固定为 15 fps，通过原始时间戳选择最近帧，不用文件帧号直接假设帧率；
- 窗口中点作为观测结束时刻 `t0`，不根据未来轨迹质量移动 `t0`；
- 观测历史为 `t0-2/15 s, t0-1/15 s, t0` 三帧；
- 评估未来为 `t0+j/15 s, j=1,...,9`，预测与 pseudo-GT 必须在这一时间网格上对应。

## Query 采样

### 主数量

- 每个视频先生成 64 个候选 query；
- 主结果固定使用 48 个 query；
- 从同一 64 点集合取嵌套的前 32 点和前 48 点，分别作为密度敏感性分析；64 点作为高密度上限。

这样可以在不重复运行三次跟踪器的情况下比较 `N=32/48/64`，并保持不同密度之间的点身份关系。

### 采样区域

- 在 `t0` RGB 帧上对目标物体做人工审核的 polygon mask；
- mask 向内腐蚀 2 个处理分辨率像素，排除明显背景混合边界；
- 候选像素要求在三帧历史中均有有限、正深度，并位于有效 TrackCraft3R 区域；
- 以目标 mask 质心最近的候选像素为起点，使用归一化目标 bounding-box 坐标做 deterministic farthest-point sampling；
- 距离相同按 `(y,x)` 字典序打破平局；
- 最小候选间距为 `max(4 px, 0.02 × bbox_diagonal)`；
- 先取得 64 个候选，再按 FPS 顺序截取前 32、48、64 个。

### Query 有效性

- 正式主结果要求至少 48 个 query 通过历史有效性检查；不足则该视频不能进入主 E2；
- 每个 query 的 9 个未来 pseudo-GT 时刻至少有效 8 个；
- 每个 horizon 的有效 edge 覆盖率低于 80% 时，该视频窗口排除或单独列为失败，不进行静默插值；
- 所有 query、mask、有效性和采样顺序写入 manifest。

## 三维与时间协议

- 统一使用相机坐标系，单位为米；
- 2D query 使用 `t0` 处理帧坐标，3D history 使用同一 `t0` 相机坐标；
- 源视频统一重采样到 15 fps，并保存每个处理帧对应的原始时间戳；
- `future_3d` 与 pseudo-GT 按时间戳映射到同一个 9 步网格，禁止继续使用未核验的“数组前 9 步直接对应”假设；
- 若预测模型只输出更密的时间网格，使用时间戳最近邻匹配，距离超过半个目标步长则标记为无效；不做跨越动作边界的长距离插值。

## 实验运行

### 图和随机性

- 主图：2D symmetric kNN `k=3`；
- 对照：observed 3D symmetric kNN `k=3`；
- 每个视频、每个 query density、每个图运行 oracle 及 Gaussian `1/5/10/20%`、outlier `5/10/20%`；
- 每个非 oracle 条件使用 20 个固定随机种子，seed base 为 `20260922`；
- solver 保持 component-aware uniform-gauge L2；本阶段不加入 Huber、IRLS、learned gate 或 hybrid 过滤。

### 指标

主指标：

- normalized-horizon temporal RVE AUC；
- 每个 horizon 的 edge RVE；
- 图连通性、bridge-edge ratio、`lambda_2`、KKT condition number。

次指标：

- component-centered point error；
- ADE/FDE；
- common translation shift；
- 历史帧图 Jaccard、边长度 CV 和范围；
- query density/subset 对共同点预测的方差。

所有结果按 source video、运动类别和 query density 分层；随机种子只用于估计噪声方差，不作为独立视频样本。

## 预注册判据

### 数据闸门

正式运行前必须同时满足：

- 至少 5 个独立视频；
- 每个视频至少 48 个有效 query；
- 至少两个运动类别；
- 时间戳映射报告无未解决冲突；
- 主图和对照图在所有视频上都能构成单一连通分量，或明确报告失败视频。

### 保留 2D `k=3` 的条件

2D `k=3` 继续作为默认图，当且仅当：

- 在至少 5 个视频中没有连通性硬失败；
- 相比 3D `k=3`，主指标的宏平均不差超过 5%，且视频级中位数不在多数视频上恶化；
- 2D 的实现和调参复杂度明显更低；
- query density 从 32 到 48 时没有出现系统性结构误差增加。

若 2D 与 3D 在正式 E2 仍接近，保留 2D 作为主方案、3D 作为附录对照。若 2D 在多个独立视频上恶化超过 5%，只有在 3D 同时满足全局连通性和覆盖要求时才改用 3D；若 3D 断连，则图选择保持未决，先进行预注册的连通性修复对照，不自动切换。

## 2026-09-23 E2-G 结果后的补充决策

E2-G 中 2D `k=3` 的共享评估边 RVE AUC 宏平均比 3D 高约 14.56%，未通过上述保留门槛；但 3D `k=3` 在 5 个视频中的 4 个断连，不能按原规则直接替代 2D。因而原“2D 失败即切换 3D”规则暂停执行。

当前状态为：2D `k=3` 保留为工程候选，3D `k=3` 保留为数值对照，正式默认图暂不冻结。下一步只增加一个 `3D-k3-plus-2D-component-connectors` 连通性修复对照，并补做 48→64 query-density 检查；该对照通过前不进入 E3。

E2-G.1 完成后，候选层级进一步明确：2D `k=3` 仍是主工程候选；`3D-k3-plus-2D-component-connectors` 作为后续模型侧结构消融的次级候选；原始断连 3D `k=3` 只作数值诊断对照。E2-G.1 不足以冻结最终默认图，但已完成当前无模型图结构筛选阶段。

### 进入 E3 的条件

正式 E2 只决定图和数据协议，不证明 Relation Head 可学习。只有在 MotionForesight observed-only `P0,H` 可缓存后，才进入 E3；E3 仍需比较 zero correction、geometry-only 和 query-feature head。

## 任务分工

### 代码 AI

- 下载/整理固定 manifest 中的 Ego4D 视频；
- 生成 15 fps 时间戳索引和 2.0 秒窗口；
- 读取人工 polygon mask 并生成 64/48/32 query manifest；
- 运行 TrackCraft3R、生成 pseudo-GT 和 baseline future tracks；
- 在正式运行前分别用 16、48、64 个 query 做一次接口 smoke test；若模型或处理器不支持可变 query 数，停止并报告接口限制，不得退回 8 点后把结果标为正式 E2；
- 执行 2D/3D `k=3` E2、20 个噪声种子、全部指标和本地/AutoDL 复跑。

### 研究审查

- 审核视频是否真正独立、窗口是否无切镜、mask 是否指向同一目标；
- 核对时间戳、相机坐标、单位、query 有效性和未来信息边界；
- 复核分层汇总和预注册判据，决定是否保留图并放行 E3。

## 当前阻塞

需要新增 3--4 个满足条件的 Ego4D source video，并为每个视频提供 `t0` 帧 polygon mask。现有 two verified Ego4D clips 可重用；car 和短 bottle 测试样本不进入正式 E2。

## 下一步

先由代码 AI 实现数据预检、时间戳网格、polygon-mask query sampler 和 manifest；数据达到闸门后执行正式 E2。
