# 第6帧query的名义密度采样准备

日期：2026-10-10

状态更新：用户已确认当前query取样无问题，批准记录见同结果目录`query-approval.json`。下方未批准字段为准备阶段历史状态；GPU授权仍保持关闭。

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent；research-progress-logger
- Origin Mode: CPU preparation / code verification
- Verification Status: VERIFIED（采样、真值接口、官方CPU预处理与回传文件）；GPU性能未测试
- Overall Confidence: CAUTION（名义密度近似，作者完整采样入口未公开）

## 用户决定与实现

- 用图像大小和官方约1000点推算名义像素间距，256×256时为sqrt(65536/1000)=8.09543081px；252×252预处理后7.96893971px。
- 固定整幅图网格：半个网格宽度为起点，向下取整到像素；整幅图实际1024个格点。仅保留第6帧原目标mask内的有效点，不根据预测表现移动网格或筛点。
- 统一沿用原第6帧query及18个目标物体；本轮保留原实验的22帧输入（0..21），用于与旧48点结果对照。最近24帧/首帧稠密候选另存，不将其覆盖到原实验。原有像素中心归一化不变。
- 有效网格点数低于48时，精确复制原queries.json与gt_tracks.npz，SHA256逐文件相同；没有重新抽48点。
- 候选有效性只使用query帧可见性、正深度、画面内UV及GT投影一致性，不使用未来遮挡或预测误差选点。
- 原始48点预测和首帧24帧稠密候选均保留。后者已标为历史候选，当前活动配置指向本轮协议。
- 本轮只生成query、仿真真值与预览；未下载新数据，未加载模型、运行forward、调用GPU或训练。

## 已完成的采样结果

18/18场景成功，共1280点、13个网格采样、5个旧48点精确回退。所有query在第6帧有效且可见，四个数据分片发布MD5通过，处理前后SHA256不变。

| 场景 | query数 | 采样方式 |
| --- | ---: | --- |
| scene_3626 | 48 | 原48点 |
| scene_1782 | 93 | 网格 |
| scene_3881 | 77 | 网格 |
| scene_2344 | 72 | 网格 |
| scene_3782 | 48 | 原48点 |
| scene_5916 | 134 | 网格 |
| scene_1924 | 60 | 网格 |
| scene_5025 | 68 | 网格 |
| scene_1143 | 68 | 网格 |
| scene_4203 | 68 | 网格 |
| scene_2854 | 48 | 原48点 |
| scene_1179 | 48 | 原48点 |
| scene_813 | 77 | 网格 |
| scene_346 | 71 | 网格 |
| scene_5702 | 50 | 网格 |
| scene_2329 | 87 | 网格 |
| scene_3457 | 48 | 原48点 |
| scene_1580 | 115 | 网格 |

本地5项密度采样测试、7项GT/接口测试、22项公开评测协议测试通过；远端5项采样及7项GT/接口测试通过。合成场景测试覆盖第6帧固定点身份、22帧GT、任意N>=48接口和精确48点回退。输入哈希、query归一化往返及GT点身份对应均通过。

18/18场景的官方CPU图像预处理通过，均为(22,3,252,252)，未构造模型或访问CUDA。首次检查超过60秒连接等待窗口仍在执行，远端CPU配额为0.5核、进程167线程；已结束该超时检查，在环境变量及Torch中限制1线程后复核成功。没有发生模型推理失败。

query预览包约7.75MiB，SHA256为`1ac37bba7eafb03fa374553cd6e0bcbf03a34445ff569754b06a789c8030d566`；本地77个审阅文件逐项哈希验证通过。已查看总览，网格位置覆盖第6帧原目标可见区域。新query是否批准仍由用户确认。

## 后续官方验证集对照设计

此项为后续计划，当前未下载或运行官方验证序列。

1. 获取作者所用WorldTrack验证集的小子集，记录原序列ID、来源、相机真值字段与单位；优先选择明确目标物体。项目任务仍为物体预测。
2. 同一段至少64真实帧的序列，固定其query点身份，分别输入前22帧与前64帧。禁止循环、复制或拼接MOVi-F的24帧凑成64帧。
3. 为隔离帧数效应，两种输入都沿用同一query帧与点集；主比较仅评分共同前22帧，并用共同窗口固定动态点集合及GT筛选规则。每次预测按同一公开公式计算尺度，同时报告固定t0尺度诊断，防止将尺度差异误认为轨迹改善。
4. 官方query0/64帧完整协议另行报告；query0与当前query6的影响用同一官方序列的独立对照检查。若query6需要新的GT材质点，不能仅重命名首帧query0标注。
5. 64帧完整官方指标作为另一个结果报告，不能将其与22帧不同时间窗口的平均误差直接相减来归因于帧数。
6. 对照顺序：同一数据内控制输入长度；同一数据与长度内控制query帧/采样和归一化；最后在匹配条件下比较数据集。预先固定序列，不按模型表现挑样本。小子集用于定位差距，不能宣称复现完整200段论文均值。

## 文件

- 活动配置：`configs/uniquery4r_movif_density_query6_2026-10-10.json`
- 采样脚本：`scripts/prepare_uniquery4r_movif_density_query6.py`
- 推理入口：`scripts/run_uniquery4r_movif_experiment1.py`支持任意N>=48；新增可选公开动态点评分，GPU与query批准参数仍为必要条件。
- 预览导出：`scripts/export_uniquery4r_density_query_review.py`
- 远端数据：`/root/autodl-tmp/uniquery4r-movif-density-query6-2026-10-10`
- 本地预览与证据：`experiences_results/uniquery4r-movif-density-query6-2026-10-10`
- 审阅页：`experiences_results/uniquery4r-movif-density-query6-2026-10-10/query-review.html`
- CPU证据：同目录`cpu-test-evidence.json`、`logs/official-preprocessing-cpu.json`、`remote-final-gt-interface-tests.log`；未批准新query前不运行GPU。

academic-research-suite用于保持采样变更、CPU证据和未执行性能实验之间的界限；research-progress-logger用于记录本轮工作。
