# L1 扰动实验：结果复核与事件分支判断

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-10
- Verification Status: ANALYZED
- Version Label: l1_perturbation_audit_v1
- 工作范围：复核用户提供的 33 次运行产物、视频与统计；未重新运行 L1。
- 原始目录：D:\Pycharm\post_graduate_project\experiences_results\L1-perturbation

## 1. 当前方法决定

用户确认直接事件分支为当前主线，去掉主线中的 MotionMap 式模式图，模式图版本保留为备选。此决定不自动选定分支学生、训练损失或部署方式。第一阶段独立评判器的省略沿用上一轮讨论状态，第二阶段轨迹评判器未调整。

## 2. 结论

这批产物支持“加噪改变了记录的 latent，并在部分运行中改变生成文本”。尚未展示足够清晰、相互区分、与目标物体有关的新增事件分支，也未验证这些 latent 对三维轨迹预测的帮助。

对最终文本及完整 reason 的人工检查发现：Bunny 主要围绕兔子与啮齿动物互动、冲突等笼统叙事，Sintel 主要围绕人物采取行动、透露信息或遭遇挑战，变化存在部分语义侧重点，但描述宽泛且重叠，未稳定分离出具体未来事件。Flower 一直预测开花，与记录后续基本一致；它适合稳定性检查，不宜被要求强行产生多种未来。

判断标准只要求高层动作或交互结果可区分，不要求文字表达精确速度、距离、旋转角或三维轨迹。未将字面不相同直接计为新模式，也未用唯一记录未来排除所有合理替代未来。

## 3. 实验条件与证据边界

- 记录中的模型为 Eurayka/Future-L1-SFT，不是 RL 权重的效果验证。
- 每视频一条 sigma=0 baseline，五档 sigma（0.005、0.01、0.03、0.1、0.2），每档 seed=11、22，共 11 次；三个视频合计 33 次。
- 记录设置为 bf16、贪心文本解码、8 帧输入、256 新 token 上限；只对 active latent 施加按 RMS 缩放的独立高斯噪声。
- 文件记录展示独立 sigma/seed 运行；未包含 SemDiD 式事件前瞻、联合分支保留及反馈搜索结果。因此本次评价的是随机提议环节，未检验完整主动搜索方案。
- 输入是 2.002 秒 Flower、15 秒 Bunny 预告片前缀、25 秒 Sintel 预告片前缀。后两者含字幕、剪辑与场景切换，不是连续人—物交互样本。
- 问题统一为 What will happen next in the video?，没有指定目标物体或明确未来时域，容易得到剧情级泛述。
- 提供目录没有实验源代码、模型 revision/权重哈希及实际送入模型的帧索引/张量。本轮根据文件和记录核对，不将其写为完整实验复现。

## 4. 独立复算

| 样本 | 非零扰动次数 | latent 数值改变 | 原文改变 | 完整答案改变 | 非零输出不完整 | 完整答案字面种类，含 baseline |
|---|---:|---:|---:|---:|---:|---:|
| Flower | 10 | 10 | 0 | 0 | 0 | 1 |
| Bunny | 10 | 10 | 4 | 3 | 1 | 4 |
| Sintel | 10 | 10 | 3 | 3 | 0 | 3 |
| 合计 | 30 | 30 | 7 | 6 | 1 | 不跨不同视频累加为模式数 |

- 所有 33 个保存数组均为有限数值，32 个输出包含完整 answer 标签。
- Bunny sigma_0.005_seed_22 达到 256 token 上限，具有 answer 起始标签但缺少结束标签。原汇总的五种答案字符串包含这个截断结果；完整答案是四种字面版本。另一个 Bunny 运行也达到 256 token，但已完成答案及终止标记，不能仅据 token 数判断损坏。
- 所有 33 次运行在首次 latent 之前的 reason 与各自 baseline 一致，文本差异发生在后续部分。
- 保存数组复算的 MSE、平均余弦和 relative L2 与汇总的对应指标全部在设定数值容差内一致。
- Flower 在 sigma=0.2、seed=22 时 relative L2=0.244845、平均余弦=0.975411，答案仍完全相同；Bunny 在 sigma=0.2、seed=11 时 relative L2=0.427713，答案也与 baseline 相同。数值差异本身不等价于事件差异。
- sigma 与文字变化没有单调关系：Bunny 的较小扰动也会改变文本，而两次 sigma=0.2 均保持 baseline 原文。当前数据不支持选择“最佳 sigma”。

### 文件核查

- 对源目录 115 个文件建立 SHA-256 清单，分析结束时全部保持不变。
- 三个观察视频 SHA-256 与各自 metadata 及实验记录一致。
- 33 份 raw.txt 均与单次 JSON 完全一致，文本哈希全部匹配；单次 JSON 与 summary 对应字段没有差异。
- latent SHA-256 在保存的 float16 数组转回 float32 后有 29/33 匹配，剩余四个不匹配：Flower 0.005/22，Bunny 0.01/11，Sintel 0.005/11 与 0.005/22。保存精度变化是可能原因，缺少保存前数组和哈希实现，原因未确证。该差异不改变上述数值距离核对结果，不将其直接解释为文件损坏。

## 5. 逐组事件判断

以下分组依据完整文本进行人工审阅，属于定性分析，未使用外部模型评分。

### Flower

- 全部输出相同：“The flower will fully open, revealing its bright red petals and yellow center.”
- 输入复核帧显示花苞逐渐张开；记录后续显示红色花瓣进一步展开、黄色花蕊显露。
- 这是合理的主要未来，未产生新增分支。稳定输出与场景本身较明确的变化相容，单凭本例不能认定模型忽略了所有潜变量信息。

### Bunny

| 运行 | 最终预测内容 | 判断 |
|---|---|---|
| baseline 及六个非零运行 | 继续介绍三只啮齿动物，与兔子互动，可能冲突或冒险 | 一组宽泛叙事，缺少独立目标动作 |
| 0.005 / 11 | 啮齿动物与兔子互动，可能冲突或合作 | 加入“合作”这一方向，但仍与冲突并列，未单独形成合作分支 |
| 0.005 / 22 | 兔子遇到或互动，潜在对抗或重大事件 | 输出截断；与其它回答大量重叠 |
| 0.01 / 11 | 兔子遇到或互动，潜在对抗或惊讶 | 有侧重点变化，事件定义仍宽泛 |
| 0.1 / 11 | 兔子遇到啮齿动物，互动或冲突 | 与主要输出高度重叠 |

部分变化修改了对输入画面的陈述：

- baseline 将最后画面描述为“小啮齿动物特写”；
- 0.005 / 11 将其描述为“三只啮齿动物站在一起”；
- 其它变化版本描述为“大兔子看起来惊讶”。

人工抽取的观察视频 14.920 秒帧仍显示大兔子。三只啮齿动物集中出现的画面位于提供原片的 15 秒附近，即观察边界之后。因而前两种陈述与所复核的观察末段不符，部分其它变体反而更接近观察内容。不能把这种观察解释的变化全部记为新未来，也不能把所有错误都归因于扰动，因为 baseline 本身已有问题。

### Sintel

| 运行 | 最终预测内容 | 判断 |
|---|---|---|
| baseline 及七个非零运行 | 红发人物将采取重要行动或作出决定 | 很宽泛 |
| 0.01 / 22、0.03 / 22 | 室内人物可能透露信息或面临挑战 | 有语义侧重点变化，但两个可能事件混在一句 |
| 0.2 / 22 | 人物将发挥重要作用，可能透露信息或采取行动 | 与前两类包含、交叠 |

观察视频 24.920 秒已转为室外城市远景，而输出主要围绕更早的室内人物与故事意义。记录后续出现飞行动物及人物反应，但没有理由将与该唯一后续不同的候选一律判错；这里主要问题是候选过于笼统，未形成清晰且可用于目标轨迹条件的事件分支。

## 6. 与预期逐项对应

| 预期 | 当前证据 |
|---|---|
| latent 扰动产生数值影响 | 保存数组与噪声统计支持 |
| 扰动能够影响后续生成 | 部分完整答案发生变化 |
| 新增明确、可区分的事件模式 | 本批输出尚未充分展示 |
| 新增事件与观测相容 | 未系统验证，且存在观察陈述错误 |
| 多分支可作为第一阶段高质量监督 | 当前产物尚不足以支持 |
| latent 保留对第二阶段有用的运动信息 | 没有轨迹解码或下游评价证据 |
| SemDiD 联合主动搜索有效 | 本次未包含该环节 |

本次适合归为“数值与文本敏感性试验”。保留随机扰动提议的研究方向，但不把这一批答案种类数当作已实现的有效事件覆盖。样本、提问、SFT 权重和未接入事件搜索共同限制解释范围；不据此否定完整 L1 主动分支方案。

## 7. 需要更正的记录解释

原记录由 zero_noise_deterministic=true 推出 baseline 可复现，证据过强。每个样本仅一条零噪声结果，该布尔值在现有文件中没有重复运行支持。准确记录应为“贪心零噪声基线已运行一次，重复一致性尚未验证”。

all_runs_finite=true 支持数值稳定性，不认证文字、视觉状态或物理合理性。答案字符串变化数也不直接对应事件模式数。

## 8. 统计解释检查

检查覆盖 11/11；本分析不进行总体显著性检验，不给出伪精确置信区间。

| 项目 | 处理 |
|---|---|
| Simpson 分组问题 | 分视频报告，不将样本差异混成统一改善趋势 |
| 生态推断 | 不由三视频汇总推广至任意对象 |
| 选择偏差 | 明确是便利公开视频，含动画剪辑，代表性有限 |
| Collider 偏差 | 无回归控制；未作该类因果分析 |
| 基础率忽略 | 不将候选数量或出现频次解释为真实事件概率 |
| 均值回归 | 无受支持的前后干预改善结论 |
| 幸存者偏差 | 保留截断输出及所有 sigma，不只统计格式成功样本 |
| 多重探索 | 报告全部五档两种子，不挑最佳 sigma |
| 分析自由度 | 定性事件判断标准明确，属于事后探索 |
| 因果夸大 | 扰动影响依据记录控制条件；完整代码和重复基线未核验 |
| 反向因果 | 扰动为生成前干预，未提出新的反向因果结论 |

独立输入为三个视频，30 个非零运行共享输入、部分共享噪声种子，不将它们当作30个独立场景。

## 9. 复核产物

- [逐运行统计 CSV](D:/Pycharm/post_graduate_project/reports/analysis/2026-09-10-l1-perturbation/run_audit.csv)
- [结构化复核结果](D:/Pycharm/post_graduate_project/reports/analysis/2026-09-10-l1-perturbation/audit_summary.json)
- [源文件哈希清单](D:/Pycharm/post_graduate_project/reports/analysis/2026-09-10-l1-perturbation/source_manifest.json)
- 本地只读分析脚本：D:\Pycharm\post_graduate_project\reports\analysis\2026-09-10-l1-perturbation\audit.py
- 执行命令：& 'C:/Users/Andcher/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' 'reports/analysis/2026-09-10-l1-perturbation/audit.py'；工作目录为项目根，PYTHONHASHSEED=0、PYTHONUTF8=1。
- 分析执行 exit=0；CSV_ROWS=33；CONSISTENCY_ISSUES=0（单次 JSON 对 summary）；SOURCE_FILES_UNCHANGED=True。这不是远程模型实验重跑。
- 当前下一讨论位置：依据上述证据判断随机提议如何服务于有效事件分支，不扩大为新的架构决定。

### 输入与后续抽帧

以下为独立人工复核用均匀抽帧，不宣称与实际 qwen_vl_utils 输入索引完全相同。后续只用于检查记录事件，不作为模型输入。

![Flower 观察](D:/Pycharm/post_graduate_project/reports/analysis/2026-09-10-l1-perturbation/flower_observation.jpg)
![Flower 后续](D:/Pycharm/post_graduate_project/reports/analysis/2026-09-10-l1-perturbation/flower_recorded_future.jpg)
![Bunny 观察](D:/Pycharm/post_graduate_project/reports/analysis/2026-09-10-l1-perturbation/bunny_observation.jpg)
![Bunny 后续](D:/Pycharm/post_graduate_project/reports/analysis/2026-09-10-l1-perturbation/bunny_recorded_future.jpg)
![Sintel 观察](D:/Pycharm/post_graduate_project/reports/analysis/2026-09-10-l1-perturbation/sintel_observation.jpg)
![Sintel 后续](D:/Pycharm/post_graduate_project/reports/analysis/2026-09-10-l1-perturbation/sintel_recorded_future.jpg)
