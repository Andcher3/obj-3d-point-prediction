## Material Passport
- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-10
- Verification Status: ANALYZED
- Version Label: round-two-complete-search-audit-v1

# L1 新视频与反馈搜索实验复核

## 结论

目前尚未达到“稳定生成多个事件不同、与当前观测和指定物体相容的未来候选”的预期。重复基线、完整输出和反馈调用已经得到本轮记录支持；语义去重、候选质量判断以及生成过程中的事件级协同搜索仍未实现到此前讨论的程度。

cut_0 的反馈轮提供了有限的正面信号：同一反馈提示下，改变扰动会产生“拿起并移动”“移动”“放下”三种答案。不过，其中存在事件包含关系、动作阶段差异和观测解释错误，尚不足以确认为三个合格未来模式。cut_1、cut_2 的最终三个候选分别完全重复。

本结论来自第一阶段本身的检查。第二阶段 decoder 尚未接入是独立的未验证事项，并非本轮第一阶段不达预期的判断依据。当前直接事件分支为主线、MotionMap 模式图为备选的决定保持不变。

## 1. 复核材料与执行边界

- 新结果：D:/Pycharm/post_graduate_project/experiences_results/L1-perturbation/complete-search
- 实际代码目录：D:/Pycharm/post_graduate_project/experiences_results/l1_latent_perturbation_code
- 实验记录：上述代码目录的 EXPERIMENT_RECORD_2026-09-09.md，第 11 节。
- 共核查 153 个源文件：三个输入视频、66 份 raw、66 份 latent、3 份 JSON 报告、3 份日志及代码目录 12 个文件。复核前后 SHA-256 全部一致。
- 读取全部 66 个候选；重算 693 对比较、三组最终选择及反馈提示；逐一核对 JSON 与 run.log；观看三个视频各 12 张复核帧。
- 本地执行只包含文件解析、数组比较、调用供应代码的纯 CPU 选择函数和小型反例测试。未运行 L1 推理、未重新训练、未修改供应代码和实验结果。
- 原有 public-short 结果未重新计入本轮。

复核入口：D:/Pycharm/post_graduate_project/reports/analysis/2026-09-10-l1-complete-search-audit/audit.py

执行命令：

    $env:PYTHONHASHSEED='0'; $env:PYTHONUTF8='1'; & 'C:/Users/Andcher/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' 'reports/analysis/2026-09-10-l1-complete-search-audit/audit.py'

退出状态：0。核心输出：candidate_total=66；source_files=153；source_files_unchanged=true。这属于产物分析和选择逻辑复算，不构成模型推理复现。

## 2. 数值与运行记录

每个视频有 12 个初始候选（2 个零噪声基线、10 个扰动）及 10 个反馈候选。

| 项目 | cut_0 | cut_1 | cut_2 |
|---|---:|---:|---:|
| 候选总数 | 22 | 22 | 22 |
| 两次基线 raw/数组完全相同 | 是 | 是 | 是 |
| 完整 answer、latent 标签、结束标记 | 22 | 22 | 22 |
| 非空且有限的保存 latent | 22 | 22 | 22 |
| 初始 10 个扰动 latent 相对基线改变 | 10 | 10 | 10 |
| 初始扰动答案相对基线改变 | 0 | 0 | 1 |
| 初始轮不同答案字符串数，含基线 | 1 | 1 | 2 |
| 反馈轮不同答案字符串数 | 3 | 1 | 1 |
| 最终保留槽位数 | 3 | 3 | 3 |
| 最终不同答案字符串数 | 3 | 1 | 1 |

所有保存数组均为 float16、形状 (4,4096)。初始扰动相对基线的 L2 范围分别为 0.02255–0.22801、0.02175–0.22363、0.02369–0.21834。数值变化不等价于事件变化。

三个视频的 231 对比较、最终选择及反馈提示全部复算一致，JSON 与对应日志一致。三个最终候选集合内部各有 3 对比较，九对全部被供应比较器标记 same_event=true；cut_0 的相似度为 0.5625、0.6923、0.8，另两个视频为 1。

本轮补齐了零噪声重复对照，也没有上轮的截断答案。重复性结论仅覆盖当前各视频两次基线，尚未覆盖跨设备或跨软件版本运行。

## 3. 逐视频事件复核

### cut_0：存在动作表述变化，尚不能计作三个可靠模式

目标配置：the pink rectangular cutting board。

初始 12 个答案完全相同，均预测：

> The person in green gloves will pick up the blue cutting board and move it to a different location on the countertop.

这里直接预测了蓝色物体，偏离粉色目标。其首段 reason 也已经写出该未来动作，并把红、绿手部分别解释成不同的人。复核帧显示同一第一视角下两只不同着色的手在协作，不能仅凭颜色确认有两个人。

反馈轮 10 个答案分布：

- 6 条：红色手套的人拿起粉色物体并移至台面另一位置。
- 1 条：红色手套的人将粉色物体移至台面另一位置。
- 3 条：红色手套的人将粉色物体放在台面上。

三种最终答案分别来自 r1_09、r1_01、r1_07。第一、二种可以表达同一搬移动作，第三种也可能是该动作的结束阶段。它们没有明确给出共同预测时段下三个分离的事件结果。输入末段粉色物体已经在手中，“拿起”的时间定位亦需谨慎，不能把动作阶段变化自动当作模式发现。

有限正面证据：反馈轮在相同提示、相同解码配置下，扰动配置不同会得到不同的动作动词组合。保留这一观察，但不将 6:1:3 解释为事件概率。

反馈轮首段 reason 原文称绿色手套的人正在拿起并移动蓝色物体，随后粉色物体进入下一步。这与反馈提示中的已有候选十分接近，提示“候选被当作事件延续”的风险；尚未通过消融确定其因果来源。

![cut_0 复核帧](D:/Pycharm/post_graduate_project/reports/analysis/2026-09-10-l1-complete-search-audit/cut_0_review.jpg)

### cut_1：持续清洁，未出现额外事件模式

22 条答案全部完全相同：继续用紫色物体清洁橙色碗内部。最终三个槽位也完全重复。

输入后半段可见手在碗内持续操作，因此这个回答与可见动作有一定相容性。但视频带有明显的大面积纯色覆盖，工具材质和实际操作类型的判断存在不确定性，不能仅据答案确认是清洁。无论其单次预测是否正确，本轮都没有显示多个事件分支。

![cut_1 复核帧](D:/Pycharm/post_graduate_project/reports/analysis/2026-09-10-l1-complete-search-audit/cut_1_review.jpg)

### cut_2：只有“用刀”的措辞增补

21 条答案为“开始切砧板上的食物”；初始 r0_05 额外加入“with the knife”。反馈轮十条与原基线完全相同。这属于同一动作的工具说明变化。

复核帧中，双手主要在碗上方处理手持物，不能据此确认下一步一定转为砧板切割。目标配置同时包含 food 和 knife，尚未精确对应单个待预测物体；因此即使回答切割动作，也未说明具体哪个物体将如何运动。

![cut_2 复核帧](D:/Pycharm/post_graduate_project/reports/analysis/2026-09-10-l1-complete-search-audit/cut_2_review.jpg)

三个视频均为 8 秒、500×282、400 帧。复核图为独立抽帧，未声称与模型实际选择的 8 帧完全相同。当前 bundle 没有实际输入帧索引、无纯色覆盖的对应视频、观察段之后的真实后续及同条件多未来标注。因此不进行完整准确率或覆盖率评分，也不把未出现在真实记录中的未来自动判错。

## 4. 影响结论的实现问题

### 4.1 事件距离仍为词集合 Jaccard

complete_scheme.py:23–54 将词去停用词后计算集合交并比，没有双向蕴含或其他事件语义模型。目标词被加入每个答案的词集合，也不构成对回答对象正确性的检查。

直接调用现有函数得到：

- “The hand lifts the cup.” 与 “A person picks up a mug.”：similarity=0.0。近义动作因词汇不同被拉开。
- “The hand lifts the cup then places it on the table.” 与交换动作顺序的句子：similarity=1.0。词相同但顺序不同被完全合并。

这两个反例仅用于定位机制缺口，不是语义模型基准测试。代码内 semantic_scorer 接收单条候选并给质量分，未替换事件对之间的 Jaccard；仅填写该回调亦不会解决事件等价比较。

### 4.2 固定填满槽位，same_event 未参与拒绝重复

complete_scheme.py:140–160 在剩余候选存在时持续填满 branch_count。compare_events 的 same_event 只记录在报告中，没有成为排重条件。

四个完全相同答案的测试保留 b3、b2、b1，唯一答案数=1。因此最终“3 个分支”是配置槽位数。真实结果 cut_1、cut_2 复现了此现象；cut_0 三条答案也全被自身比较器标为 same_event=true。

### 4.3 实际未使用候选质量判断

BranchCandidate.quality 默认 0.0（complete_scheme.py:68）；run_complete_search.py:180–194 创建候选时未填写，199–207 调用搜索时未传入 semantic_scorer。

首条选择全部同分，由 branch_id 的字典序决定；后续只剩词差异项参与评分。同分时同样由 ID 决定。最终各视频首条均为 r1_09_sigma_0.2_seed_22，并不证明 sigma=0.2、seed=22 最优。

这不意味着必须恢复一个独立大模型判别器。它说明当前运行没有实际执行代码文字所描述的“质量与多样性联合判断”，格式有效也不包含目标正确或物理合理。

### 4.4 反馈发生在完整生成之后，尚非 latent 内部联合搜索

run_complete_search.py:152–168 逐条调用 model.generate；每个视频执行 12+10 次完整生成。complete_scheme.py 将已完成答案写入新问题，随后从视频与新提示重新生成。已保留 latent 没有作为下一轮的父状态输入；未共享/维护分支 KV cache，也未根据事件反馈修改当前生成中的 latent。

现有路径验证了“独立扰动生成→集合选择→文字反馈后重新生成”的编排。此前设想的“一次输入维护多个活动分支、在生成中促使事件分离”尚未由此验证。批量或单请求输出多个候选也仍需多次自回归计算，不应称为一次神经网络前向。

### 4.5 缺少反馈提示下的零扰动对照

r1 所有候选 sigma>0。cut_0 的对象纠正、与初始轮的答案变化同时伴随提示改变；尚未测量仅改变提示、sigma=0 的结果。

在 r1 内同一提示下三种答案的差异支持“扰动对该提示下的输出有影响”；r0→r1 的整体变化不能全部归因于扰动有效性提升。

## 5. 新出现的机制线索

每个视频、每轮内部，进入第一个 latent 前的文本完全相同（6 组均只有 1 种首段）。三个初始首段均已包含未来动作判断：cut_0 明确预测移动蓝色物体；cut_1 写出清洁意图；cut_2 写出准备切割。

这修正了“第一个 reason 始终只概括观察”的过强理解。本次实际输出有提前决定动作的迹象。后续 latent 扰动可能受到已有文字条件牵引而继续同一事件；此为有原文依据的机制假设，尚无改变首段/扰动位置的对照来证明因果。它也不意味着应在本轮自动改为首段文本分叉。

## 6. 记录与复现边界

- 供应 bundle 包含搜索编排，但不含实际加噪函数所在 src/model/future_l1.py 和 monkey patch 文件；精确加噪位置本次只能依据记录，不能从 wrapper 独立核实。
- 新 complete_search_report 未保存完整解析参数、模型 revision、token 数、每步噪声统计、输入视频哈希/实际帧索引和每候选质量。部分配置可从代码默认值及文件名重建，不能一律当作运行现场元数据。
- 当前数组全部非空；但 valid_format 只检查 np.isfinite(...).all()，空数组也会通过该条件。反例测试已确认，此隐患未导致本轮无效数组。
- test_complete_scheme.py 使用人工候选和 FakeDecoder，检验回调与数量，未验证近义事件去重、错误对象拒绝或真实三维解码。没有将其 PASS 当作研究效果通过。

## 7. 统计解释检查（11/11）

1. Simpson：分别报告视频与轮次，未用合计掩盖差异。
2. Ecological：66 次生成不等于 66 个独立场景；结论仅覆盖 3 个视频。
3. Selection/Berkson：便利样本、小规模，未外推开放世界性能。
4. Collider：未拟合条件因果模型；不据最终筛选集合推断原始分布。
5. Base rate：不以 6:1:3、相似度或槽位数量估计真实发生概率。
6. Regression to mean：不据更换视频后的变化宣称算法总体提升。
7. Survivorship：检查全部 66 条，包含完全重复候选，不只展示最终 9 条。
8. Look-elsewhere：保留全部噪声设置；不挑选 r1_09 宣称最优。
9. Forking paths：探索性复核；没有事后构造显著性或冻结阈值。
10. Correlation/causation：指出提示改变与加噪并存，缺 r1 零扰动对照。
11. Reverse causality：报告反馈答案影响后续首段的风险，不作已证实因果结论。

## 8. 完成内容、输出与后续待验证项

已完成：源文件一致性、全部候选与数组核对、输入视频复核、选择逻辑重算及机制反例。所有新增文件位于报告目录，供应实现未改动。

产物目录：D:/Pycharm/post_graduate_project/reports/analysis/2026-09-10-l1-complete-search-audit

- audit.py：可复算审计入口。
- audit_summary.json：按视频、轮次汇总和代码反例结果。
- run_audit.json / run_audit.csv：66 条逐候选记录。
- source_manifest.json：153 个源文件 SHA-256。
- cut_0_review.jpg / cut_1_review.jpg / cut_2_review.jpg：已人工查看的输入复核帧。

后续验证焦点：事件等价比较实际参与候选保留、反馈零扰动对照、目标物体及动作时段相容性。当前证据支持继续检验假设，尚不支持直接批量生成第一阶段监督真值。完整主动潜状态搜索与第二阶段轨迹效用均保持未验证。
