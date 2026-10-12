# 第一阶段未来文本教师：输出质量与新方法核查

日期：2026-09-08

> 同日资源复核更正：作者 Hugging Face 已公开 FUTURE-L1 的 [SFT](https://huggingface.co/Eurayka/Future-L1-SFT/tree/main) 与 [RL-DAPO_v133k2k](https://huggingface.co/Eurayka/Future-L1-RL-DAPO_v133k2k/tree/main) 权重，本轮通过浏览器确认了实际分片、配置与 tokenizer 文件。上一轮“未找到训练后权重入口”的核查不完整，现予更正。详细入口和接入工作量见 [资源核查记录](D:/Pycharm/post_graduate_project/reports/progress/2026-09-08-teacher-checkpoints-and-inference-readiness.md)。未进行本地模型推理，公开 RL 检查点与论文最佳分数的对应关系尚未确认。

## 结论与证据范围

VisualCOMET、KM-BART、DIVE 保留为基线或多样性方法参考；上一轮确认的多条文本生成能力，尚不足以确立它们作为高质量教师。FUTURE-L1 保留为有针对性的新增候选，VANS 的公开资源较便于实际验证。NEP 与 Video-CoE 提供相关训练思路，教师资格继续待验证。

本轮检查了论文正文、评测协议、公开生成图、作者仓库与模型文件列表。未下载权重、未进行本地推理。论文选例的质量、整体实验结果和实际部署效果分别记录。额外未来描述属于候选伪标签；当前未选定教师，也未修改两阶段方法。

## 1. 三篇新工作的共同评测边界

[NEP 的 FutureBench 定义](https://arxiv.org/html/2505.22457#S3)包含已知的最终事件或目标，并用选择题评价中间事件推理。Video-CoE 和 FUTURE-L1 均报告该基准成绩。它检验的是受未来结尾约束的预测，直接移除结尾后会改变输入条件。分数不能直接解释为自由多分支伪标签的正确率。

这里的关键差别是条件来源：历史叙述可由观测前缀得到；已经发生在未来的结尾，通常不属于预测时已知信息。训练标签使用真实后续内容是正常监督，不能与预测时提供未来结尾混为一谈。

## 2. 较早方法的实际输出

### KM-BART 与 VisualCOMET

[KM-BART 表 2、表 6](https://aclanthology.org/anthology-files/pdf/acl/2021.acl-long.44.pdf)展示：在图像和“某人拿着信封”的当前事件条件下，AFTER 候选包括打开信封、交给另一人、拥抱另一人。它确实产生不同事件；其中拥抱对指定信封后续运动的描述较弱。移除当前事件文字会明显改变候选内容。

其有当前事件文字条件的 AFTER 人工比较胜率为 52.0%，对手是 VisualCOMET；这项指标衡量相对偏好，未给出候选绝对合格率。作者选例和比较结果支持将两者作为基线，尚未支持直接大量生成目标物体模式标签。

### DIVE

[DIVE](https://arxiv.org/html/2408.08021v1)通过处理泛泛标注和对比学习改善生成。人工比较支持其合理性、多样性优势，但使用成对胜率；多样性标准同时涉及语义和措辞。附录办公室样例仍出现作者标为不合理的“发号施令”等 AFTER 推断。

本轮另查看了猫的样例，其中错误属于 INTENT 推断，不能当作 AFTER 专项错误率。该方法可作为减少泛泛候选的参考，尚缺与当前指定物体任务一致的分支可靠性证据。

## 3. 用户新增的三篇方法

### Fostering Video Reasoning via Next-Event Prediction（NEP）

- 方法将视频前缀与真实后续的文本描述配对，构建 V1-33K，并尝试预测、批评和推理蒸馏训练。训练形式对本项目有参考价值。[论文](https://arxiv.org/html/2505.22457)
- 论文图 2 的篮球多路径图是推理结构示意，不能当作训练后模型成功输出多分支的定量证据。
- 公开数据中的推理和 critique 主要围绕记录中的未来；它们尚未为同一前缀的其他合理未来提供独立有效性标签。[数据](https://huggingface.co/datasets/haonan3/V1-33K)
- 判断：保留数据构造与训练参考地位；目前未证明可直接提供高可靠性多分支监督。

### Video-CoE

- 主要机制是构造观测视频的历史事件链，再预测后续事件；历史链中的多个节点不等于多个替代未来。[论文 §4](https://arxiv.org/html/2603.14935#S4)
- 自由文本测试删除选择题选项，保留问题；表 4 的 CoE-SFT 38.13%、CoE-GRPO 32.42% 是五个模型的相对胜出比例。该实验没有测量每组替代未来的有效数量。[表 4 与附录 F](https://arxiv.org/html/2603.14935)
- 直接查看公开 HTML 版图 7，发现体育报道画面与水烟准备、驾车的回答和评语存在对应疑点。可能涉及图例拼接；本轮不将该图计为可靠成功例，也不由单图推断整体模型质量。[原图](https://arxiv.org/html/2603.14935v1/judgement_full.png)
- 判断：视觉依据与事件推理值得参考，自由多分支教师能力仍待验证。

### FUTURE-L1

- 方法在文字推理中插入连续视觉潜表示。TwiFF-Bench 的开放问答结果更接近当前需求：最终答案为 2.97/5，3.04 为答案与推理评分的平均值。它们是模型评判分数，不能换算为伪标签合格比例。[论文表 2、附录 C](https://arxiv.org/html/2606.05769)
- 两项报告评测均采用确定性解码。文中的 temporal diversity 约束同一次推理内相邻潜表示，尚未验证不同输出之间的未来模式多样性。
- 查看了装配铲雪辅助装置的成功例和棒球犬失败例。二者均属于选择题；成功例的动作步骤已出现在选项中。失败例偏离记录中的后续，单凭这一点也不能断言它属于物理不合理的替代未来。
- 判断：保留为较有针对性的新增候选，开放回答证据较直接；训练后权重已在同日后续复核中确认，自由分支质量仍待验证。

## 4. VANS 的补充核查

[VANS](https://arxiv.org/html/2511.16669v2)的公开文字案例包含向鸡肉撒奶酪等具体后续动作，具有人—物交互粒度。但展示不同未来的图例改变了情境或假设，例如被烫与吃辣；它验证条件控制能力，未证明同一观测与问题下的自主分支覆盖。部分问题带程序性目标，使用时需要与数据条件对应。

本轮核实 [Hugging Face 文件列表](https://huggingface.co/KlingTeam/VANS/tree/main)确有 `VANS_mllm.safetensors` 与 `VANS_vlm.safetensors`。这只确认资源存在；加载、依赖和生成效果未经本地测试。

## 5. 公开资源状态

截至本轮查看页面的状态；“未找到权重入口”不等同于断言作者从未发布。

| 方法 | 本轮核实到的资源 | 当前判断 |
|---|---|---|
| NEP | [作者代码](https://github.com/sail-sg/Video-Next-Event-Prediction)、V1-33K 数据；未找到训练后权重链接 | 有训练复现材料，现成教师入口未确认 |
| Video-CoE | 论文注明代码、模型将发布；本轮未核实到作者发布入口 | 保留方法参考与待定候选 |
| FUTURE-L1 | [代码](https://github.com/OpenGVLab/Future-L1)、[50K 数据](https://huggingface.co/datasets/Eurayka/Future-L1-50K)、作者发布的 SFT 与 RL-DAPO_v133k2k 完整模型分片已核实 | 可从已训练权重接入推理，需使用自定义潜表示解码实现；本地运行待验证 |
| VANS | [代码](https://github.com/KlingTeam/VANS)、实际权重文件可见 | 具备进一步加载验证的资源条件 |

FUTURE-L1 数据页面包含打开盒子、摆放药盒、磨刀等训练文本，也包含游戏及镜头相关内容。这些是数据目标，不能作为训练后模型输出效果的证据。

## 6. 对当前研究判断的修订

当前需要验证的是：固定观测和目标物体后，额外候选是否同时具备情境相容性、事件差异、物体指向性及适当时间跨度。教师不必恢复世界的完整概率；其给出的额外正标签需要有足够可靠的依据。

三篇新工作提高视频事件预测的相关能力，但公开结果主要关注匹配一次记录未来。尚未验证它们会稳定保留合理低频分支。围绕单一答案的训练可能影响分支覆盖，这是待测假设，不能直接视为已经证实的缺陷。

因此，本轮保留 FUTURE-L1 和 VANS 的候选身份，下调旧模型作为主教师的优先级；NEP、Video-CoE、DIVE 的训练机制继续作为方法参考。教师与模式图监督来源均未选定。

## 7. 实际查看的主要原始图例

- [NEP 图 2](https://arxiv.org/html/2505.22457v1/tree.png)：方法动机示意，不计为生成模型的多分支结果。
- [DIVE 附录比较图](https://arxiv.org/html/2408.08021v1/qual_app.png)：同时展示 VisualCOMET、KM-BART、DIVE 推断。
- [DIVE 猫与沙发样例](https://arxiv.org/html/2408.08021v1/ea2.png)：包含意图推断，避免误归类为未来事件结果。
- [Video-CoE 自由回答与评判](https://arxiv.org/html/2603.14935v1/judgement_full.png)：记录图文对应疑点。
- [Video-CoE 失败例](https://arxiv.org/html/2603.14935v1/bad_case_0.png)：论文涉及时间标注失败；相关发生比例不能当作整体未来预测错误率。
- [FUTURE-L1 成功例](https://arxiv.org/html/2606.05769v1/case2.png)、[失败例](https://arxiv.org/html/2606.05769v1/bad_case.png)：均为选择题。
- [VANS 主要结果](https://arxiv.org/html/2511.16669v2/8_mainvis.png)、[改变假设的结果](https://arxiv.org/html/2511.16669v2/7_generalization_vis_world.png)：分别用于查看动作粒度和条件变化。

本报告补充并限定 [2026-09-07 调研](D:/Pycharm/post_graduate_project/reports/ideas/2026-09-07-multibranch-future-teachers-and-trajectory-supervision.md)，不删除其中的历史证据及第二阶段讨论。
