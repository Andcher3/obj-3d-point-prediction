# VANS 与 FUTURE-L1 多未来文本及潜状态接口讨论

日期：2026-09-08

## 已完成

核对两种模型的作者推理代码及 FUTURE-L1 方法部分，区分文本随机采样、完整推演的分支变化、潜视觉状态导出和未来轨迹条件接口。此次完成静态核查与方法讨论；未下载权重、执行模型推理或验证多分支生成质量。

## 实现证据

- VANS 的 `generate_with_query` 已使用随机文本解码，默认 `do_sample=True`、`temperature=0.7`、`top_p=0.9`。同一输入可通过独立调用获得多个文本候选；原实现没有保证候选具有不同事件含义。只改变后续 Wan 视频生成的随机种子，不等于改变上游文本事件。
- FUTURE-L1 评测默认使用确定性解码，但继承的生成参数构造函数在温度大于零时启用采样；自定义 `_future_l1_sample` 明确具有 `torch.multinomial` 分支。因此存在对完整交错推演进行随机解码的实现依据。
- FUTURE-L1 的连续潜状态通过上一隐藏状态及可选投影头递推，离散 token 负责文本及潜状态区间的控制。它没有在每个潜状态上直接采样高斯噪声。相同前缀及固定控制路径下，重复潜状态递推不自动产生不同未来。
- FUTURE-L1 交替生成文本和一个或多个潜视觉区间，再给出最终回答。训练时，潜状态以真实视频中较后帧的 Qwen3-VL 视觉嵌入为对齐目标。一次推演内的多个区间属于推演过程，不能直接认定为多个互斥未来。
- 评测封装已有 `FUTURE_L1_EXPORT_LATENTS` 与导出目录设置，生成结果读取 `latent_states` 和 `latent_mask`。导出接口提供提取依据，尚未运行验证。
- 论文的 temporal-diversity reward 约束同一次推演中不同潜状态区间的重复，并未直接验证开放未来模式的覆盖能力。

## 当前候选思路（未选定）

1. VANS：从同一观测独立采样完整文本回答，再按事件含义归并；重复占主导时，可试验在不改变已知观测事实的前提下，要求提出与已有候选不同的合理后续。该互补提问会改变候选提议分布，不能把回答频次当作真实未来频率。
2. FUTURE-L1：采样整段文本—潜视觉交错推演，保留每次的 `(未来文本 T_i, 潜状态序列 H_i)`。仅在固定的同一 H 上重新生成最终文字，可能主要增加措辞变化。不同 H 是否真的表达不同事件仍需核查。
3. 教师离线多次采样与学生预测时输出显式模式图是两种接口；离线采样不要求最终系统重复采样教师。教师输出的额外分支仍是候选伪标签。
4. 用户提出直接利用 H 的想法有训练机制依据。H 可能保留最终文字省略的交互关系与空间配置；这些信息是否帮助目标物体三维轨迹属于待验证假设。
5. 第二阶段可以比较以文本 T、潜状态 H 或二者共同作为条件的效果。直接使用 H 不要求先把 H 转成自由文本，但原模型的内部文本推演仍然存在。

## 仍待判断

2026-09-08 后续澄清：用户强调以原有 narration 方案为比较基准，判断 H 是否更有利于第二阶段；信息充分性和模式可分性应对两种表示使用同样标准。保留无文本条件等消融只用于分析机制，不将“包含文字以外的信息”作为采用 H 的唯一门槛。当前主要问题转为怎样在教师生成过程中主动鼓励事件差异；详见同日 `2026-09-08-event-diverse-teacher-generation-recap.md`。

- H 中包含多少目标物体未来信息，多少属于场景外观和推理过程？原训练没有直接规定三维坐标、目标点身份或与轨迹时间步的一一对应。
- H 的相似度是否反映事件差异？连续向量及视觉对齐均不自动提供 MotionMap 所需的模式空间。
- 只保留 H 是否丢失原模型通过交错文本与 KV 上下文使用的信息？保留文本—潜状态配对可用于这一判断。
- 额外 H 仍是预测假设，不等于额外真实未来轨迹标签。采用真实未来编码特征训练时，需要处理它与历史条件下预测 H 的差异。
- FUTURE-L1 是离线教师还是预测时第一阶段？离线设置需要学生学习对应表示；在线设置则保留完整教师推演成本。这一用途尚未由用户确认。

## 证据链接

- [VANS 文本生成](https://github.com/KlingAIResearch/VANS/blob/main/vans/models_mllm/agent_mllm.py)
- [VANS 视频生成调用顺序](https://github.com/KlingAIResearch/VANS/blob/main/vans/pipelines/wan_video_new.py)
- [FUTURE-L1 论文方法](https://arxiv.org/html/2606.05769#S3)
- [FUTURE-L1 自定义生成](https://github.com/OpenGVLab/Future-L1/blob/main/src/model/future_l1.py)
- [FUTURE-L1 混合前向及投影](https://github.com/OpenGVLab/Future-L1/blob/main/src/train/monkey_patch_forward.py)
- [继承的 Qwen3-VL 生成参数](https://github.com/OpenGVLab/Future-L1/blob/main/lmms-eval/lmms_eval/models/qwen3_vl.py)
- [FUTURE-L1 推理封装及潜状态导出](https://github.com/OpenGVLab/Future-L1/blob/main/lmms-eval/lmms_eval/models/future_l1.py)

## 输出及后续讨论

- 本记录：`D:\Pycharm\post_graduate_project\reports\progress\2026-09-08-multinarration-and-latent-future-interface.md`
- 进展索引：`D:\Pycharm\post_graduate_project\reports\progress_log.md`
- 下一讨论点：确认教师的离线或在线角色，再决定文本与潜状态分别承担监督、模式表示和第二阶段条件中的哪些职责。此处未确定模型、采样数量、阈值或训练损失。
