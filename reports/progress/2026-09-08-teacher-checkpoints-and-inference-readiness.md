# 四篇未来文本工作的权重与推理代码核查

日期：2026-09-08

## 已完成

核对 VANS、FUTURE-L1、NEP、Video-CoE 的作者资源，重点区分论文训练后权重、通用底座、推理入口与本地运行验证。未下载模型或执行第三方模型代码。

## 核查结果

| 工作 | 论文训练后权重 | 公开代码 | 接入判断 |
|---|---|---|---|
| VANS | 已确认，含 MLLM 与视频侧 safetensors 文件 | 推理 Demo、模型加载、文本生成和视频生成代码 | 可从作者权重开始接入；需配置环境、路径和设备，文本独立接口需要整理 |
| FUTURE-L1 | 已确认 SFT 与 RL-DAPO_v133k2k 两个仓库的完整模型分片 | 训练、评测及潜表示感知的推理实现 | 可从作者权重开始接入；需使用项目自定义解码并适配自有视频 |
| NEP | 本轮仍未找到作者发布的论文训练后检查点 | 训练示例、数据及评测相关材料 | 当前不列入现成权重教师；获得检查点或进行训练复现后才能按该方法接入 |
| Video-CoE | 本轮未核实到公开入口 | 论文仍注明代码和模型将发布；未核实到对应作者仓库 | 当前不列入可直接接入的教师 |

“有权重与推理代码”表示材料可用，尚未表示本地环境运行通过。

## FUTURE-L1：重要更正

上一轮权重核查不完整。作者账号 Eurayka 已发布：

- [Future-L1-SFT 文件](https://huggingface.co/Eurayka/Future-L1-SFT/tree/main)：四个 safetensors 分片、索引、模型配置、tokenizer、视频预处理和聊天模板。模型卡明确关联本论文，标注 SwimBird_Qwen3VL。
- [Future-L1-RL-DAPO_v133k2k 文件](https://huggingface.co/Eurayka/Future-L1-RL-DAPO_v133k2k/tree/main)：同样有四个权重分片与推理配置，页面总量约 17.6 GB。该数字是文件大小，不是显存要求。
- 核查时 SFT 仓库可见提交 `94c8e7a80284ab19e2b6f455350be2fc711c3405`，RL 仓库可见提交 `07b1619bdf16ca29b833f344960f4af8b3cac6eb`。
- [作者代码](https://github.com/OpenGVLab/Future-L1)包含 [future_l1.py 推理适配](https://github.com/OpenGVLab/Future-L1/blob/main/lmms-eval/lmms_eval/models/future_l1.py)、[FutureBench 脚本](https://github.com/OpenGVLab/Future-L1/blob/main/lmms-eval/examples/eval_futurebench_future_l1.sh)、[TwiFF 脚本](https://github.com/OpenGVLab/Future-L1/blob/main/lmms-eval/examples/eval_twiffbench_future_l1.sh)。需要项目自定义的前向与潜表示解码，不能仅据普通 Qwen 加载成功认定方法接入完成。
- 脚本默认多进程与 FlashAttention 配置属于评测设置，不能据此断言最小硬件必须为八张卡。未进行硬件测试。
- RL 仓库名称包含 DAPO；公开页面未说明它与论文最佳 LA-DAPO 指标的一一对应关系，本轮不将该权重标为已验证的最佳检查点。

## VANS：文本路径与默认 Demo 的区别

- [权重页面](https://huggingface.co/KlingTeam/VANS/tree/main)可见 `VANS_mllm.safetensors`（37.3 MB）和 `VANS_vlm.safetensors`（2.84 GB）。完整 Demo 还依赖 Qwen2.5-VL-3B-Instruct 与 Wan2.1-T2V-1.3B 底座。
- [官方仓库](https://github.com/KlingAIResearch/VANS)提供 `app.py`。检查发现 GPU 编号和权重路径需要按环境修改；README 安装示例也需要正常整理，未认定为零修改启动。
- [agent_mllm.py](https://github.com/KlingAIResearch/VANS/blob/main/vans/models_mllm/agent_mllm.py)有 `generate_with_query`，输入视频与文字后先得到文本；[pipeline](https://github.com/KlingAIResearch/VANS/blob/main/vans/pipelines/wan_video_new.py)随后才进入视频生成。
- 根据调用关系，第一阶段只取未来文本有独立封装的依据，预计可省去运行 Wan 视频生成部分。该判断来自静态代码；独立加载、依赖裁剪和批量接口尚未实现或测试。

## 其他作者入口

- [NEP 仓库](https://github.com/sail-sg/Video-Next-Event-Prediction)、[关联 V1 仓库](https://github.com/haonan3/V1)、[作者模型列表](https://huggingface.co/haonan3)：本轮未发现对应训练后模型，通用 Qwen 权重不计入该方法检查点。
- [Video-CoE 论文](https://arxiv.org/abs/2603.14935)：目前公开说明仍为代码与模型将发布；排除了名称类似但论文不同的 CoE 仓库。

## 对工作量的影响

以可获取权重和推理实现为条件，当前保留 VANS 与 FUTURE-L1，两者都可免去为获得首个教师而完整复现训练的工作。VANS 的交互 Demo 较现成，独立文本路径需整理；FUTURE-L1 已有文本预测评测路径，需接入自有视频并保留自定义解码。实际安装及推理耗时尚无测量，不据此给出确定工期。

## 文件更新与后续待决

- 已修订 [同日输出质量报告](D:/Pycharm/post_graduate_project/reports/ideas/2026-09-08-stage1-teacher-output-quality-audit.md)中的 FUTURE-L1 权重状态。
- 已更新 [研究进展索引](D:/Pycharm/post_graduate_project/reports/progress_log.md)。
- 未决：两种已发布模型在本地的实际加载、自由多分支输出质量及 RL 检查点指标对应关系；教师仍未选定。
