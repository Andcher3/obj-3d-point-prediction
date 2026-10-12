# 方法构想再对齐：单通道 motion primitive token 与 smoothing 推理

- 日期：2026-07-01
- 完成内容：读取用户对方法疑问的统一回复，并将当前方法收敛为单通道 motion primitive tokenizer、causal encoder、parallel prediction query decoder、human-camera双流预测、transition phase监督与P2LHAP式smoothing推理的版本。
- 本地产出：
- C:\Users\Andcher\.codex\attachments\a6f885be-0468-4d42-83e9-bd89fdea2864\pasted-text.txt
- reports/progress_log.md
- 关键理解/结果：当前版本去掉boundary head和uncertain类别，camera motion暂按单标签主导动作处理；切换时机由平滑后的human/camera未来标签序列第一次变化位置读取。主要疑虑集中在transition phase人工标注成本、VQ tokenizer训练细节、单通道token数量与实时性、以及camera单标签假设的适用边界。
- 下一步：按example.pdf内容页风格重组PPT：先讲阅读文献，再讲任务定义与背景，最后讲方法结构与训练推理流程。
