# 最新方法草案对齐：Motion-Primitive Causal Seq2Seq Transformer

- 日期：2026-06-30
- 完成内容：读取并梳理用户提供的最新方法草案，围绕motion primitive tokenizer、causal encoder、prediction query decoder、双流human-camera预测、transition phase、boundary与预训练目标形成阶段性理解，并整理待确认问题。
- 本地产出：
- C:\Users\Andcher\.codex\attachments\2560f97e-6cba-4e20-b772-578ef6992cf8\pasted-text.txt
- reports/progress_log.md
- 关键理解/结果：草案主线是从手机IMU patch中学习motion primitive token，使用causal Transformer编码历史上下文，再用prediction queries预测当前到未来的human/camera状态序列；主要待确认点包括token粒度、VQ codebook训练方式、causal与parallel decoder命名关系、phase/boundary标签来源、以及P2LHAP式smoothing是否写入最终推理。
- 下一步：与用户对齐方法定义后，按example.pdf内容页模板重新组织发言稿和PPT。
