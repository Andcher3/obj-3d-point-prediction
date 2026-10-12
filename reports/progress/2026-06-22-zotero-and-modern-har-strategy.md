# 记录 Zotero 本地阅读习惯并调整 HAR 现代方法主线

- 日期：2026-06-22
- 完成内容：确认用户已将第一篇 HAR 综述保存到 Zotero 本地文件夹，更新 AGENTS.md 和 AGENTS_zh.md，使后续论文阅读优先读取本地 PDF；同时根据近年 HAR 文献趋势，整理 DBN/DRL 的学习优先级和现代模型落地路线。
- 本地产出：
- AGENTS.md
- AGENTS_zh.md
- reports/literature/2026-06-22-har-modern-method-positioning.md
- reports/progress_log.md
- reports/progress/2026-06-22-zotero-and-modern-har-strategy.md
- 关键理解/结果：DBN 和 DRL 当前不应作为主线；更合理的落地路线是以 masked AutoEncoder / self-supervised IMU representation learning 为主，保留极薄监督 baseline 作为 sanity check，并为后续 Transformer/Attention、多任务预测和 OIS 事件预警打基础。
- 下一步：围绕 masked autoencoder/self-supervised HAR 选读一到两篇更近论文，并设计 UCI HAR 上的第一版自监督预训练实验。
