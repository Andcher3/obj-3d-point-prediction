# 制作 P2LHAP 学术海报

- 日期：2026-06-28
- 完成内容：基于 P2LHAP 原文、用户保存的外部总结和前一轮勘误，制作一版高信息密度学术海报，重点记录论文动机、数据管线、Patch-to-Label Seq2Seq 模型结构、channel-independent Transformer、训练目标、smoothing strategy、forecast 输出形式、实验结果、消融实验和对方向一的迁移价值。
- 本地产出：
  - `reports/literature/p2lhap_2403.08214.pdf`
  - `reports/literature/p2lhap_2403.08214.txt`
  - `reports/posters/2026-06-28-p2lhap-academic-poster.html`
  - `reports/posters/2026-06-28-p2lhap-academic-poster.pdf`
  - `reports/posters/2026-06-28-p2lhap-academic-poster.png`
- 关键理解/结果：P2LHAP 的核心价值是用 patch-level activity label sequence 统一 recognition、segmentation 和 forecast；当前状态延续和切换趋势可以从未来 patch 标签序列中派生。海报中保留了输入形状、默认参数、损失函数、主要实验表格、消融实验和局限，避免把论文结论泛化到 UCI HAR 或手机运镜双标签场景。
- 验证情况：使用 Chrome 导出 PDF，使用 Poppler 转出 PNG 预览，并通过浏览器 DOM 检查确认主要区块无文本溢出。
- 下一步：围绕 P2LHAP 的 patch-to-label 输出形式，进一步讨论方向一是否应采用双标签序列预测：`human_action_sequence + phone_motion_sequence`。
