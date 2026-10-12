# 制作 Future Patch Label Sequence 方法海报、PPT 与发言稿

- 日期：2026-06-30
- 完成内容：使用 imagegen 生成 Future Patch Label Sequence 方法的模型结构图、训练策略图和切换推理图，并基于这些图制作学术海报、周三组会 PPTX 和 5 分钟发言稿。
- 本地产出：
  - `reports/posters/assets/future-label-sequence-transformer-architecture.png`
  - `reports/posters/assets/future-label-sequence-training-pipeline.png`
  - `reports/posters/assets/future-label-sequence-switch-inference.png`
  - `reports/posters/2026-06-30-future-label-sequence-method-poster.html`
  - `reports/posters/2026-06-30-future-label-sequence-method-poster.png`
  - `reports/posters/2026-06-30-future-label-sequence-method-poster.pdf`
  - `reports/weekly/2026-07-01-future-label-sequence/2026-07-01-future-label-sequence-method.pptx`
  - `reports/weekly/2026-07-01-future-label-sequence/2026-07-01-future-label-sequence-speech.md`
  - `reports/weekly/2026-07-01-future-label-sequence/2026-07-01-future-label-sequence-imagegen-prompts.md`
  - `scripts/create_future_label_sequence_materials.py`
- 关键理解/结果：组会材料主线确定为“用 future patch label sequence 统一当前分类、未来预测、延续概率、切换时机和下一动作”，并明确 parallel query Transformer decoder 是当前主方案；自回归 decoder 和 transition-type head 作为后续消融或扩展。
- 下一步：组会前试讲一次，确认 5 分钟内是否需要压缩相关工作依据，组会后根据导师反馈决定先找连续 HAR 数据集还是先实现 UCI HAR 上的 encoder sanity check。

