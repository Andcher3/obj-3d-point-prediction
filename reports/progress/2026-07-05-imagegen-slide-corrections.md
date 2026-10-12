# 修正 Motion-Primitive 周会 PPT 页面

- 日期：2026-07-05
- 完成内容：基于现有周会 PDF 页面导出图，使用 imagegen 对第 2、4、5、6、7、8 页进行局部修正。
- 本地产出：
  - `reports/weekly/2026-07-05-slide-image-edits/source_pdf_pages/`
  - `reports/weekly/2026-07-05-slide-image-edits/imagegen_corrected_pages/page-02-corrected-imagegen.png`
  - `reports/weekly/2026-07-05-slide-image-edits/imagegen_corrected_pages/page-04-corrected-imagegen.png`
  - `reports/weekly/2026-07-05-slide-image-edits/imagegen_corrected_pages/page-05-corrected-imagegen.png`
  - `reports/weekly/2026-07-05-slide-image-edits/imagegen_corrected_pages/page-06-corrected-imagegen.png`
  - `reports/weekly/2026-07-05-slide-image-edits/imagegen_corrected_pages/page-07-corrected-imagegen.png`
  - `reports/weekly/2026-07-05-slide-image-edits/imagegen_corrected_pages/page-08-corrected-imagegen.png`
- 关键理解/结果：统一了 `t:t+n` 输出范围、future token 的 channel 维度、`O(C·L^2)` 复杂度、`1-cos` 预训练损失、phase head 与 smoothing 的职责边界，以及“第一个新 label 出现位置”作为切换时机的定义。
- 下一步：将修正版页面替换回 PPT/PDF，并在试讲时检查公式可读性和时间是否符合 5 分钟汇报要求。
