# 修订周三组会 PPT v2 的图标与信息图表达

- 日期：2026-06-30
- 完成内容：根据“矢量图必须能独立辅助理解”的规范，重做周三组会 PPT v2 的第 1 页和第 9 页，移除泛用装饰图标，改为论文方法缩略图、label sequence、encoder-decoder、小型表格和训练管线等信息图表达。
- 本地产出：
  - `reports/weekly/2026-07-01-future-label-sequence-v2/2026-07-01-weekly-research-and-method-v2.pptx`
  - `reports/weekly/2026-07-01-future-label-sequence-v2/contact_sheet.png`
  - `reports/weekly/2026-07-01-future-label-sequence-v2/slides/slide_01.png`
  - `reports/weekly/2026-07-01-future-label-sequence-v2/slides/slide_09.png`
  - `reports/weekly/2026-07-01-future-label-sequence-v2/rebuild_full_image_deck.mjs`
- 关键理解/结果：新版 PPT 仍保持每页整页图片插入；检查确认 PPTX 共 9 页，每页 1 张图片，图片边界均为 `[0,0,1600,900]`，没有留白。第 1 页和第 9 页的视觉元素从泛用图标改为能直接表达数据流、输出形式、模型结构和实验路线的图形。
- 下一步：组会前按新版 contact sheet 试讲，重点压缩第 2-4 页论文方法介绍和第 6-8 页方法设想之间的衔接。
