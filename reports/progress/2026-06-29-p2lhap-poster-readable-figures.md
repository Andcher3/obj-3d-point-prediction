# 放大 P2LHAP 海报中的 imagegen 图表

- 日期：2026-06-29
- 完成内容：根据用户反馈，重新调整 P2LHAP 学术海报版式，显著放大三张 imagegen 图表，解决 HTML 中偏小、PDF 中难以看清的问题。
- 本地产出：
  - `reports/posters/2026-06-28-p2lhap-academic-poster.html`
  - `reports/posters/2026-06-28-p2lhap-academic-poster.pdf`
  - `reports/posters/2026-06-28-p2lhap-academic-poster.png`
- 关键理解/结果：海报宽度从 2600px 进一步扩大到 3500px；模型数据流图和标签序列推理图显示尺寸提升到约 1434 x 718，实验概览图提升到约 1136 x 569。图像仍引用原 imagegen 生成 PNG，没有使用代码构建图。
- 验证情况：使用 Playwright 检查三张图片加载成功且所有 section 无溢出；重新导出 PDF，并用 Poppler 渲染最终 PNG 后完成目检。
- 下一步：后续论文海报中，imagegen 图表应优先预留足够版面，避免为了塞入更多文字而牺牲图像可读性。
