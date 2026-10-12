# SensorLM 第 04 页损失函数说明修订

- 日期：2026-06-24
- 完成内容：基于用户手动调整后的第 04 页截图，仅重绘底部说明区，将原来的简略参数说明替换为对齐损失 InfoNCE、生成损失 next-token cross entropy 和总目标的展开说明。
- 本地产出：
- reports/weekly/native_sensorlm_slides_v4/sensorlm_paper_slide_04.png
- 关键理解/结果：第 04 页现在能够直接说明 L_con 如何在 batch 内拉近匹配 sensor-caption pair、推远负样本，以及 L_cap 如何在 sensor 条件下逐 token 预测 caption。
- 下一步：将该页替换进完整汇报幻灯片，并与发言稿中模型结构部分保持一致。
