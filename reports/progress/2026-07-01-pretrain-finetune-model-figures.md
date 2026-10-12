# 补充生成预训练与微调阶段模型结构图

- 日期：2026-07-01
- 完成内容：额外生成两张与当前PPT风格一致的模型结构图，分别展示无标签预训练阶段和有标签微调阶段的结构；将两张图统一为1600×900画布，便于插入PPT。
- 本地产出：
- reports/weekly/2026-07-01-motion-primitive-forecast-deck/model_figures/pretraining_structure.png
- reports/weekly/2026-07-01-motion-primitive-forecast-deck/model_figures/finetuning_structure.png
- reports/weekly/2026-07-01-motion-primitive-forecast-deck/model_figures/model_structure_contact_sheet.png
- 关键理解/结果：预训练图突出无标签IMU到future motion token对齐；微调图突出当前状态从g_t预测、未来状态经forecast decoder和causal re-encoder后预测。
- 下一步：可将两张结构图作为现有PPT方法部分的替换图或附录图使用。
