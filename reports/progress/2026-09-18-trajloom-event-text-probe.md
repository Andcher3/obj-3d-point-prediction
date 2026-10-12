# TrajLoom 事件文本条件复测

- 日期：2026-09-18
- 完成工作：在两个 RoboTAP 视频上完成正确/相反事件文本、三个配对随机种子的 12 次推理；复用无文本结果，按目标区域分析并生成可视化；一次独立同种子复跑逐字节一致。
- 本地输出：`archive/experiments/trajloom/TrajLoom-probe/analysis/event_text_probe_metrics.json`、`event_text_visuals/`、`trajloom_event_text_probe_bundle.tgz`。
- 关键结果：三个种子下，两个视频均为正确事件 ADE 低于无文本、相反事件 ADE 高于无文本；文本效应约为随机种子效应的 0.33 倍，能稳定调整轨迹，但没有主导宏观事件。
- 结论：TrajLoom 可作为文本条件随机二维轨迹基线或结构参考，当前不选为开放世界三维轨迹第二阶段主干。
- 报告：`reports/analysis/2026-09-18-trajloom-event-text-probe/report.md`。
- 下一步：第二阶段主干仍需围绕指定物体三维轨迹接口选择；语义条件实验需使用更大样本和预测时生成的 narration。
