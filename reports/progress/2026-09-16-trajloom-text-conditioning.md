# TrajLoom 文本条件矩阵实验

- 日期：2026-09-16
- 完成工作：修正 GT 形状与时间窗口对齐；在服务器分析 12 次 TrajLoom 文本条件推理；下载指标、日志和全部预测产物。
- 本地输出：`archive/experiments/trajloom/TrajLoom-probe/analysis/text_matrix_metrics.json`、`analysis/text_matrix_stdout.txt`、`text_matrix.log`、`outputs/text_matrix/`。
- 关键结果：两个视频共 12/12 推理成功。固定 seed=42 改变文本会改变轨迹；一致文本 seed=42/43/44 也产生不同轨迹。文本效应在两个视频上幅度不一致，尚不能证明事件级多模态、可靠指令遵循或概率校准。
- 报告：`reports/analysis/2026-09-16-trajloom-text-conditioning/report.md`。
- 下一步：增加视频和条件样本，采用集合覆盖/校准指标，检查 tokenizer 警告并继续保持单次样本不等于概率的解释边界。
