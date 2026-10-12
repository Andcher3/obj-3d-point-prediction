# MolmoMotion 静止轨迹原因诊断

日期：2026-09-21

## 已完成

- 将此前剪刀和浇水壶的完整TrackCraft输入迁移到当前GPU服务器并校验哈希。
- 使用当前模型、源码和推理脚本复跑六条旧指令。
- 修正饮料瓶采样率，从7.5 fps重新生成15 fps TrackCraft输入。
- 测试原目标放置指令、短动作句、动作后移窗口及旧样本成功提示词。
- 比较五组输入的历史像素运动、三维运动、深度和未来伪轨迹。

## 输出

- `reports/analysis/2026-09-21-molmomotion-static-trajectory-diagnosis/report.md`
- `experiences_results/MolmoMotion-static-diagnosis-2026-09-21/`
- `scripts/analyze_molmomotion_input_motion.py`
- `scripts/run_trackcraft_release_inference.py`
- `scripts/prepare_molmomotion_shifted_bottle_input.py`

## 结果

旧输入在当前环境中产生明显非静止、随文本变化的轨迹，排除模型权重和通用推理实现故障。饮料瓶在修正采样率、缩短提示词、移动观测窗口及替换为旧成功提示词后仍近静止。原因主要是公开AR检查点在当前视觉/三维输入上的零运动退化；H3历史运动弱、DA3+TrackCraft与官方管线的分布差异及开放世界场景共同构成风险。

## 后续

若继续追查单一组件，需要获得官方ViPE+AllTracker预处理或官方完整样本，在同一视频上做输入管线互换；当前证据不足以把退化唯一归因于Tracker、相机坐标或模型本身。
