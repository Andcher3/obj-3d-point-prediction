# DAVIS数据在本项目中的用途与时间

- 日期：2026-10-09
- 完成：查阅2026-09-27、09-28、09-29及10-09实验记录，确认数据来源和用途。无模型运行或实验修改。

## 数据角色

本项目使用DAVIS 2017 TrainVal 480p的RGB视频和逐帧目标实例掩码。掩码用于在目标物体内部选择query，以及检查预测点投影后是否仍在目标范围。该包不提供本项目所需的可靠3D点轨迹真值，因此三维教师轨迹由模型估计。

## 时间线

- 09-27：用户要求查阅实验原视频并寻找更多样本，下载、校验DAVIS官方包，整理候选。
- 09-28：按明确物体目标筛选后，使用car-turn、drift-turn、boat、soccerball进行DA3/TrackCraft3R教师生成及MotionForesight接口检查，期望建立关系解码器的监督数据。教师质量未通过，四视频上的关系解码器训练未启动；已有旧Ego4D关系解码器训练属于另一批数据。
- 09-29：这四视频用于Route A/B教师构造与质量诊断，仍未获准训练。后续排查query位置、三维登记及投影链。
- 10-09：复用四段物体视频测试UniQuery4R，在用户批准query后生成22帧与7帧跟踪视频，并诊断drift-turn偏移。

## 来源记录

- `reports/progress/2026-09-27-surface-relation-video-review-and-candidate-search.md`
- `reports/progress/2026-09-28-object-video-candidates-and-training-evidence.md`
- `reports/progress/2026-09-28-davis-joint-decoder-four-video-quality-gate.md`
- `reports/progress/2026-09-29-davis-ab-teacher-experiment.md`
- `reports/progress/2026-10-09-uniquery4r-gpu-visual-inference.md`

记录技能：research-progress-logger。这里只整理实际用途，不将DAVIS二维标注或教师模型输出当作3D真值。
