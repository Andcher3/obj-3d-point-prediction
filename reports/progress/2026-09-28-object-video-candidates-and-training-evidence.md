# 物体视频候选与关系解码器训练证据核对

- 日期：2026-09-28
- 完成：从本地已校验的 DAVIS 2017 TrainVal 官方包追加提取 `car-turn`、`car-roundabout`、`drift-turn`、`bus`、`boat`、`soccerball`、`train` 七段独立物体序列及逐帧实例掩码，编码 MP4 并核验帧数。帧 14–35 的指定物体掩码均非空；训练目标必须只取物体实例，排除旧清单中的人/动物目标。
- 输出：`experiences_results/surface-relation-candidate-videos-2026-09-27/videos/davis2017-*-review24fps.mp4`，同目录 `selected-frames/` 与更新后的 `README.md`。本轮仅整理候选视频，未启动三维模型训练。
- 证据核对：`scripts/surface_relation_joint_decoder_probe.py` 与 `configs/surface_relation_joint_decoder_probe_2026-09-27.json` 表明此前关系解码器确实训练：MotionForesight 冻结；四段旧 Ego4D 整视频留一，每折另外三段以 7 个未来槽位的 TrackCraft3R 位移伪标签训练解码器，留出 2 槽验证；五种子、每种 400 步；最后在未参与该折训练的整段视频测试。关系版并未通过预设的跨视频收益信号（0/4）。只说明该具体探索实现没有稳定收益，不代表拟议的后续联合训练方法已测试。
- 认识：新增 DAVIS 视频提供 RGB 和二维目标掩码，尚无同一 query 的可信未来三维轨迹监督，不能直接增加三维训练样本数。已有训练用的完整视频 TrackCraft3R 伪轨迹允许离线生成标签，但其误差和目标点同一性仍需要独立审计。
- 后续三维监督来源线索：PointOdyssey 官方数据包含 RGB、实例掩码及三维轨迹，但必须筛选目标为物体而非人物/动物的序列；Kubric MOVi 官方数据含实例掩码、逐帧物体三维位姿和物体坐标，适合构造合成物体点轨迹受控对照；HOT3D-Clips 含真实手持物体视频、物体模型与逐帧姿态，可进一步评估真实物体监督的可用性。三者均尚未在本轮新增下载或完成与 MotionForesight 输入坐标的适配。来源：`https://pointodyssey.com/`、`https://github.com/google-research/kubric/blob/main/challenges/movi/README.md`、`https://github.com/facebookresearch/hot3d/blob/main/hot3d/clips/README.md`。
- 下一步：在物体目标范围内人工复核视频运动和遮挡；补齐可靠三维标签并按原始视频隔离训练、验证和测试后，才能用新视频检验新增结构的有效性。
