# PointOdyssey 冻结关系接口实验：数据与坐标预检

日期：2026-09-26

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Verification Status: PARTIAL（数据预检完成；五组训练与接口裁决未执行）
- 数据：PointOdyssey v1.2 官方 HuggingFace `sample.tar.gz`；两个独立序列。

## 已完成

- 下载官方样本；按原始 30 fps 每两帧取一帧，构造 15 fps、7 帧观测加 15 帧未来的窗口。
- 审计官方 2D/3D 轨迹、相机投影、mask 与逐帧三维有效性；t0 query 可见，未来遮挡点只要有有效三维真值仍纳入评价，未来可见率另报。
- 先用单个 mask ID 作接口烟测；再依据 t0 RGB 与实例 mask 将同一人物、机器人各自的多个局部 mask 合成物体级目标。物体级 query 均仅由 t0 信息确定，未用未来轨迹或模型误差筛选。
- 两个序列都完成 RGB-only DA3、观测期 TrackCraft3R 和冻结 MotionForesight 推理。模型未接收未来 RGB、真值深度或真值相机。均输出 `P0=(15,48,3)`、`H=(15,48,16)`，时间槽位对应有效。
- 修复本次独立预检副本中“原图与 DA3 处理尺寸不同”时内参被错误缩放的问题；没有覆盖旧实验结果。

## 结果

| 目标 | t0 官方 GT 投影 p90 | 未来 GT 投影 p90 | 每步最少有效 3D GT | 每步最少可见 | GT→DA3 t0 四折留出残差 p90 / bbox |
|---|---:|---:|---:|---:|---:|
| 机器人整体（`r4_new_f`） | 0.26 px | 0.23 px | 48/48 | 30/48 | 34.1% |
| 右侧人物整体（`scene_d78_0318_3rd`） | 0.27 px | 0.34 px | 48/48 | 29/48 | 21.0% |

两个物体级目标的官方真值本身通过投影与有效性审计；它们与 RGB-only 观测几何之间的 t0 登记未通过预定的 10% 门槛。作为定位诊断，单个局部 mask 的 TrackCraft t0 登记 p90 分别为 38.1%、75.7%；直接 DA3 t0 为 16.8%、13.3%。两视频 TrackCraft query-anchor 最大误差分别为 1.24、2.21 个 dense 像素。DA3 与 TrackCraft 在少数点上的几何分歧明显，单用稳定对应关系不能保证整物体登记通过。对所有可选的单 mask 局部片段做事后诊断，直接 DA3 登记合格数分别为 3/16、1/5；这些事后片段不能替代事先定义的物体级目标。

## 边界与阻塞

- 本轮只完成数据/接口预检。B/F/N/U/V 五组没有训练，不能判定冻结关系头有效或无效。
- 官方完整 PointOdyssey 数据约 185 GB；当前远端数据盘约剩 3.7–3.8 GB，本地 D 盘约剩 24 GB。正式不少于 30 个独立测试视频的协议尚无可行的完整数据驻留方案。
- 已上传物体级新 query/GT 清单，但准备继续远端复跑时 SSH 网关连续返回 `Connection closed`；两个物体级目标的 MotionForesight 新 query 缓存尚未生成。原单 mask 接口缓存均已下载到本地。
- 单 mask 对应物体局部零件；不能用它代替“同一目标物体的多点”主试验。物体级 grouping 目前仅人工核对两例，正式数据集需要统一、独立于结果的目标定义。

## 输出

- 协议：[冻结接口裁决实验](2026-09-26-surface-relation-interface-decision-experiment.md)
- 审计：[物体级审计 JSON](../../experiences_results/pointodyssey-interface-preflight-2026-09-26/object_group_audit.json)、[机器人 DA3 登记](../../experiences_results/pointodyssey-interface-preflight-2026-09-26/object_r4/da3_t0_registration.json)、[人物 DA3 登记](../../experiences_results/pointodyssey-interface-preflight-2026-09-26/object_scene/da3_t0_registration.json)
- 单 mask 接口缓存：[机器人](../../experiences_results/pointodyssey-interface-preflight-2026-09-26/export_r4/)、[人物](../../experiences_results/pointodyssey-interface-preflight-2026-09-26/export_scene_d78/)
- 实现：`scripts/pointodyssey_interface_preflight.py`、`scripts/pointodyssey_export_interface_pilot.py`、`scripts/pointodyssey_label_registration_audit.py`、`scripts/pointodyssey_object_da3_audit.py`、`scripts/pointodyssey_candidate_geometry_audit.py`；物体标注 `configs/pointodyssey_pilot_object_targets_2026-09-26.json`。

## 下一步

恢复远端访问后，先复核物体级 query 的完整 P0/H 与观测期几何，再寻找不使用测试真值作为预测输入的可靠标签坐标接口。只有通过预先锁定的数据质量和独立视频数量门槛，才启动五组主实验；不能把当前预检失败写成方法设计失败。
