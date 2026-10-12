# PointOdyssey 完整 22 帧 DA3 预处理资源阻塞

日期：2026-09-26

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Verification Status: RED_FLAG（资源失败；未产生可用标签）

## 执行内容

- 恢复 AutoDL 双 V100 服务器连接。
- 复用已审计的 `r4_new_f` 与 `scene_d78_0318_3rd` 两个 PointOdyssey 物体级目标。
- 从本地官方样本中提取每个目标的 22 帧、15 fps RGB 输入并上传到远端。
- 按 `7+15` 协议启动 `r4_new_f` 的 DA3 22 帧预处理，`process_res=518`。

## 结果

- DA3 在第一目标的 DINO 注意力阶段显存不足，错误为尝试额外申请约 `13.10 GiB`。
- 单张 V100 总显存约 `31.73 GiB`，失败时约 `31.18 GiB` 已占用。
- 进程退出前未生成 `full22_user_res518.npz`；没有拼接、伪造或使用不完整结果。
- 第二目标尚未启动，避免重复触发同一资源失败。

## 解释边界

- 这是完整 22 帧 DA3 输入在当前单卡显存路径上的资源阻塞。
- 它不构成 PointOdyssey 数据质量结论，也不构成冻结关系接口失败证据。
- 双 V100 的显存不能由当前 DA3 进程自动合并；需要降低处理分辨率、改变注意力实现或使用单卡可容纳的输入。

## 当前状态

- 已完成的 7 帧 PointOdyssey `P0/H` 接口缓存仍然有效。
- 完整 `7+15` TrackCraft query-only 标签尚未生成。
- B/F/N/U/V 主实验未启动。
- 按实验执行规范，资源失败后暂停自动重试；低分辨率 `process_res=392` 只能作为明确标注的诊断性重跑。

## 输出

- 远端日志：`/root/autodl-tmp/projects/motionforesight/outputs/pointodyssey_full22_da3/r4_new_f/prepare.log`
- 远端输入：`/root/autodl-tmp/projects/motionforesight/outputs/pointodyssey_full22_inputs/`
