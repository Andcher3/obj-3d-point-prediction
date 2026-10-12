# 22 帧官方协议获取尝试复核与路线决策

- 日期：2026-09-24
- 状态：完整输入链通过；完整 TrackCraft 标签未获得；当前路线停止继续远程资源调参

## 已确认

- 已从保留的 15 fps RGB 序列生成完整 `T=22` 输入。
- 时间切片为观测 `0:7`、未来 `7:22`，元数据和输入字段审计通过。
- TrackCraft3R 在 V100 上两次未生成完整 dense track map：一次明确 CUDA OOM，另一次在降低空间分辨率后进程退出。
- 没有伪造、拼接或用 16 帧结果填充 22 帧标签。

## 解释边界

`future_frames_supplied_to_model=false` 只证明未来 RGB 没有作为 MotionForesight 前向输入切片传入。若 DA3 是在完整 22 帧上联合推理，再从中取前 7 帧 depth/pose，则仍需单独标记“深度估计上下文包含未来帧”；这不能直接等同于 observed-only 预测条件。

因此，当前 22 帧输入通过不代表：

1. 已经得到官方训练标签；
2. 已经得到完整 TrackCraft 伪 GT；
3. 已经证明 `P0/H` 与未来标签处于无泄漏的同一坐标链。

## 决策

1. 将完整 `7+15` 输入生成记录为流程复现成功，将 TrackCraft dense 获取记录为资源阻塞。
2. 停止继续在当前 V100 上做无边界的分辨率、显存或进程调参；其研究收益不足以改变监督来源结论。
3. 保持 `relation_loss_ready=false`，不启动真实 Relation Head、decoded-coordinate relation-loss pilot、IRLS、Huber 或 learned gate。
4. 若未来获得更大显存或作者标签，必须先补做 DA3 observed-only 与 full-window 前 7 帧的坐标/深度差异审计，并显式记录 future-context leakage，再进行训练标签验证。
5. 当前研究转向方法和接口层：保留 E0/E1/E3-controlled 结果，edge head 与 node residual 不冻结，等待可靠监督来源或另立替代监督实验。
