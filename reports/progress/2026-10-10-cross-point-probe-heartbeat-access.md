# 跨点信息实验定时检查：会话访问受限

日期：2026-10-10，15:37（Asia/Shanghai）。

- 已执行准备任务定时检查；`write_stdin`访问原SSH传输session 52083被权限策略拒绝：`approval required by policy, but AskForApproval is set to Never`。
- 本轮未能读取远端日志、下载进程或磁盘状态，无法给出新进度或ETA；不能据此判定服务器、SSH网络或下载进程失败。
- 本地准备报告仍可读取。上轮已完成数据审核、微调包远端SHA256/ZIP CRC；本轮未修改这些结果。
- 未尝试绕过权限限制；未开启GPU、加载模型、forward或训练。
- 后续需恢复会话访问才能确认底座下载与剩余CPU准备状态。

## 15:45重新核查

- 用户要求重新检查后，原session 52083返回`Unknown process id`；重新建立SSH会话5431，服务器指纹与此前一致。
- 远端准备进程1966和aria2c进程1968均正常运行，下载日志持续增长。前次检查失败未导致远端下载中断。
- 总下载约1.2MiB/s。TrackCraft3r约1.5GiB/16.54GiB（9%），684KiB/s，ETA 6小时22分47秒；Wan DiT约0.9GiB/5.2GiB（17%），596KiB/s，ETA 2小时7分6秒。仅为当前速度估计，非完成承诺。
- 数据盘剩余33GB；微调包校验记录与160场景GT准备状态仍通过，全部模型资产完成记录尚未生成。未开启GPU、加载模型或训练。

采用research-progress-logger记录本次核查结果；主体准备记录仍为`2026-10-10-cross-point-probe-cpu-preparation.md`。
