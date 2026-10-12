# 跨点信息实验：下载完成核查

日期：2026-10-11，00:07（北京时间）

## 已完成

- 检查远端`assets-complete.json`、固定revision来源清单、微调包记录、数据准备记录和进程状态；全部列出的下载文件均已取得，无缺失资产。
- 资产完成记录于10月10日21:51:55生成。第一轮aria2退出码为-6，恢复器随后第二轮退出码为0并完成校验；当前协调器和下载器均已退出，属于任务完成。
- TrackCraft3r完整大小17,761,542,554字节（16.542GiB），实际SHA256与固定版本发布者SHA256一致；DA3模型大小/SHA256也通过，均无aria2控制文件。
- 从固定Wan revision的HF镜像API补查LFS元数据，与完成记录中的实际哈希和大小比对：DiT、VAE、spiece.model、tokenizer.json均通过。两份小tokenizer JSON的发布者LFS SHA256不可用，大小通过；不能将自算哈希称为发布者哈希。
- MotionForesight源码ZIP CRC、微调包ZIP CRC及此前本地/远端一致性检查均通过；这些包未提供发布者SHA256，保留该限制。
- MOVi-F的100训练/20验证/40测试场景GT已完成，原MD5、GT和划分检查通过记录保持；数据盘剩12.23GiB。

## 输出

- `experiences_results/cross-point-probe-preparation-2026-10-10/assets-complete-2026-10-11.json`
- `experiences_results/cross-point-probe-preparation-2026-10-10/asset-manifest-2026-10-11.json`
- 更新原准备报告及`reports/progress_log.md`，按research-progress-logger保留完成证据。

## 状态与待完成项

本次仅核查和传回小型记录；没有下载大模型到本地、加载模型、forward或训练。GPU保持关闭。

下载完成不等于全部无卡准备完成：DA3源码/兼容依赖、MotionForesight首帧query和12fps时间接口的CPU核查、完整query预览交付仍待完成。`preparation-complete.json`中的仅观测模型缓存尚未生成，尚不通知用户开启GPU。
