# 核查既有视频并寻找新样本

- 日期：2026-09-27
- 完成：从 AutoDL `/root/autodl-tmp/ego4d-formal-e2/clips/` 取回 6 段 Ego4D 视频，其中 4 段对应本次联合解码实验；本地与远端 SHA-256 一致。为 4 段生成 t0 周围约 16 帧的便捷预览；预览经重采样，不作为模型精确输入。分别在 `experiences_results/surface-relation-joint-decoder-video-review-2026-09-27/clips/`、`experiment-windows/`，说明见该目录 `README.md`。
- 新样本：从 DAVIS 2017 官方 TrainVal 480p 下载并完整性验证 ZIP，提取 8 个独立序列及其逐帧 2D 掩码，编码 8 个可播放 MP4；帧数、解码和目标掩码面积已检查。文件和候选用途见 `experiences_results/surface-relation-candidate-videos-2026-09-27/README.md`。
- 结果与限制：八段序列覆盖短期小位移、大幅平移、关节运动以及薄目标/遮挡；视频之间独立于既有四段 Ego4D。DAVIS 无原生三维轨迹，尚未建立同坐标未来监督，不能直接成为本项目三维模型评价集。Ego4D 新视频需有效授权凭据，匿名对象请求返回 403；PointOdyssey 其余带三维真值的包大于本地剩余空间。TAPVid-3D 视频仍受原来源许可约束。
- 下一步：先由研究者看四段原片与 t0 窗口，确认动作、目标一致性、遮挡；DAVIS 候选可人工淘汰目标过细/出框或实例定义不清的片段。正式三维实验需独立、可靠的三维监督，不能把这批二维视频直接加入上次统计。
