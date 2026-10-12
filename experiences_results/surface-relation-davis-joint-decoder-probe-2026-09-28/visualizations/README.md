# DAVIS 四视频 TrackCraft3R 轨迹可视化

**已作废：本目录使用了错误的 query 时间对应，不得用于身份、质量或训练判断。**
源帧 14 直接 mask query 的修正版位于
`experiences_results/surface-relation-davis-frame0-direct-2026-09-28/visualizations/`。

本目录包含四段 DAVIS 物体视频的既有 TrackCraft3R query 轨迹可视化。所有视频保持原始 `854x480` 帧尺寸；编码 `8 fps` 仅用于播放，不代表真实采集帧率。

时间协议：源帧 `14--35`，observed-only 为 `14--20`，`t0=20`，教师评价未来为 `21--29`。每段均使用 48 个稳定 query 编号。

每段目录包含：

- 22 帧完整教师 TrackCraft3R 轨迹视频；
- 7 帧 observed-only TrackCraft3R 轨迹视频；
- t0 原始 RGB 帧上的教师/观测 query 对比图；
- 轨迹来源说明、质量闸门状态和 SHA-256 清单。

当前四段可视化均生成完成；原标签质量闸门结果仍为 `0/4 PASS`，可视化不改变训练闸门。

`drift-turn` 的教师轨迹后期有部分 query 投影出原始图像边界，目录状态标为 `COMPLETE_PARTIAL_FAIL`，视频保留可生成部分；其余三段投影完整。

