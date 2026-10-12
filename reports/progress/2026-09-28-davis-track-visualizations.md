# DAVIS 四视频 TrackCraft3R 轨迹可视化

- 日期：2026-09-28
- 范围：`car-turn`、`drift-turn`、`boat`、`soccerball`

## 完成工作

- 只使用已有的 22 帧完整教师 query 轨迹、7 帧 observed dense TrackCraft3R、query 映射、相机元数据、RGB 帧和逐帧 mask；未重新运行 DA3、TrackCraft3R 或 MotionForesight。
- 每段生成：22 帧教师轨迹叠加视频、7 帧 observed-only 轨迹叠加视频、t0 源帧 20 的教师/观测 query 对比图。
- 视频保持 DAVIS 原始帧尺寸 `854x480`。编码 `8 fps` 只用于播放，未作为真实采集帧率或实验时间协议。
- t0 query 使用稳定编号 `0--47`；对比图用红色 `T` 表示完整教师、蓝色 `O` 表示 observed-only，黄色连线表示对应 query 的投影差异。
- 轨迹直接投影到原始 RGB 帧；没有生成临时逐帧文件，渲染结束后目录中只保留 MP4、PNG、manifest 和 README。

## 结果

| 视频 | 可视化 | 原质量闸门 | 教师视频可见点数 |
|---|---|---|---|
| `car-turn` | COMPLETE | FAIL | 22 帧均 48 |
| `drift-turn` | COMPLETE_PARTIAL_FAIL | FAIL | 后期因投影出画面逐步减少，最低 9 |
| `boat` | COMPLETE | FAIL | 22 帧均 48 |
| `soccerball` | COMPLETE | FAIL | 22 帧均 48 |

`drift-turn` 的点数下降表示教师三维轨迹投影超出原始图像范围，不表示轨迹文件缺失；完整数组仍为 `(22,48,3)`。该段按要求标记为 `COMPLETE_PARTIAL_FAIL`，保留可生成部分。四段质量闸门仍为 `0/4 PASS`，可视化不改变训练决定。

## 输出

- 根目录：[visualizations](../../experiences_results/surface-relation-davis-joint-decoder-probe-2026-09-28/visualizations/)
- 索引：[visualization_index.json](../../experiences_results/surface-relation-davis-joint-decoder-probe-2026-09-28/visualizations/visualization_index.json)
- 每段目录包含独立 `README.md`、`visualization_manifest.json` 和 `sha256_manifest.json`。
- 本地输出与远端投影包的来源哈希已记录；远端磁盘仍约剩余系统盘 `2.3G`、数据盘 `3.2G`，无模型进程运行。

## 下一步

保持真实视频 Relation Head 训练暂停。当前可视化用于核查 t0 对齐、投影出界和跨上下文差异，不构成监督质量放行依据。
