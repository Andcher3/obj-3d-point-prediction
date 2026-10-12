# soccerball 轨迹可视化

- 可视化状态：`COMPLETE`；轨迹投影状态：`COMPLETE`；原质量闸门：`FAIL`。
- 本轮只读取既有轨迹和输入文件，未重新运行 DA3、TrackCraft3R 或 MotionForesight。
- 教师轨迹：`full22_query_tracks.npz`，完整源帧 `14--35`。
- 观测轨迹：`observed7_trackcraft_dense.npz` 按 `query_frame0_mapping.json` 采样，源帧 `14--20`。
- t0：源帧 `20`，dense 索引 `6`；query 稳定编号为 `0--47`。
- 预览视频编码 `8 fps` 仅用于播放；DAVIS 原始采集帧率未知，未使用 MP4 预览帧率作为实验时间。

## 文件
- [完整 22 帧教师轨迹视频](teacher_full22_trackcraft_query_overlay.mp4)
- [7 帧 observed-only 轨迹视频](observed7_trackcraft_query_overlay.mp4)
- [t0 教师/观测点对比图](t0_teacher_observed_query_comparison.png)
- [可视化 manifest](visualization_manifest.json)
- [SHA-256 清单](sha256_manifest.json)

t0 对比图中红色 `T` 为完整教师轨迹，蓝色 `O` 为 observed-only 轨迹，黄色连线表示同一 query 的投影差异。
