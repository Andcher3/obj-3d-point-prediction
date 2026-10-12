# MolmoMotion 原始视频文本遵从复测

日期：2026-09-21

## 已完成

- 将官方 MolmoMotion `model.pt`、TrackCraft3R 代码与权重、Depth Anything 3、UMT5 tokenizer 迁移到 GPU 服务器并恢复运行环境。
- 校验 MolmoMotion `model.pt` SHA256：`506beccd01e9edd4d3ebf0bf88fec00b83530eec525571168187c7c4888ee205`。
- 校验 TrackCraft3R `model.safetensors` 两端 SHA256：`d70084c009a9c894654685c1744217a3f86558a2189d3e896d29251ec177a7ad`。
- 对 `test_video.mp4` 完成 DA3 相机/深度估计和 TrackCraft3R 稠密三维轨迹处理，得到 12 帧、832×480 的 `track_map`，并在饮料瓶上选取 8 个查询点。
- 按 MolmoMotion H3/F30 协议，用前三个历史时刻预测 30 个未来时刻；未来帧仅用于离线 TrackCraft 伪真值，不进入模型条件。
- 对“放在鼠标旁”“放在黑色水杯旁”“放在电脑旁”“继续拿着”四条指令完成 greedy 推理，四次均成功解析为 `(8,30,3)` 有限轨迹。
- 生成四条单独视频、四宫格对比视频和最终帧图，并完成本地解码检查。

## 结果

- 四条指令的输入 token 哈希不同，说明文本输入实际进入了模型；sequence score 也不同。
- 四次生成文本哈希完全相同，四个轨迹数组哈希均为 `7ab5b06c88ec5d353b76e544bf46aefef35106168e04ff40e417dc5dec4752cd`。
- 任意两条指令的轨迹差异为 0；每条轨迹在 30 个预测时刻内也完全不变。
- 指标中的约 `0.000625 m` 路径长度来自最后历史点到量化后的首个预测点，不能解释为预测期运动。
- 可视化中 8 个预测点停留在饮料瓶当前位置，没有朝鼠标、黑色水杯或电脑移动。

当前公开 AR checkpoint 在这条视频与 DA3+TrackCraft3R 零样本输入链路上未表现出文本遵从。该结果不外推到官网展示所用的 Flow-Matching 版本，也不评价整个 MolmoMotion 方法。

后续诊断发现，本轮原始TrackCraft输入实际按原视频每4帧采样一次，即7.5 fps；结果元数据曾误记为15 fps。重新生成15 fps输入后结论未改变。详见 `reports/analysis/2026-09-21-molmomotion-static-trajectory-diagnosis/report.md`。

## 本地输出

- `experiences_results/MolmoMotion-raw-video-trackcraft-2026-09-21/aggregate_metrics.json`
- `experiences_results/MolmoMotion-raw-video-trackcraft-2026-09-21/drink_bottle_ambiguous_future/`
- `experiences_results/MolmoMotion-raw-video-trackcraft-2026-09-21/visualizations/comparison_2x2.mp4`
- `experiences_results/MolmoMotion-raw-video-trackcraft-2026-09-21/visualizations/comparison_2x2_final.png`
- `scripts/prepare_molmomotion_trackcraft_input.py`
- `scripts/render_molmomotion_trackcraft_comparison.py`

## 后续实验边界

如继续测试，应把公开 AR checkpoint 与官网 Flow-Matching 模型分开记录；随机采样对照需让不同文本共享同一随机种子，避免把采样噪声误判为文本效应。
