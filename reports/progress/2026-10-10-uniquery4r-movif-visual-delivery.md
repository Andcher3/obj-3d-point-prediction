# MOVi-F 实验1：18段轨迹视频回传与校验

日期：2026-10-10

## 完成工作

- 用户关闭GPU并重启服务器后恢复视频回传，未加载模型、启动forward或训练。
- 本地断点8,454,144字节与远端前缀SHA256一致，接着传回7,017,894字节；完整包15,472,038字节。
- 完整包SHA256为`4ccea85a05caf8e88c74c1d556cc542fc20058e32bf93fbabcc4d55cfef36d61`，校验通过；23个归档成员安全解压，包内文件哈希全部通过。
- 18段H.264对照视频逐段完整解码22帧，尺寸1536×560。生成本地播放页、每段t0截帧和0/6/13/21帧总览，抽查两组图的三列布局与标记。

## 查看方式与证据

- 播放页：`experiences_results/uniquery4r-movif-experiment1-2026-10-09/visual-review.html`。
- 每段从左至右：模型直接二维跟踪、模型三维点通过估计相机回投影、仿真真值。相同颜色对应相同query；真值空心点表示近似可见性检查判为不可见。
- query取第6帧；完整22帧为离线教师跟踪输入，不能称为未来预测。原生12fps，8fps慢放。
- 传输/解码校验：同目录`visual-packaging-checks.json`、`logs/visual-transfer-manifest.json`。
- 数值原件：`logs/experiment1_gt_result_22frames.json`；实验分析见`2026-10-09-uniquery4r-movif-experiment1-results.md`。
- 原始NPZ与无损原图仍在远端，本地紧凑包只包含视频、评分与记录。此区别已保留于逐样本检查字段；不伪称原始数组在本地校验通过。
- 本地t0图与分帧图来自H.264解码截帧，不能替代数值轨迹。本轮未重跑模型、重新估计焦距、调整轨迹或改变评分结果。

## 代码与状态

- `scripts/uniquery4r_prepare_connection.py`增加可选流式断点续传；调用前单独核对前缀哈希。
- `scripts/extract_verified_uniquery4r_review.py`先检查完整包SHA256，再限制为目录内、不重复、非链接的普通文件；已有冲突文件拒绝覆盖。
- `scripts/package_uniquery4r_movif_experiment1.py`验证文件哈希与全部视频解码，生成播放页。
- 本轮交付已完成，待用户观看。GPU保持关闭；没有关系模块训练结论。

技能影响：research-progress-logger记录传输、校验和本地/远端原件区别，保留周会可复查证据。
