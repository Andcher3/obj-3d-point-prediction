# UniQuery4R 部署可行性与资源核查

- 日期：2026-10-09
- 完成：核查官方项目、论文、GitHub 推理实现、Hugging Face 权重与显存实测。
- 技能：academic-research-suite 用于区分官方实测和本项目资源估计；research-progress-logger 用于本地记录。
- 状态：仅完成部署前信息检查。未连接关闭的服务器，未下载大权重、安装环境或执行推理；无其他代理参与。

## 可部署性

官方推理代码及 `Kosmo-Research/UniQuery4R` 权重已公开。模型卡说明约 1.65B 参数，权重文件大小约 6.6 GB。输入 RGB 序列，输出二维对应位置、三维点、场景流及相机参数，支持直接输入稀疏 query。测试公开权重无需重新训练。

官方 WorldTrack 四数据集宏平均较强，不能据此确认本项目目标视频的教师质量，也未找到该表对 TrackCraft3R 的直接对照。完整视频输入仅用于离线轨迹恢复和教师评估，不进入未来预测条件。

## 资源依据

官方 GitHub README 在单卡 A800-80GB、PyTorch 2.5.1 上实测，504 px、20 帧稠密推理：

| 查询分块大小 | 峰值显存 |
| --- | --- |
| 5000 | 14.2 GB |
| 10000 | 16.4 GB |
| 30000，默认 | 25.7 GB |

本项目建议起始配置：单卡 RTX 4090 24 GB，64 GB 系统内存，8–16 个 CPU 核，数据盘预留 100 GB；对 22 帧、长边 504、48-query 测试预计可行，尚未在该卡实测。48 GB 单卡可提供更大余量。系统内存 32 GB 为较紧凑的备选；100 GB 磁盘是环境、缓存、权重与输出的预留建议，不是官方最低要求。

48-query 模式会减少解码工作，但完整视频编码及特征金字塔仍存在，不能按查询数比例估计全部显存。稠密可视化建议 `chunk_size=5000` 起步。对 V100 不宣称不支持；官方采用 FP16 和 PyTorch SDPA，其实际 attention 后端与峰值显存需要检查，不能直接套用 A800 的显存表。官方脚本为单设备推理，双卡显存不会自动合并。

环境：Linux、Python 3.10、PyTorch >=2.4；CUDA 构建按具体 GPU 与驱动选择。初次比较逐视频运行 UniQuery4R 与既有教师，不同时加载 MotionForesight 或其他大模型。

## 接口检查

1. `uniquery4r/models/uniquery4r.py::_build_pair_idx` 默认生成 `(0,t)` 与 `(i,i)`。论文支持任意源/目标帧，公开 `forward` 尚无直接 `source_frame_index` 参数；t0=6 查询需要轻量帧对适配及验证，不预设适配后精度不变。
2. README 示例将 query_uv 写为像素单位；`forward` 注释、`QueryBank` 及 `PairCrossAttnAggregator` 实际采用 `[0,1]` UV，再映射到 grid_sample 坐标。部署按实际处理图像的 `(W-1,H-1)` 归一化，保留原图到处理图的变换。
3. 三维输出参考序列第 0 帧，尺度在无真值时仅确定到全局比例。仍需检查与 MotionForesight 观测几何的接口，不直接视为米制真值。
4. 优先审计模型原始输出，分开保存任何 t0 后处理结果，避免通过人为偏移获得的小 anchor 误差冒充跟踪精度。

## 待执行测试范围

候选为现有 car-turn、drift-turn、boat、soccerball 四段物体视频，各 22 帧、相同 48 个目标 query、长边 504。先核对输入点和坐标转换，再输出二维轨迹覆盖视频及 t0 图，比较掩码内比例、人工确认的物理点身份、三维关系稳定性及 7/22 帧上下文差异。掩码命中只能说明点在物体内，不能独立证明物理点身份正确；关系稳定性检查不强制所有目标刚性。教师质量通过后才讨论关系模块训练。

以上为建议，未执行，也未作为已通过的实验协议。

## 原始来源

- 官方项目：https://kosmoresearch.github.io/UniQuery4R/
- 官方仓库与显存表：https://github.com/kosmoresearch/UniQuery4R
- 权重及文件大小：https://huggingface.co/Kosmo-Research/UniQuery4R/tree/main
- 模型与 query 接口：https://raw.githubusercontent.com/kosmoresearch/UniQuery4R/main/uniquery4r/models/uniquery4r.py
- UV 规范：https://raw.githubusercontent.com/kosmoresearch/UniQuery4R/main/uniquery4r/models/query_bank.py
- 预处理：https://raw.githubusercontent.com/kosmoresearch/UniQuery4R/main/uniquery4r/utils/load_fn.py
- 论文方法与尺度约定：https://arxiv.org/html/2608.17283v1

本地输出：本记录及 `reports/progress_log.md`。
