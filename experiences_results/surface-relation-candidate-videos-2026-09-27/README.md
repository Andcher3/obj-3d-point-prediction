# 新视频候选：DAVIS 2017 TrainVal 480p

来源：DAVIS 2017 官方下载页面 `https://davischallenge.org/davis2017/code.html` 的 `DAVIS-2017-trainval-480p.zip`。下载 ZIP 832,766,765 字节，通过 7-Zip 完整性测试；本地 SHA-256 为 `e3d0b5b77c3d031b000a19e0e25e3e2cac65d183755601bc2cf066df1a2aa492`。仅提取下面八条独立序列的 JPG 与逐帧 2D 实例掩码，重编码为 MP4 以便人工查看。MP4 的 **24 fps 只是预览播放速率**，原始 JPG 序列无统一可据此确认的原视频时间戳；研究时需另定时间采样协议。

下面 `t0` 为原始帧编号 20；面积和中心位移从官方掩码计算，中心位移是帧 20→36 的像素位移，只是筛选线索，并非 3D 运动指标。掩码实例 ID 在该窗口内连续有效，网球拍除外。

| 序列 | 官方划分 | 帧数 | 建议研究对象（掩码 ID） | t0 像素面积 | 帧 20–36 目标中心位移 | 候选用途 |
| --- | --- | ---: | --- | ---: | ---: | --- |
| `bear` | train | 82 | 熊（1） | 45,536 | 16.7 px | 大目标、短期平移较小 |
| `bmx-bumps` | train | 90 | 自行车（1） | 6,327 | 381.8 px | 大幅运动；后续面积低至 770 px，遮挡/出框风险 |
| `breakdance` | val | 84 | 跳舞者（1） | 41,920 | 97.5 px | 复杂非刚性/肢体运动 |
| `dog` | val | 60 | 狗（1） | 27,165 | 99.0 px | 动物关节运动 |
| `soapbox` | val | 99 | 赛车小车（1） | 8,685 | 20.3 px | 与车上人员分离的目标；需复核掩码归属 |
| `stroller` | train | 91 | 婴儿车及乘客（2） | 15,301 | 68.6 px | 移动物体与人交互，实例定义含乘客 |
| `swing` | train | 60 | 秋千座椅/绳（2） | 3,077 | 32.8 px | 细结构诊断；座椅/绳实例边界需人工复核 |
| `tennis` | train | 70 | 网球拍（2） | 1,250 | 120.0 px | 薄/小目标困难样本；后续部分帧不可见 |

八个视频均在 `videos/davis2017-<序列>-review24fps.mp4`；对应输入 JPG 和原始调色板 PNG 掩码在 `selected-frames/DAVIS/{JPEGImages,Annotations}/480p/<序列>/`。已用 ffprobe 核对八个 MP4 的可解码帧数与原始 JPG 数一致，均为 854×480。

逐段打开：[熊](D:/Pycharm/post_graduate_project/experiences_results/surface-relation-candidate-videos-2026-09-27/videos/davis2017-bear-review24fps.mp4) · [BMX 自行车](D:/Pycharm/post_graduate_project/experiences_results/surface-relation-candidate-videos-2026-09-27/videos/davis2017-bmx-bumps-review24fps.mp4) · [街舞](D:/Pycharm/post_graduate_project/experiences_results/surface-relation-candidate-videos-2026-09-27/videos/davis2017-breakdance-review24fps.mp4) · [狗](D:/Pycharm/post_graduate_project/experiences_results/surface-relation-candidate-videos-2026-09-27/videos/davis2017-dog-review24fps.mp4) · [赛车小车](D:/Pycharm/post_graduate_project/experiences_results/surface-relation-candidate-videos-2026-09-27/videos/davis2017-soapbox-review24fps.mp4) · [婴儿车](D:/Pycharm/post_graduate_project/experiences_results/surface-relation-candidate-videos-2026-09-27/videos/davis2017-stroller-review24fps.mp4) · [秋千](D:/Pycharm/post_graduate_project/experiences_results/surface-relation-candidate-videos-2026-09-27/videos/davis2017-swing-review24fps.mp4) · [网球拍](D:/Pycharm/post_graduate_project/experiences_results/surface-relation-candidate-videos-2026-09-27/videos/davis2017-tennis-review24fps.mp4)。

**边界：** DAVIS 只提供 RGB 和二维实例掩码，没有可直接用于本项目的未来 3D 点轨迹真值，也尚未生成 MotionForesight 输入、query、DA3 或伪标签。它们是新增的人工审查/受控输入候选，不应计入上一次四视频实验，也不能据此验证三维预测收益。DAVIS 官方 train/val 标签仅为源数据集划分；这些被人工查看过的序列不应作为新的盲测数据。

补充来源：PointOdyssey v1.2 官方站 `https://pointodyssey.com/` 提供 RGB、实例分割、相机和 3D 轨迹，但此前本地 sample 的两个序列已在 F2 使用；其他 test 包在官方 HuggingFace 页面约 26.5 GB，超出当前 D 盘约 19 GB 的可用空间。TAPVid-3D 官方仓库提供逐片段 3D 轨迹示例，但原视频受 Waymo/Aria/Panoptic 各自许可约束，未擅自下载纳入。Ego4D 官方文档要求有效的许可凭据，现有下载端点匿名访问返回 403；目前只有已取回的六段旧片段，尚无新 Ego4D 视频。

## 2026-09-28：仅保留物体目标的新候选

从上述已校验的 DAVIS 官方压缩包追加提取了七段独立序列，保留原始 JPG 和调色板实例掩码，并编码本地 MP4 供查看。掩码 ID 均指单个指定物体；模型火车只取蓝色车头（ID 4），不把全列不同实例合并成一个物体。逐帧检查原始帧 14–35（若 t0=20，则可容纳 7 帧观测、15 帧未来）时，这些目标掩码都非空；MP4 已用 ffprobe 核对可解码帧数。

| 序列 | 明确目标及掩码 ID | 原始帧数 | 帧 14–35 掩码最小面积 | 筛选备注 |
| --- | --- | ---: | ---: | --- |
| `car-turn` | 银色汽车，逻辑实例 1（调色板 PNG 索引 1；OpenCV BGR `[0,0,128]`） | 80 | 12,477 px | 静机位，目标逐渐靠近、转向 |
| `car-roundabout` | 前景灰色汽车，逻辑实例 1（调色板 PNG 索引 1） | 75 | 63,584 px | 大面积目标，背景有其他车但另行排除 |
| `drift-turn` | 漂移赛车，逻辑实例 1（调色板 PNG 索引 1） | 64 | 14,863 px | 转向和尺度变化明显 |
| `bus` | 公交车，逻辑实例 1（调色板 PNG 索引 1） | 80 | 79,319 px | 大面积目标，部分遮挡 |
| `boat` | 船体，逻辑实例 1（调色板 PNG 索引 1） | 75 | 23,179 px | 目标平移，水面可能干扰三维重建 |
| `soccerball` | 足球，逻辑实例 1（调色板 PNG 索引 1） | 48 | 2,547 px | 有柱子遮挡，较小目标，列为困难候选 |
| `train` | 蓝色火车车头，逻辑实例 4（调色板 PNG 索引 4） | 80 | 16,282 px | 模型铁路场景，只取车头，不合并车厢 |

本地视频：`videos/davis2017-{car-turn,car-roundabout,drift-turn,bus,boat,soccerball,train}-review24fps.mp4`；图像/掩码：`selected-frames/DAVIS/{JPEGImages,Annotations}/480p/<序列>/`。24 fps 仅为预览编码时的播放速度，不代表原视频真实拍摄帧率或模型采样协议。

任务范围仅限指定**物体**的三维点轨迹。上节历史候选中的 `bear`、`dog`、`breakdance` 等人/动物目标，以及包含乘客的 `stroller`，均排除当前物体实验；`swing`、`tennis` 暂不列为优先候选，目标过细或可见性存在风险。新增七段只有 RGB + 二维实例掩码，**尚无同一 query 的可靠未来三维轨迹标签**；它们是视频候选，不是已可训练或已可评价的三维数据。
