# 64帧以上三维轨迹真值样本获取核查

日期：2026-10-10

## Material Passport

- Origin Skill: academic-research-suite（发布源与实际文件核查）；research-progress-logger（记录）
- Origin Mode: dataset availability verification
- Verification Status: 来源说明及远程数组头已核验；完整视频、目标物体质量及重复帧尚未检查。

## 结论

可以按单个NPZ获取64帧以上的带三维轨迹真值样本。优先使用TrackCraft3R作者公开评测包中的长Kubric序列，保留物体任务；另有WorldTrack/ADT长序列可用于核对论文基准条件。当前MOVi-F的24帧限制不能代表所有Kubric生成数据的时长。

## 实际远程文件头核验

使用Hugging Face目录API读取文件大小；HTTP Range请求只读取ZIP目录及NPY头，不保存大型文件、不加载对象数组、不解码视频、不运行模型。所有部分内容请求均检查HTTP 206后才读取有限字节。

| 发布文件 | 大小（bytes） | 已读数组形状 |
| --- | ---: | --- |
| `kubric_da3/00001.npz` | 215953991 | `images_jpeg_bytes: (81,)` |
| `kubric_da3/00002.npz` | 119899165 | `images_jpeg_bytes: (81,)`; `world_coords: (399360,81,3)`; `extrinsics_w2c_gt: (81,4,4)` |
| `kubric_da3/00003.npz` | 225420176 | `images_jpeg_bytes: (81,)` |
| `adt_mini_da3/Apartment_release_clean_seq131_0.npz` | 294430597 | `images_jpeg_bytes: (300,)`; `tracks_XYZ: (300,825,3)`; `extrinsics_w2c_gt: (300,4,4)` |

三段Kubric压缩文件共561273332 bytes，约535.27 MiB，可独立下载，无需全量数据集。这里只验证三个文件及一个ADT文件，不宣称所有50个Kubric文件都具有81帧。

## 真值与预测字段区分

作者数据说明和读取代码一致：

- Kubric `world_coords` 为世界坐标下的逐像素三维轨迹真值，索引对应首帧像素；`occluded`为遮挡标注；`is_bkgd`区分前景与背景，不保证提供单个物体的实例ID。
- WorldTrack `tracks_XYZ` 为每帧相机坐标下的稀疏三维轨迹标注；`visibility`为可见性。
- 两者 `fx_fy_cx_cy` 和 `extrinsics_w2c_gt` 为发布者声明的真实相机参数。
- `depth_map` 和 `extrinsics_w2c` 是DA3/ViPE估计结果，不能当真值。

数据包来自TrackCraft3R作者，但监督目标使用其保存的原始GT轨迹，不使用TrackCraft3R预测轨迹。仍需完整下载后检查数组有限性、相机真值投影、帧内容和物体选择。

## 使用意义与边界

- 同一81帧Kubric序列可使用前22帧和前64帧、同一query点比较上下文长度的影响；共同前22帧评分与完整64帧评分分别报告。
- 该Kubric数据与当前MOVi-F数据及UniQuery4R论文WorldTrack基准均需区分；可用于帧数受控实验，不能直接标作完整论文复现。
- 官方WorldTrack下载入口已确认。若核对论文原始基准，还需取得原始版本，确认序列ID、原始点集与处理一致；增强版保留GT不自动保证所有输入处理与作者一致。
- 仅选择有明确运动目标物体的样本，ADT涉及人类活动时应保留物体表面点。当前未看完整视频，尚不能保证上述具体样本都有适合本任务的目标。
- 当前未下载完整样本、未连接服务器、未启动GPU或模型推理；用户当前query批准及GPU关闭状态不变。

## 来源

1. [TrackCraft3R发布的数据卡](https://huggingface.co/datasets/trackcraft3r/trackcraft3r-eval/blob/main/README.md)：本地requests读取完整2780字节说明，HTTP200。
2. [Kubric单文件目录](https://huggingface.co/datasets/trackcraft3r/trackcraft3r-eval/tree/main/kubric_da3)：目录API列出50个独立NPZ及大小。
3. [官方Kubric读取实现](https://github.com/cvlab-kaist/TrackCraft3r/blob/main/evaluation/eval_dense_kubric.py)：读取真实轨迹和真实相机；按实际可用帧数选取。
4. [TrackCraft3R评测说明](https://github.com/cvlab-kaist/TrackCraft3r/blob/main/evaluation/README.md)：原始GT与估计深度/相机分开保存。
5. [St4RTrack官方WorldTrack入口](https://github.com/HavenFeng/St4RTrack#evaluation-benchmarks)：原始WorldTrack发布者入口；Google Drive具体文件列表本轮未核验。
