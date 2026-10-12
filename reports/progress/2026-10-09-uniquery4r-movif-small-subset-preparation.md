# MOVi-F 小样本真值实验1准备

日期：2026-10-09

## Material Passport

- 工作类型：实验执行准备 / CPU 数据处理。
- 用户授权：选择少量 Kubric / MOVi-F 物体样本，执行实验1；GPU 当前关闭，准备完成后由用户开启。
- 当前状态：四个官方分片下载校验、18场景GT构造及CPU输入检查已完成；18场景查询点总览及完整CPU审计已传回本地，单场景大图和标签副本继续传输。远端推理所需数据完整，等待query确认与GPU开启。尚无 UniQuery4R 推理结果或关系预测训练结果。
- 标签来源：官方 MOVi-F 仿真表面坐标和物体位姿；不依赖教师模型生成真值。

## 已完成

1. 确认允许按文件下载小子集。核查 TAPIP3D 作者转换包首个场景的发布 SHA256、2048条三维轨迹和相机数据；该包缺少实例归属，未用于本轮物体主实验。
2. 选用官方 MOVi-F `256x256 / 1.0.0 / validation` 前四个分片。独立场景数为5+4+5+4=18，总源文件510,919,413字节（约487.25MiB），无需下载整个数据集。官方 GCS 文件 generation 与 MD5 已固定。
3. 实现轻量 TFRecord 读取、实例选择、查询采样和仿真轨迹构造；无需 TensorFlow、模型加载或 GPU。
4. 实现实验1的 GPU 运行接口和 CPU 评分脚本；运行必须显式确认 GPU 和查询预览，GT 文件与模型输入分离。
5. 七项本地 CPU 单元测试通过：TF Example解析、截断拒绝、相机转换、物体表面点身份保持、固定尺度评分、合成场景完整准备与重复生成哈希、图像边界掩码收缩。
6. 全部18个场景均生成 `(22,48,3)` 真值。各场景t0投影误差最大值为0.3534–0.3567px，t0深度相对误差p90为4.719e-6–5.807e-5。仅证明数据接口与仿真标签构造正确，尚未测量教师误差。
7. 四分片均通过发布者MD5检查，同时记录SHA256。下载使用远端aria2c；完整源数据与RGB未下载到本地。
8. 18场景全部通过输入哈希、二维坐标往返和query身份检查。实际调用官方图像预处理，22帧结果为 `(22,3,252,252)`，数值有限；未加载模型或运行forward。

## 推理前修正与复查

- 重复生成时排除哈希清单自身，避免清单自引用导致失配；重复生成测试与最终输入校验通过。
- 掩码收缩显式将图像外部视为背景，避免边缘query超出模型归一化坐标范围；18场景重新采样及检查通过。
- 官方预处理不放大256像素图像，实际尺寸为252×252；配置及query坐标映射已按实际输出修正。此前失败日志保留，最终CPU检查通过。

## 固定协议

- 22帧窗口：观测0–6，后续7–21；query取帧6，48点，无重复采样。
- MOVi-F原生12fps，观测跨度0.5秒、未来跨度1.25秒。本轮检验跟踪教师，不将其误写为 MotionForesight 的15fps预测输入。
- 每场景一个物体：从观测期有平移>0.01仿真单位或转动>0.05rad、t0掩码内有至少200个内部像素的实例中选可见面积最大者。只用0–6帧信息，不根据未来轨迹或教师表现选样。
- 掩码向内收缩2px，再做确定性最远点采样，点间最小间隔2px。查询位置对应t0固定仿真表面点，后续身份由物体局部坐标保持。
- RGB原始256×256，模型按官方预处理使用252×252；不同场景使用一致处理方式。
- 真值轨迹按官方 point_tracking 实现，用`object_coordinates / 65535 - 0.5`与每帧`bboxes_3d`的仿射映射计算。相机四元数和位置给出已知坐标转换；深度为相机到点的距离，不能当作z深度。
- 模型仅接收RGB和二维query。完整22帧是离线教师输入，未来图像不进入未来预测模型的条件。

## 实验1评分

比较教师与独立仿真GT的点位置误差、三维边向量误差、1128个固定点对的长度误差，以及二维输出/三维投影误差。按场景汇总，分开可见与遮挡点，保留全部排除记录。可见性根据实例与深度容差推导，是辅助近似标记；主三维指标保留所有有限真值点。

教师存在单目尺度不确定性，报告原始输出及一项整体尺度调整。该尺度仅由t0点对估计，后续固定；不使用未来标签拟合旋转、平移或尺度，不逐点强制对齐。三维误差用目标真实bbox对角线归一化；绝对误差使用仿真单位，不把模型尺度误称米。

该子集用途是教师误差标定；UniQuery4R是否见过相关训练场景需核查，暂不宣称未见数据泛化。MOVi-F本轮为刚体仿真场景，不能扩展为形变目标的有效性结论。三维GT不自动提供表面测地距离，本轮尚不计算完整表面路径距离。

有这些GT后，后续关系模块训练可以直接用GT。本轮教师比较用于量化教师误差，并非使用GT训练的必要前提。

## 文件

- `configs/uniquery4r_movif_experiment1_2026-10-09.json`
- `scripts/prepare_uniquery4r_movif_gt.py`
- `scripts/run_uniquery4r_movif_experiment1.py`
- `scripts/evaluate_uniquery4r_movif_gt.py`
- `scripts/test_prepare_uniquery4r_movif_gt.py`
- `scripts/package_uniquery4r_movif_gt_previews.py`
- 本地预览：`experiences_results/uniquery4r-movif-experiment1-2026-10-09/`
- 全部18场景query总览：`experiences_results/uniquery4r-movif-experiment1-2026-10-09/query_contact_sheet.jpg`
- CPU证据：同目录的 `gt-preparation-status.json`、`source-audit.json`、`inference-input-audit.json`、`official-preprocessing-cpu.json`。
- 远端数据：`/root/autodl-tmp/uniquery4r-movif-experiment1/`

## 后续条件

CPU准备已完成，18场景总览已提供。请用户确认query并开启GPU，再执行22帧教师推理与独立真值比较。单场景本地副本传输不影响远端准备状态；未声称这些副本已全部校验。当前不得执行模型forward、训练或伪造实验1数值结果。

## 来源

- [Kubric / MOVi 官方说明](https://github.com/google-research/kubric/blob/main/challenges/movi/README.md)
- [MOVi-F 官方数据定义](https://github.com/google-research/kubric/blob/main/challenges/movi/movi_f.py)
- [官方物体表面点轨迹构造](https://github.com/google-research/kubric/blob/main/challenges/point_tracking/dataset.py)
- [官方数据集分片信息](https://storage.googleapis.com/kubric-public/tfds/movi_f/256x256/1.0.0/dataset_info.json)
- [TAPIP3D 数据说明](https://github.com/zbw001/TAPIP3D/blob/main/DATASET.md)

技能影响：academic-research-suite用于固定协议与区分准备、教师评价和方法训练；research-progress-logger保存脚本、证据和准备状态。
