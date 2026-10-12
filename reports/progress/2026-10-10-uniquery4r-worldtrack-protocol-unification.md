# UniQuery4R：WorldTrack评测规则统一与CPU检查

日期：2026-10-10

当前范围：用户随后取消了64帧匹配要求。64帧入口仅保留为可选代码；实际准备改为现有MOVi-F的24帧，见`2026-10-10-uniquery4r-movif-protocol-sync.md`，无需获取WorldTrack数据或开启GPU。

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent；research-progress-logger
- Origin Mode: validate / code verification
- Verification Status: VERIFIED（仅CPU代码测试与预处理；论文性能尚未复现）
- Version Label: protocol_cpu_v1
- Overall Confidence: CAUTION（UniQuery4R完整评测入口未公开，实际数据与GPU推理待验证）

## 完成工作

按用户要求建立独立的64帧评测入口，保留旧MOVi-F数据、预测和评分。核查UniQuery4R论文、公开模型接口及固定版本WorldTrack代码。本轮没有下载新数据、加载模型、运行forward、调用GPU或训练。

此入口用于检查轨迹教师能否达到论文报告的跟踪精度；项目的目标物体未来轨迹预测任务不变。WorldTrack完整基准包含非物体目标，当前未下载或运行这些数据，不将基准点集规则替换为本项目的训练目标定义。

## 已统一的公开规则

| 项目 | 新入口规则 |
| --- | --- |
| 时间输入 | 原始序列前64帧；不重复、不插值、不用22帧缓存替代 |
| query帧 | 输入第0帧，直接调用UniQuery4R原生接口 |
| 点集 | 使用首帧可见、画面内的全部标注点；不限制48点，不做物体mask收缩筛选 |
| 动态点 | 用GT连续帧三维位移长度的累计和大于0.01筛选；不是只比较起点与终点 |
| 遮挡 | 点选定后，后续遮挡不排除其三维评分 |
| 坐标 | 真值与预测统一在输入第0帧相机坐标系 |
| 尺度 | 在动态点子集上重新求GT和预测点到原点距离中位数之比；只调整一个整体尺度 |
| 三维输出 | 评分原始warp3d绝对位置，不改用warp3d_delta，不进行焦距拟合或逐点平移 |
| 指标 | EPE平均三维位置误差；APD为0.1/0.3/0.5/1.0米四阈值命中率的平均，边界采用<= |
| 汇总 | 先求各数据集的视频均值，再对四个数据集等权平均；完整比较要求各50段 |
| 模型输入 | RGB与二维query；真值相机、深度、轨迹仅用于选点及评测，不输入模型 |
| 预处理 | 官方max_size=504、patch=14保持比例处理；64帧一次编码，允许query轴分块，不使用时间窗口拼接 |

附加检查会显式拒绝非有限轨迹、缺失真值外参或不足64帧的文件。参考实现对非有限值有不同处理，因此评分函数一致性的证明范围为有限、有效输入。无动态点的视频记录为空，禁止把缺少动态点的情况当完整复现通过；原参考汇总在这种情况下存在EPE/APD分母差别，当前没有验证原200段是否涉及这一边界情况。

## 相机字段来源

WorldTrack原始文件的extrinsics_w2c是真值外参。TrackCraft3R发布的增强版数据中，同名字段是估计外参，真值存储在extrinsics_w2c_gt。代码强制区分这两种来源，增强版必须使用真值字段转换GT。没有真值外参时拒绝用单位矩阵代替。

## CPU验证

- 本地20项测试通过；远端同样20项通过。
- 每次测试包含120组随机有限轨迹与固定版本官方评分函数的逐项比较；尺度、EPE、APD及四个阈值命中率一致。
- 检查首帧选点、后续遮挡保留、动态累计位移、动态子集重新计算尺度、外参来源、相机坐标转换、帧对提取、四数据集等权汇总、输入SHA256及未授权推理拒绝。
- 远端使用合成64帧输入完成官方load_and_preprocess_images检查，形状、数值范围及文件哈希通过；该检查没有构造模型或加载权重。
- 四个新脚本编译及命令行帮助检查通过。

证据：

- `experiences_results/uniquery4r-worldtrack-protocol-2026-10-10/local-cpu-tests.json`
- `experiences_results/uniquery4r-worldtrack-protocol-2026-10-10/remote-cpu-tests.json`
- `experiences_results/uniquery4r-worldtrack-protocol-2026-10-10/remote-official-preprocess-check.json`

## 文件

- `scripts/uniquery4r_worldtrack_protocol.py`：选点、真值坐标、动态点、尺度、指标与汇总。
- `scripts/prepare_uniquery4r_worldtrack.py`：真实64帧准备、query预览、来源与哈希登记。
- `scripts/run_uniquery4r_worldtrack.py`：默认CPU审计，可选官方CPU预处理；GPU和query均明确批准后才允许推理。
- `scripts/test_uniquery4r_worldtrack_protocol.py`：20项CPU测试及120组官方函数比较。
- `configs/uniquery4r_worldtrack_protocol_2026-10-10.json`：参数与尚未核实项目。
- `reports/reference_code/worldtrack-st4rtrack-0f9a3f4/`：官方评分函数文本快照及来源说明。

远端独立工作目录为`/root/autodl-tmp/uniquery4r-worldtrack-protocol-cpu-2026-10-10`，原实验目录未修改。

## 当前限制与下一步状态

1. 现有MOVi-F原视频只有24帧，缓存为22帧/query6；无法提供真实连续64帧，不能通过改元数据复用成该协议。
2. 本轮尚未取得并核对WorldTrack原始200段数据，因此没有真实64帧数据预处理或推理结果。
3. UniQuery4R未公开完整评测脚本。默认二维query归一化沿用其致谢的OpenD4RT公开稀疏评测做法；另保留像素中心缩放方式。两者明确登记，后续不按最好指标择优。作者实际归一化方式仍待核实，不能声称整个作者入口逐字等价。
4. UniQuery4R原生接口与本次调用方式已静态核对；64帧GPU推理、峰值显存和模型指标未测试。
5. 原始WorldTrack下载入口在St4RTrack README中；TrackCraft3R增强版PointOdyssey只有23段，不作为原50段的完整替代。本轮未下载数据。

代码与CPU检查阶段完成。当前无需开启GPU；真实64帧数据准备和query预览确认完成后，才具备开展同规则推理的输入条件。现有误差差距的原因仍未由本轮代码检查解释。

## 主要来源

- [UniQuery4R论文](https://arxiv.org/html/2608.17283v1)：WorldTrack数据、前64帧、动态点评价及论文指标。
- [固定版本WorldTrack点集与动态点评测](https://github.com/HavenFeng/St4RTrack/blob/0f9a3f44a7ebac76600cd31ec9eea5228ad7db91/dust3r/track_eval.py)：首帧可见点、动态位移阈值及动态子集评分。
- [固定版本评分函数](https://github.com/HavenFeng/St4RTrack/blob/0f9a3f44a7ebac76600cd31ec9eea5228ad7db91/dust3r/track_eval_util.py)：整体尺度、EPE及阈值命中率。
- [固定版本真值读取器](https://github.com/HavenFeng/St4RTrack/blob/0f9a3f44a7ebac76600cd31ec9eea5228ad7db91/dust3r/datasets/tapvid3d.py)：前64帧及相机坐标转换。
- [增强版数据字段说明](https://huggingface.co/datasets/trackcraft3r/trackcraft3r-eval/blob/main/README.md)：预测外参与真值外参字段、PointOdyssey段数。
- [OpenD4RT公开稀疏评测入口](https://github.com/Lijiaxin0111/Open-d4rt/blob/main/eval_track3d_in_worldtrack.py)：默认query归一化候选的来源。
