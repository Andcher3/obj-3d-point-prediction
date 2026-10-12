# UniQuery4R 官方相机、展示与评测来源核查

日期：2026-10-10

## 完成工作

使用 academic-research-suite 的 fact-check/source-verification 流程，核查论文全文及补充材料、官方网页、公开代码、完整仓库文件树、GitHub 全状态 issues 与 Hugging Face 社区。使用 research-progress-logger 保存记录。本次没有连接服务器、运行模型或改动实验数组。

官方 main 当前提交为 `95119742e61c64c27b09349035019c8cd7b41e62`，与部署提交相同。GitHub 递归文件树 `truncated=false`。

## 已确认的事实

### 1. 公开推理的相机参数来自模型

- 模型相机头预测每帧三维平移、四元数旋转、垂直/水平 FoV（视场角）。
- `pose_encoding_to_extri_intri` 用图像尺寸与 FoV 转成 fx/fy，主点固定为图像中心。它没有在推理接口接收真实相机内外参。
- 外参通过 `normalize_cameras_to_first` 转入输入第0帧相机参考系；第0帧参考系不等于本实验的query帧6。
- 模型源码明确在混合精度环境外使用 FP32 进行相机解码，注释说明 FP16 位姿运算可能漂移。此为已实施的数值处理，不能解释为已证实本实验的焦距偏差原因。

来源：[模型](https://github.com/KosmoResearch/UniQuery4R/blob/95119742e61c64c27b09349035019c8cd7b41e62/uniquery4r/models/uniquery4r.py)，[相机转换](https://github.com/KosmoResearch/UniQuery4R/blob/95119742e61c64c27b09349035019c8cd7b41e62/uniquery4r/utils/pose_enc.py)，[论文第3节及补充A Camera Head](https://arxiv.org/html/2608.17283v1)。

### 2. 场景点云导出与动态轨迹展示使用不同输出路径

- `scripts/infer_4d.py` 默认 `--points-from depth`：预测深度加预测内外参逆投影成每帧点云；可切换 `warp3d` 自配对三维点。
- 同一脚本另行生成 `frame_points` 动态轨迹，默认使用跨帧 `warp3d` 绝对三维位置；可选择首帧 `global_points[0]` 加独立位移输出 `warp3d_delta`。此两种预测路径没有被强制设为一致。
- `scripts/view_4d.py`/`MotionViewer` 展示动态 `frame_points`，背景使用 `global_points`（自配对 warp3d），并未改用前述默认深度导出点云。
- 官方主页是可旋转视角的 Three.js 三维浏览器，不是本项目的原始RGB逐像素投影检查。前端读取预生成的三维坐标、轨迹及相机框。
- 为核查来源，读取了 drift-chicane 与 scooter-black/tma 的小型 manifest，仅含显示/坐标/二进制索引等字段，没有原始内参、生成命令或焦距校正来源。scooter-black/tma 的展示对齐为自身单位变换。前端 `view-calibrations.json` 的 position/target 是浏览器观看位置与朝向，不能当成采集相机内外参的校准。
- 未下载网页二进制场景/视频。没有找到逐个主页资产生成脚本，不能承诺每个资产完全等同默认CLI，也不能声称资产用了真实内参或人工焦距校正。

来源：[推理脚本](https://github.com/KosmoResearch/UniQuery4R/blob/95119742e61c64c27b09349035019c8cd7b41e62/scripts/infer_4d.py)，[轨迹浏览器](https://github.com/KosmoResearch/UniQuery4R/blob/95119742e61c64c27b09349035019c8cd7b41e62/uniquery4r/vis/motion_viewer.py)，[主页](https://kosmoresearch.github.io/UniQuery4R/)，[汽车展示元数据](https://kosmoresearch.github.io/UniQuery4R/gallery-data/drift-chicane/manifest.json)，[对照展示元数据](https://kosmoresearch.github.io/UniQuery4R/comparison-data/scooter-black/tma/manifest.json)。

### 3. 论文三维评测经过真值辅助的整体尺度对齐

- WorldTrack 使用第一帧为查询来源、前64帧；动态点表4报告64帧×50序列，四数据集等权平均。
- 三维点和位移分别按 `median(norm(GT))/median(norm(pred))` 校准一个整体尺度，再计算指标。此处需要评测真值，不是推理时拥有米制尺度。
- 动态点指标直接比较参考坐标系三维位置，不能据此推导预测内参或原图回投影准确。没有证据表明表4以GT相机替换预测相机再重新生成三维轨迹；完整评测脚本未包含在当前公开代码树中，只能按论文声明核查。
- 相机表5报告相机姿态/轨迹的 AUC、ATE、相对平移及相对旋转误差；没有独立焦距/FoV准确性表。
- 本实验采用query帧6、252×252、目标物体48点、22帧及固定t0点对尺度，和论文评测条件不同。论文EPE不能直接与本实验绝对三维误差数值对比。

来源：[论文第4节表3–6及补充A尺度约定](https://arxiv.org/html/2608.17283v1)，[官方README评测说明](https://github.com/KosmoResearch/UniQuery4R#evaluation-on-worldtrack)。

## 相似问题的公开记录

- GitHub `issues?state=all&per_page=100` 返回 `[]`；仓库issue搜索也为空，公开issue页面显示Open 0。截至本次检查未找到公开或已关闭的偏移/焦距问题讨论。
- Hugging Face Community 页面无讨论、closed=0；通用检索 UniQuery4R + focal/intrinsics/projection/misalignment/drift 未找到作者确认同类问题的报告。检索空缺不证明问题不存在。
- 论文正文限制作出的是严重遮挡、大旋转、快速运动及对称部位身份交换等说明；未找到专门解释“点云收缩、查询帧定位偏移、焦距预测偏差”的段落。
- 官方README示例把 query_uv 写作像素单位，但 forward 源码要求 [0,1]；这一接口文档不一致此前已在部署前发现。当前实验准备/运行脚本确实使用归一化query，不能把该文档问题当成现有偏移的已证实原因。

来源：[GitHub issues](https://github.com/KosmoResearch/UniQuery4R/issues)，[全状态issues API](https://api.github.com/repos/KosmoResearch/UniQuery4R/issues?state=all&per_page=100)，[HF社区](https://huggingface.co/Kosmo-Research/UniQuery4R/discussions)，[论文限制](https://arxiv.org/html/2608.17283v1)。

## 证据判断与限制

官方代码是公开推理实际参数来源的直接证据；论文是评测协议与作者结果的原始描述，仍为预印本，本次未独立复现排行榜。主页与作者仓库不是独立重复实验来源。作者展示和自报指标有同源局限；没有新的资金利益冲突证据，不作未核实的期刊/同行评审认定。源码适合验证实现事实，不能代替泛化性能证据。

本实验支持相机内参存在明显独立误差，三维点也存在独立误差。外参平均表现相对较好，但scene_5702的未来旋转误差约11.91°，平移比较还依赖尺度，不能推广为所有外参均准确。三维浏览器中预测点云与预测轨迹看起来协调，不足以证明它们与原RGB或真值一致。

当前状态：官方来源核查完成；没有开启GPU、新推理、训练或替换教师输出。逐个主页展示资产的后处理与论文完整评测实现仍未公开核实。
