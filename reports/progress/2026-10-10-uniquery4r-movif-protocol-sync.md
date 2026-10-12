# MOVi-F：除帧数外同步公开评测规则

日期：2026-10-10

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent；research-progress-logger
- Origin Mode: validate / CPU preparation
- Verification Status: VERIFIED（代码测试、真实输入准备与CPU预处理）；性能尚未测试
- Overall Confidence: CAUTION（数据集、点集定义、单位及作者未公开的入口细节仍有差别）

## 当前决定

按用户最新要求取消64帧匹配，不下载WorldTrack数据。使用现有18个MOVi-F场景，每段原生24帧，保留原目标物体身份。旧22帧/第6帧query/48点实验及其结果保持原样。

## 已完成的同步

- query改为输入第0帧，使用UniQuery4R原生首帧稀疏接口。
- 原来的掩码收缩、48点采样取消，改用第0帧目标物体掩码内的全部候选表面像素。
- 用仿真材质坐标和物体真值变换重建这些点的24帧真实轨迹；按首帧可见、有限、画面内和正深度筛选，保留后续遮挡点的三维标签。
- 首帧可见性沿用同实例mask与2%射线深度一致性定义，没有根据模型预测或未来误差排除点。
- 动态点定义为GT连续帧三维位移长度累计和大于0.01；动态点子集单独计算一个整体median-norm尺度（GT和预测点到原点距离中位数之比）。
- 评分warp3d三维绝对位置，采用EPE及四阈值命中平均；不拟合焦距、不逐点平移、不额外做旋转/平移对齐。
- 场景等权汇总；MOVi-F单独标记，不输出伪造的WorldTrack四数据集平均。

MOVi-F没有WorldTrack预先发布的同一套稀疏轨迹点，因此本轮“完整点集”具体指指定目标物体首帧可见的全部表面像素。没有扩大到其他物体、人物或背景；项目任务不变。

## 实际CPU结果

- 18/18场景准备成功，并通过官方24帧RGB预处理、形状和输入SHA256检查。
- 每段query数860–7697，共71,111点，全部通过累计位移动态点规则。
- 完整目标候选材质点首帧真值投影误差p90及最大值约0.35–0.36像素；这一误差来自仿真像素/材质坐标离散构造，不是模型预测误差。
- 本地及远端各22项协议测试通过；包含120组固定官方评分函数比较。
- 本地7项MOVi-F真值测试通过，新增完整首帧表面点重建检查：13,456点、不收缩或采样、24帧身份与运动一致。
- 四个原始数据分片发布MD5通过，处理前后SHA256一致；没有新增下载。
- GPU始终未调用，没有模型加载、forward、预测指标或训练。
- 回传query总览SHA256与远端一致：`123acc779fb4e7d51799d65d4ae3a3d637c085780784aeb9371b8352a0be9792`。SSH连接已关闭。

## 文件与证据

- 新准备脚本：`scripts/prepare_uniquery4r_movif_protocol_sync.py`。
- 参数：`configs/uniquery4r_movif_protocol_sync_2026-10-10.json`。
- 通用入口支持显式`--frames 24`；64帧默认仅保留为可选WorldTrack路径。
- 远端新输入：`/root/autodl-tmp/uniquery4r-movif-protocol-sync-2026-10-10`。
- 本地准备记录：`experiences_results/uniquery4r-movif-protocol-sync-2026-10-10/preparation.json`。
- 真实输入CPU检查：同目录`cpu-input-check.json`。
- 新query总览：同目录`query_contact_sheet.jpg`。黄色为完整目标点集，密集点会覆盖物体外观。
- 协议测试：`experiences_results/uniquery4r-worldtrack-protocol-2026-10-10/{local,remote}-cpu-tests-native24.json`。
- 公开规则及源码引用：`reports/progress/2026-10-10-uniquery4r-worldtrack-protocol-unification.md`。

## 尚未得到的结论

新query及帧序列需要重新推理；旧48点输出不能直接变成新完整点集结果。尚无证据证明上述同步会缩小指标差距，也没有获得新模型精度结果。

动态点筛选在本轮新点集中保留了全部点，因此它本身没有改变本轮GT点集。原旧缓存是否全部满足同一动态点规则，本轮没有重新评分。

MOVi-F的阈值仍以仿真场景单位表达，米制等价未核实；作者完整query归一化入口未公开，默认沿用其致谢的OpenD4RT公开方式。不能将本轮称为完整WorldTrack论文复现。可以在新推理完成后，检查首帧query及完整点集规则下的本地表现。

下一步状态：CPU准备完成，query预览已传回；等待用户确认新query并明确开启GPU。未运行推理。
