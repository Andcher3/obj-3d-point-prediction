# MotionForesight 视频未来轨迹研究进展

ACTIVE_RESEARCH_SCOPE=VIDEO_MOTION_FORECASTING

本索引只记录当前完全基于视频的研究方向。旧 IMU/HAR/OIS 索引已保存到 `archive/imu/progress_log.pre-2026-09-03.md`。

## 进展索引

- 2026-10-12 [项目上传到 GitHub 私有仓库](progress/2026-10-12-github-private-workspace-upload.md)：创建 Andcher3/post_graduate_project 并推送 main；代码、文档及适量图片视频上传，3,531 个数据/权重/缓存等原文件使用 218 个目录占位说明代替，本地原文件保留。新增上传清单和排除规则；无训练、推理或历史 IMU 命令执行。

- 2026-10-11 [同物体点共享用于未来三维预测的针对性查新](progress/2026-10-11-object-point-sharing-novelty-search.md)：完成关键词与引用链核查；HRN、DPI-Net、VGPL、AdaptiGraph及HOPNet存在明确思想先例，区分状态/动作条件与视频任务差异。二维生成、已见帧跟踪及一般全场景注意力单列相关工作；本项目待定具体结构未判重复。记录原文位置及访问限制，无实验、远端或方法变更。

- 2026-10-11 [物体内共享运动的相关与重合分类修正](progress/2026-10-11-object-sharing-overlap-classification.md)：按用户确认的未来三维、物体内点共享范围复核；TrajLoom、跟踪及一般点注意力归为相关工作，不能据此宣布完整 idea 已实现。VGPL 原文存在物体归属图交互及共享运动的思想级先例；具体方案查新仍未完成。保留原报告并添加更正，无实验或方法更改。

- 2026-10-11 [物体点集一致性文献、指标与注意力对照](progress/2026-10-11-object-coherence-literature-metrics-attention.md)：核查 CoTracker、SpatialTracker、Gen-Points、TrajLoom、VGPL 等已有共享/协调机制；官方代码确认 MotionForesight 潜在格点已有跨时空注意力。提出真值相对的点对长度、异常点与时域结构评价，以及等输入/预算普通注意力与专门结构的对照；宽泛共享观点不构成独立创新。仅文献/代码审查和设计，无远端、GPU或新实验。

- 2026-10-11 [点间关系证据与研究价值判断](progress/2026-10-11-relation-evidence-synthesis.md)：归纳 E0/E1、早期联合解码及有真值的 G1/G2/G2-L。邻点信息对相对结构有正向增量证据，未来动态收益未建立；可见表面路径条件和辅助损失缺乏增量价值。建议保留精简跨点协同研究，停止扩展当前表面距离组件；本轮只分析，未更改方案或启动实验。

- 2026-10-11 [G2-L 未来表面距离辅助损失完成](progress/2026-10-11-future-surface-distance-aux-loss.md)：100/20/40 场景、三组同结构小模型、21 次 GPU 拟合共 56.07 秒完成。表面组 ADE 3.529116，对坐标组增加 1.112%（两比较 97.5% 区间 0.342%–2.042%），对直线距离组增加 0.564%（0.095%–1.133%）。当前辅助损失无增量收益；仅刚体真值可见路径与共享表示辅助监督，未验证完整网格或直接输出几何约束。15 份证据 SHA256 通过，无教师或自动追加调参；GPU 阶段结束。

- 2026-10-11 [G2 观测可见表面条件对照完成](progress/2026-10-11-surface-g2-visible-conditions.md)：160 场景、5 组同容量线性预测器 CPU 38.27 秒完成。表面邻接＋路径相对二维邻点＋直线 ADE 增益 0.075%（95%区间 −0.041%–0.222%），未见明确实用增益；真值可见表面 ADE 增益 0.014%。全部拟合组验证集均劣于原始 baseline，保留限制；11 份证据回传 SHA256 通过。仅可见图路径、复用 G1 测试集的探索性对照，无完整网格或未来表面距离损失，无 GPU 或自动调参。

- 2026-10-11 [跨点信息实验 GPU 执行及结果](progress/2026-10-11-cross-point-probe-gpu-run.md)：160 场景观测预测缓存与用户批准的仅标签/评价坐标标定完成；100/20/40 场景匹配对照，线性预测器加邻点历史的 ADE 增益 2.47%（两比较区间 0.35%–4.35%），点间向量误差增益 8.14%，未达事前 5% 主门槛。MLP 六次训练均验证选择零修正，不能推断信息无价值；动态改善尚不明确。全部 GPU 阶段退出 0、10:31 结束，11 份证据回传 SHA256 通过；无独立 GPU 性能复跑。

- 2026-10-11 [跨点信息实验 CPU 准备完成](progress/2026-10-11-cross-point-probe-cpu-ready.md)：全部资产与 DA3 固定源码/依赖就绪；160 场景 RGB/query/时间/原生预处理通过，160 张预览回传 SHA256/CRC/逐文件校验并生成审阅页面。9+7 项合成测试本地/远端通过，离线资源映射及未授权加载保护通过；数据盘剩 11.66GiB，无模型加载/forward/训练。等待用户开启 GPU，实际模型接口与预测/GT 尺度审计仍需完成，query 人工确认 pending。

- 2026-10-11 [跨点信息实验下载完成核查](progress/2026-10-11-cross-point-probe-download-complete.md)：全部已列数据和模型资产下载完成，TrackCraft3r/DA3及Wan模型发布者大小/SHA256通过，源码和微调包CRC通过；第二轮恢复成功，任务于10月10日21:51完成。00:07数据盘剩12.23GiB。CPU依赖/接口核查及完整query预览仍待完成，无GPU、模型加载或训练。

- 2026-10-10 [下载三段81帧Kubric真值样本](progress/2026-10-10-long-kubric-subset-download.md)：远端aria2c下载00001/00002/00003共535MiB，官方SHA256、ZIP CRC、81帧完整解码及GT有限性通过。补足世界坐标到首帧相机转换后，留出p90均低于0.005px；未来投影未独立核验。全量坐标检查137退出，分块读取复核退出0；本地回传约670KiB预览/证据并逐项验哈希。未选择query或使用GPU，完整数据留远端。

- 2026-10-10 [64帧以上真值样本获取核查](progress/2026-10-10-long-groundtruth-sequence-availability.md)：远程文件头确认3个Kubric样本各81帧，其中00002带(399360,81,3)轨迹GT；另确认ADT样本300帧/825点。单文件可下载，三个Kubric共约535MiB；区分真实相机与DA3估计字段。完整视频及物体适用性尚未核验，无完整下载、远端连接或GPU运行。

- 2026-10-10 [跨点信息实验定时检查与连接恢复](progress/2026-10-10-cross-point-probe-heartbeat-access.md)：15:37原SSH会话访问失败；15:45用户要求重查后重新连接成功，远端下载正常，总约1.2MiB/s、TrackCraft3r约9%且预计剩余6小时23分，数据盘剩33GB。微调包/GT核验保留，全部底座尚未完成，无GPU运行。

- 2026-10-10 [跨点信息独立预测实验无卡准备](progress/2026-10-10-cross-point-probe-cpu-preparation.md)：42个MOVi-F分片、100/20/40场景GT及微调包审核通过，Wan DiT完成且发布者大小/SHA256通过。21:47 Track99.19%（16.407/16.542GiB），剩约138MiB，8连接、0.202MiB/s，瞬时ETA约11分钟；最终哈希未完成。SSH会话59555恢复且指纹一致，下载进程正常，数据盘剩12.38GiB。没有本地大模型下载、GPU或训练，全部资产与CPU接口准备尚未完成。

- 2026-10-10 [协议变化核查与query批准](progress/2026-10-10-uniquery4r-protocol-difference-and-query-approval.md)：用户批准18场景query，GPU授权仍关闭。核对原固定48点与当前名义密度采样、动态子集、整段尺度及位置评分；确认遮挡保留等原已具备的规则未变，旧缓存已做尺度重算，新点集尚无推理结果。保留数据集、22/64帧、query6/0与单位等差别，未将其标为完整论文复现。

- 2026-10-10 [第6帧query的名义密度采样准备](progress/2026-10-10-uniquery4r-density-query6-preparation.md)：8.095px间距，原22帧/query6及18个物体不变；13段网格采样50–134点、5段精确复用旧48点，共1280点。本地34项与远端12项CPU测试、全部官方图像预处理通过；77份预览/证据文件已回传校验，等待新query确认。首帧稠密候选保留为历史方案；后续同一官方序列22/64帧受控对照已记录。CPU半核配额下多线程超时已处理，无GPU、模型调用或新性能结果。

- 2026-10-10 [跨点信息价值与通信结构的分离检验](progress/2026-10-10-cross-point-information-value-design.md)：建议在G1前用冻结输入的线性/小型MLP分别比较真实邻点历史与屏蔽邻点历史，再检验相同联合结构。区分信息增量价值、读取器效果及观测估计噪声；未来GT仅监督，有限负结果不否定所有关系方法。仅完成设计审查，未连接服务器或训练。

- 2026-10-10 [真值监督跨点关系实验准备判断](progress/2026-10-10-surface-relation-gt-training-readiness.md)：建议进入G1，无需等待教师；保留联合关系核心，以baseline、等容量逐点、邻点交互三组匹配对照。18场景作开发检查，正式训练建议100/20/新增40场景；仍需MotionForesight仅观测缓存、query身份、12/15fps及坐标核查。表面距离留G2，未启动GPU、训练或修改教师协议。

- 2026-10-10 [密度与48点回退前置检查](progress/2026-10-10-uniquery4r-density-fallback-precheck.md)：名义密度估算及首帧可见性检查完成。用户随后确认推算近似间距、保留query6和精确旧48点回退；此前待确认事项已解决，当前实现见上条记录。

- 2026-10-10 [稀疏query数量与当前点集核查](progress/2026-10-10-uniquery4r-sparse-query-count-check.md)：WorldTrack原始准备说明为每序列均匀下采样约1,000点；当前MOVi-F每个目标逐像素保留860–7,697点，平均3,950.61，属于目标区域稠密点集。确认“全部已有稀疏标注”与“全部mask像素”不等价，不能称采样规则已同步。只读核查，无代码修改或推理。

- 2026-10-10 [MOVi-F除帧数外同步公开评测规则](progress/2026-10-10-uniquery4r-movif-protocol-sync.md)：按用户最新要求取消64帧匹配，保留18个目标物体的原生24帧；改为首帧完整可见表面点、不收缩或抽48点、动态子集尺度及遮挡评分。18/18真实输入CPU预处理通过，共71,111点、每段860–7697点，均满足动态规则；本地/远端各22项协议测试及本地7项GT测试通过。预览已回传，无GPU或新预测结果，等待query确认。

- 2026-10-10 [统一UniQuery4R的WorldTrack评测规则与CPU检查](progress/2026-10-10-uniquery4r-worldtrack-protocol-unification.md)：新增真实前64帧/query0/全部首帧可见点/GT动态点及动态子集尺度评测入口，明确增强版真值外参字段。本地与远端各20项CPU测试通过，含120组官方函数比较；合成64帧官方预处理通过，无模型加载或GPU。现有MOVi-F只有24帧，原始基准数据尚未取得，作者完整归一化入口仍待核实，未复现论文性能。

- 2026-10-10 [论文尺度公式下的18场景诊断比较](progress/2026-10-10-uniquery4r-paper-style-scale-comparison.md)：只读CPU重算median-norm尺度，全22帧位置EPE 0.681场景单位、四数值阈值平均命中35.47%；未来EPE从旧点对尺度2.389变为0.694，预测数组未改。数据集/帧/query/单位/动态筛选不同，不能标为WorldTrack严格复现或直接与论文0.1601米/83.89%同条件比较。18场景来源哈希及原评分复算通过，无GPU或训练。

- 2026-10-10 [UniQuery4R官方相机、展示与评测来源核查](progress/2026-10-10-uniquery4r-official-camera-visualization-check.md)：公开推理内外参由模型估计，主点固定中心；区分深度点云导出、warp3d动态轨迹及三维浏览器。论文3D指标含GT辅助尺度对齐，相机表未单列焦距准确性；全状态GitHub issues与HF社区无同类公开记录，逐个主页资产生成链仍未核实。无远端连接、GPU或数组修改。

- 2026-10-10 [拟合焦距投影后原内参逆投影的三维比较](progress/2026-10-10-uniquery4r-focal-backprojection-3d.md)：保留原相机Z、共用原t0尺度，18场景3D位置误差2.389→2.725仿真单位，向量误差/bbox 9.66%→14.48%，长度误差13.30%→26.26%，2D却23.48→5.46px。额外独立t0尺度可降低绝对位置误差，但点间关系仍退化。原始基线复现、来源哈希一致、7项相关测试通过；无GPU、训练或原数组覆盖。

- 2026-10-10 [MOVi-F三维点与相机交叉归因](progress/2026-10-10-uniquery4r-movif-geometry-camera-attribution.md)：18场景CPU检查8种预测/GT组合，预测点+完整真实相机仍18.18px，真实点+预测相机28.16px；首帧固定真实相机仍19.06px，真实点+真实外参+预测内参26.50px。支持三维点和相机均有误差，不能按像素差值分摊贡献。来源哈希与既有审计一致、4项测试通过；无GPU、训练或数组修改。

- 2026-10-10 [18场景真实焦距对照与准确性复核](progress/2026-10-10-uniquery4r-movif-all-true-focal.md)：其余17段五列视频已生成并交付，scene_5702复用且核对；18段22帧完整解码与文件SHA256通过。t0两轴平均焦距误差预测26.57%、拟合30.58%，预测更接近真值10/18；真实焦距投影仅9/18优于原投影、1/18优于拟合。拟合画面改善不能等同物理焦距更准，原XYZ不变，无GPU或训练。

- 2026-10-10 [scene_5702真实焦距与二维偏移检查](progress/2026-10-10-uniquery4r-scene5702-true-focal.md)：CPU生成并回传五列视频；仅换真实焦距后未来投影误差34.64→42.04 px，未消除偏移。query转换及t0帧对核对通过，原二维t0误差11.50 px、横向分散程度约query的55%；未来绝对/相对位移误差17.33/7.77 px。三维输出未改、GPU保持关闭，无训练。

- 2026-10-10 [MOVi-F焦距重估与四列对照](progress/2026-10-10-uniquery4r-movif-focal-review.md)：18场景只用t0已知query拟合fx/fy，固定比例应用未来；未来可见点平均投影误差23.48→5.46 px，18/18下降。拟合焦距不都更接近仿真真值，不能等同三维校正；XYZ与原三维评分保持不变。CPU生成18段四列视频并回传校验完成，GPU保持关闭，无模型推理或训练。

- 2026-10-10 [交付MOVi-F实验1的18段对照视频](progress/2026-10-10-uniquery4r-movif-visual-delivery.md)：GPU关闭后通过前缀哈希续传约7MB，完整15MB视频包及包内文件SHA256通过；18段H.264均完整解码22帧，本地播放页及t0/分帧图已生成。原始NPZ仍在远端，无模型重跑或训练。

- 2026-10-09 [完成MOVi-F实验1教师与真值比较](progress/2026-10-09-uniquery4r-movif-experiment1-results.md)：已用当前批准query完成18场景22帧UniQuery4R推理与GT评分，均退出码0；未来三维向量误差场景平均为bbox对角线9.66%，点间长度相对误差13.30%。18段模型二维/三维投影/GT对照视频及H.264版本已生成，10-10已交付本地并验证；未训练关系模块。

- 2026-10-09 [准备MOVi-F小样本真值与UniQuery4R实验1](progress/2026-10-09-uniquery4r-movif-small-subset-preparation.md)：官方验证集四分片、18个独立物体场景约487MiB，远端aria2c下载及发布MD5检查完成；全部48点GT、输入哈希和官方252×252 CPU预处理检查通过，七项单元测试通过。18场景query总览与完整CPU审计已传回，单场景副本继续传输；远端已准备完成，等待query确认及GPU开启，无推理或训练。

- 2026-10-09 [核查三维真值数据并规划关系预测对照](progress/2026-10-09-surface-relation-groundtruth-experiment-plan.md)：确认PointOdyssey仿真真值、作者发布的Kubric/MOVi-F轨迹及TAPVid-3D真实标注限制；建议分开跨点通信、表面距离、GT/UniQuery4R监督三个对照，按独立物体场景训练测试。仅制定建议，无GPU、下载或训练。

- 2026-10-09 [复述表面关系引导的物体轨迹预测idea](progress/2026-10-09-surface-relation-idea-restatement.md)：核查既有记录，保留观测表面关系、跨点共享运动信息、时变三维关系与联合坐标预测；区分联合解码器候选、早期图积分和未实现的表面距离模块。记录用户对现有伪标签可用性的判断，无模型或实验变更。

- 2026-10-09 [核查焦距修正对三维点间距离的影响](progress/2026-10-09-uniquery4r-focal-distance-invariance.md)：此前drift-turn修正仅替换投影焦距，三维点保持不变；三维距离差异因此严格不变，统一尺度后中位数仍14.19%、p90仍31.24%。源码与原始哈希已核查，无模型调用。

- 2026-10-09 [比较UniQuery4R观测与完整输入的点间距离](progress/2026-10-09-uniquery4r-context-relative-distance-comparison.md)：仅用本地CPU比较四视频共同7帧的1128点对；统一t0尺度后差异中位数为car-turn 3.35%、drift-turn 14.19%、boat 0.99%、soccerball 8.54%。局部边差异更大，足球早期帧不一致；无GPU调用或原始轨迹修改，未验证三维真值或启动训练。

- 2026-10-09 [整理DAVIS在项目中的使用时间线](progress/2026-10-09-davis-project-usage-timeline.md)：确认09-27引入官方RGB/实例掩码，09-28起四物体视频用于TrackCraft3R教师构造及接口审计，10-09复用于UniQuery4R可视化；DAVIS缺少所需3D轨迹真值，未启动这四视频上的关系解码器训练。

- 2026-10-09 [核查旧TrackCraft3R可视化的相机参数来源](progress/2026-10-09-trackcraft-camera-parameter-provenance.md)：本项目旧DAVIS流程由DA3估计深度、焦距及外参，再传给TrackCraft3R并用于三维回投影；发现适配脚本仅保存首帧内参供整个上下文共用，尚未量化这一近似的影响。无模型调用或原始文件修改。

- 2026-10-09 [完成UniQuery4R漂移汽车缓存诊断](progress/2026-10-09-uniquery4r-drift-output-diagnosis.md)：输入、帧对、UV与独立投影检查通过；drift-turn的t0二维定位已有32.7–48.3px中位偏差，三维投影175.7–310.3px。只估计两个焦距参数后留出误差降至约9.6–11.0px，支持相机与三维点不一致是重要来源；无GPU调用，原始输出未修改，未生成替代教师。

- 2026-10-09 [完成UniQuery4R四物体视频GPU推理](progress/2026-10-09-uniquery4r-gpu-visual-inference.md)：用户开启RTX PRO 6000后，四视频22帧/7帧共8次实际推理和可视化完成，峰值分配显存约10.97GiB，无OOM；原始坐标保留，未训练或数值评估跟踪质量。

- 2026-10-09 [完成UniQuery4R并行CPU准备检查](progress/2026-10-09-uniquery4r-parallel-cpu-checks.md)：本地及远端合成绘制测试通过；远端官方预处理、7/22帧第6帧查询接口、PyTorch2.8/CUDA12.8/sm_120编译支持及pip check通过。无模型加载、forward或GPU运行；检查证据已同步。

- 2026-10-09 [完成UniQuery4R下载、部署准备及本地清理](progress/2026-10-09-uniquery4r-inputs-and-cpu-preparation.md)：180个输入文件哈希一致；用户重新允许本地aria2下载后，6.6GB权重上传完成且远端官方SHA256通过。环境与CPU接口通过，GPU阶段严格加载通过。清理已验证本地权重、重复发布ZIP及临时媒体约6.93GB，保留真实实验资料。

- 2026-10-09 [核查 UniQuery4R 部署与计算资源](progress/2026-10-09-uniquery4r-deployment-preflight.md)：确认公开代码与约6.6GB权重，核查20帧504px稠密推理14.2–25.7GB官方显存表；建议24GB单卡起测22帧48点、48GB提供余量。发现默认源帧0及README/源码UV单位差异；未连接服务器、部署或运行模型。

- 2026-10-08 [核查参考点约束的方差与文献依据](progress/2026-10-08-reference-point-constraint-literature.md)：明确独立误差且中心方差更小时 query–中心关系方差降低；解释表面点身份错误；核查 SE3-Nets、VGPL 的中心/参考点共享运动结构及 SpatialTracker 距离约束消融，区分已见轨迹恢复与未来预测。未连接服务器或启动实验。

- 2026-10-08 [审查物体级参考轨迹的关系监督假设](progress/2026-10-08-object-anchor-relation-hypothesis.md)：推导更准确参考点降低关系误差的条件与共同误差抵消反例；核查 ObjectForesight/FoundationPose 的角色，区分未来预测与离线跟踪监督。保留中心加时变局部关系候选，未启动实验或选择新教师。

- 2026-10-08 [完成教师误差来源本地检查](progress/2026-10-08-teacher-local-error-source-audit.md)：执行 CPU 缓存诊断，确认 A/B 掩码缩放遗漏（已知48点原函数报0，缩放后48）；复算旧四视频的输入几何差异、原始首帧偏移、交错合并与未来掩码命中。服务器保持关闭，未训练、未改写原实验结果；原 A/B 数组缺失限制进一步归因。

- 2026-10-08 [复核教师误差归因与表面距离判断](progress/2026-10-08-teacher-error-attribution-review.md)：确认 Route A anchor 是人为三维平移后的投影误差，不能独立证明原始跟踪精度；整理固定输入的逐环节诊断，指出边向量误差与表面距离误差不存在必然大小关系。未连接服务器或执行新推理。

- 2026-09-28 [诊断 DAVIS 修正版视频首帧点位偏移](progress/2026-09-28-davis-frame0-overlay-offset-diagnosis.md)：48个输入 query 在源帧14车 mask 内且两次模型输入一致，但首帧三维输出投影后分别只有28/48（仅观测）与24/48（教师）留在车内；偏移不由视频编码引起，尚需区分轨迹模型首帧回归与 DA3 相机投影接口。

- 2026-09-28 [完成 DAVIS 四视频 TrackCraft3R 轨迹可视化](progress/2026-09-28-davis-track-visualizations.md)：仅使用既有轨迹、query 映射、RGB 和 mask 生成四段 22 帧教师视频、7 帧观测视频及 t0 对比图；输出保持 854x480，编码帧率仅用于播放，原质量闸门仍为 `0/4 PASS`。

- 2026-09-28 [完成 DAVIS 四视频联合解码器质量闸门](progress/2026-09-28-davis-joint-decoder-four-video-quality-gate.md)：四段新视频均完成 48-query、7 帧 observed-only、22 帧离线 TrackCraft3R 和 MotionForesight P0/H 审计；固定标签质量闸门为 `0/4 PASS`，按至少 `3/4` 通过规则未启动留一训练。关键产物已同步且 SHA-256 一致。

- 2026-09-28 [筛选物体视频并核对解码器训练监督](progress/2026-09-28-object-video-candidates-and-training-evidence.md)：追加七段明确物体目标的 DAVIS 视频及实例掩码，排除人/动物候选；此前联合关系解码器确实在四段旧视频的 TrackCraft3R 伪标签上训练，跨视频预设收益信号未通过；新视频尚无可靠三维轨迹标签。

- 2026-09-27 [取回实验原片并整理八段新视频候选](progress/2026-09-27-surface-relation-video-review-and-candidate-search.md)：从 AutoDL 取回并核对四段联合解码实验视频及两段旧片段，为四段生成 t0 周围预览；从官方 DAVIS 2017 提取八段独立序列、二维掩码并编码预览视频。候选用于人工筛选，尚无可靠三维轨迹监督，不计入既有实验。

- 2026-09-27 [完成联合时空关系解码器探索实验](progress/2026-09-27-surface-relation-joint-decoder-probe.md)：四段既有 Ego4D 视频整段留出、五种子、等参数逐点对照；关系版预设通过信号为 0/4 视频，`6516` 的结构误差改善未同时胜过逐点版，其余三视频较 baseline 退化；训练视频晚期槽位损失下降未转化为稳定跨视频收益。结果只针对冻结输出端具体实现及现有伪标签。

- 2026-09-27 [修订预申报书预期成果](progress/2026-09-27-preproposal-expected-outcomes-revision.md)：按模板要求将成果集中于论文投稿、项目结题和研究成果评选，投稿与获奖均为争取目标。

- 2026-09-27 [将预申报书研究计划调整为四阶段](progress/2026-09-27-preproposal-four-stage-schedule.md)：研究期限改为 2026 年 9 月至 2027 年 9 月；首阶段纳入本月已完成的接口、求解器与小样本实验，后续安排表面关系、时变预测及完整验证。

- 2026-09-27 [调整预申报书方法和对照的确定性](progress/2026-09-27-preproposal-method-flexibility.md)：将可微图一致性整合与具体对照方法改为候选，保留表面关系引导时变运动及协调轨迹生成的研究主张。

- 2026-09-27 [更正预申报书观测点图来源](progress/2026-09-27-preproposal-pointmap-source-correction.md)：复核 MotionForesight 的观测 RGB 到点图流程，删去将相机参数误写为研究输入的表述。

- 2026-09-27 [澄清预申报书观测几何来源](progress/2026-09-27-preproposal-geometry-source-clarification.md)：以 DA3 等模型为候选，说明观测 RGB 到三维点图的来源，撤回过度指定的相机参数与反投影流程。

- 2026-09-27 [修订预申报书工作构想](progress/2026-09-27-preproposal-workplan-revision.md)：按摘要与研究目标明确观测表面距离、时变三维相对运动和图一致性求解的分工，并补充训练与对照安排。

- 2026-09-27 [修订预申报书研究目标](progress/2026-09-27-preproposal-objectives-revision.md)：聚焦目标查询点未来三维轨迹，删去关节运动与固定指标清单，扩写方法目标与关键难点。

- 2026-09-27 [修订预申报书摘要并澄清测试视频划分](progress/2026-09-27-preproposal-abstract-revision.md)：核对原论文结构与监督，收紧结构漂移成因表述；将“独立视频”明确为按原始视频划分的未见测试视频。

- 2026-09-27 [复核预申报书正文语言](progress/2026-09-27-preproposal-language-cleanup.md)：删去重复的防御性否定与修订过程叙述，按模板重新整理完整可填写文字。

- 2026-09-27 [重写预申报书表面关系方法方案](progress/2026-09-27-preproposal-method-redraft.md)：提出观测表面距离图、时变相对关系预测与可微一致性整合的三段式研究构想，明确不同于此前效果不佳的冻结输出端关系头，并按模板重写文字。

- 2026-09-27 [比较预申报主题并调整研究期限](progress/2026-09-27-preproposal-topic-selection.md)：对照多未来与表面关系结构漂移方向，选择问题更聚焦且已有接口实验的后者；拟定 2026 年 10 月至 2027 年 12 月五阶段计划，保留监督质量与现有关系头负结果的边界。

- 2026-09-27 [拟写研究生科研项目预申报书文字初稿](progress/2026-09-27-preproposal-text-draft.md)：按模板栏位整理研究目标、工作构想和分阶段计划；不将待验证方案或点间关系探索写成已取得的研究成果，未生成申报书文档。

当前讨论提炼：[思路与待解决问题（含 2026-09-07 修订）](ideas/2026-09-06-two-stage-two-critic-idea-synthesis.md)。当前优先关注开放世界三维轨迹的多模态预测；两阶段为候选方式，语义—近期动态分工降低优先级，每条轨迹的概率表达尚未选定。既有方法笔记继续作为非约束的历史方案。

- 2026-08-04 [审查 MotionForesight 类多未来方法](progress/2026-08-04-motionforesight-method-review.md)：检查“语义等价类 + 条件 Flow”任务、训练边界和多未来评测问题。
- 2026-08-04 [细化语义条件与点轨迹处理](progress/2026-08-04-motionforesight-points-5-7-solutions.md)：形成开放事件关系、语义条件训练、无序点集与伪轨迹处理候选方案。
- 2026-08-04 [复查 MotionForesight 修订](progress/2026-08-04-motionforesight-revision-recheck.md)：保留语义利用、零权重 token、兼容事件和固定分支边界问题。
- 2026-08-04 [整理组会方法构想](progress/2026-08-04-motionforesight-method-discussion-review.md)：将开放事件图、语义条件 Flow 和多未来评测整理为阶段性讨论版本。
- 2026-08-04 [澄清开放事件词表](progress/2026-08-04-event-vocabulary-problem-explanation.md)：固定字段结构，保留开放事件内容，区分等价、蕴含、兼容和互斥关系。
- 2026-08-04 [审查 VLM 联合枚举未来](progress/2026-08-04-joint-vlm-future-enumeration.md)：确认同一回答中的候选频数不能解释为概率。
- 2026-08-04 [设计批量事件假设协议](progress/2026-08-04-batched-event-hypothesis-generation.md)：提出多次独立调用、每次少量候选的生成方案。
- 2026-08-04 [端到端审查语义条件 Flow](progress/2026-08-04-motionforesight-end-to-end-method-audit.md)：统一任务、候选语义、轨迹监督、条件 Flow 和多样本推理边界。
- 2026-08-04 [推演非文本未来潜变量](progress/2026-08-04-nontext-future-latent-redesign.md)：将文本事件替换为从未来视频学习的连续交互潜变量候选。
- 2026-08-05 [筛选相关论文并生成方法图](progress/2026-08-05-motion-forecast-literature-method-figure.md)：整理 MotionForesight、Generative Point Tracking and Forecasting、MYRIAD，并生成无显式语义的方法总览图。
- 2026-08-05 [澄清未来样本数量](progress/2026-08-05-variable-future-sample-count.md)：确认样本数是可变推理预算 `S`，固定六路仅为展示示例。
- 2026-08-05 [完成运动预测论文介绍页](progress/2026-08-05-motion-forecast-paper-slide-images.md)：完成两篇论文的 10 张介绍页和发言稿。
- 2026-09-02 [复盘项目现状与 FlowForesight 构想](progress/2026-09-02-project-status-flowforesight-review.md)：确认当前尚处于方法设计和汇报材料阶段。
- 2026-09-03 [固定视频预测范围并归档 IMU 工作](progress/2026-09-03-motionforesight-scope-reset.md)：将当前范围固定为视频输入的指定物体未来三维轨迹分布预测，归档旧 IMU 工作，并记录连续概率和语义—物理分层两个核心创新目标。
- 2026-09-04 [审查 MotionMap 与预训练动作先验重设计](progress/2026-09-04-motionmap-foundation-prior-review.md)：将旧组会方法及数字降为非约束草案，区分显式模式概率、随机条件利用与外部先验扩展，提出连续运动潜空间概率场候选。
- 2026-09-04 [细化伪分布监督、随机条件与对抗训练](progress/2026-09-04-distribution-target-adversarial-training.md)：说明怎样构造 MotionMap 式软分布目标，分离宏观模式与模式内随机性，并界定判别器、多样性损失和 IS 类指标的作用。
- 2026-09-06 [提炼两阶段、两评判器的当前思路](progress/2026-09-06-two-stage-two-critic-idea-synthesis.md)：保留文本粗模式与模式内三维轨迹的分工，澄清两级合理性筛选与全局轨迹密度的差别，整理监督、语义等价、模式空间及评估问题。
- 2026-09-07 [调整多模态三维预测的研究优先级](progress/2026-09-07-multimodal-forecasting-priority-clarification.md)：将多模态未来预测置于首位，降低语义分工优先级；撤回助手对全局轨迹密度需求的过度解释，讨论监督、两阶段价值、概率事件与评估问题。
- 2026-09-07 [澄清分层概率与评判边界](progress/2026-09-07-hierarchical-modes-and-critic-boundary.md)：记录模式选择与轨迹实现的两阶段动机，区分已观察状态、未知意图和低频模式；确认两种轨迹数值表达均可讨论，并纠正 Stochastic Conditioning 的损失与评估指标混淆。
- 2026-09-07 [讨论模式偏离与模式图覆盖](progress/2026-09-07-mode-drift-and-map-peak-coverage.md)：保留合理跨模式候选的讨论空间，区分候选质量、源模式覆盖和概率修正；核对 MotionMap 目标构造与峰值提取的不同作用。
- 2026-09-07 [讨论语义模式图的构造](progress/2026-09-07-semantic-mode-map-construction.md)：区分共享模式空间与条件响应图，梳理 MotionMap 的多未来目标来源；提出未来事件表示及预训练候选弱监督的待验证方向。
- 2026-09-07 [核查事件语义空间与未来候选教师](progress/2026-09-07-event-semantics-and-future-teacher-review.md)：核对双向蕴含、AMR、TMR 及 GePSAn、AntGPT、ViNa；确认存在多未来文字提议先例，适用于当前模式图的额外正标签可靠性仍待验证。
- 2026-09-07 [调研语义距离后续方法并复核教师](progress/2026-09-07-semantic-distance-followups-and-teacher-audit.md)：比较 KLE、SD、SNNE 与 C-STS 后续 CASE、CCL 等工作；核对 AntGPT 的非厨房场景及固定词表、ViNa 的文字依赖和多候选生成，教师继续保持未选定。
- 2026-09-07 [放宽教师输入并复核交互范围](progress/2026-09-07-flexible-teacher-inputs-and-interaction-scope.md)：聚焦人—物交互中的物体轨迹，教师输入与数据集联合选择；ViNa 移出可直接使用候选，核查历史叙述路线及 EgoPlan 的代码、权重入口与目标条件边界。
- 2026-09-07 [调研多分支文本教师与单真实未来轨迹监督](progress/2026-09-07-multibranch-teachers-and-single-future-supervision.md)：核查 VisualCOMET、KM-BART、DIVE、VANS 等自由文本多结果路线及公开资源；回查语义匹配配对训练，明确额外文字分支尚缺当前场景的几何监督，补充 MolmoMotion 数据与已发布自回归模型。
- 2026-09-08 [核查教师输出质量及三篇新方法](progress/2026-09-08-stage1-teacher-quality-and-new-methods.md)：检查 NEP、Video-CoE、FUTURE-L1 及旧模型的公开输出、评价条件与资源；区分已知结尾选择题成绩、开放问答评分和多分支伪标签质量，记录图例疑点并调整候选优先级，教师继续保持未选定。
- 2026-09-08 [复核四篇工作的权重与推理入口](progress/2026-09-08-teacher-checkpoints-and-inference-readiness.md)：确认 VANS 与 FUTURE-L1 均已公开训练后权重和推理实现，更正此前漏查 FUTURE-L1 权重的结论；核对文本接入所需适配，NEP 与 Video-CoE 暂未确认现成检查点，尚未运行任何教师。
- 2026-09-08 [讨论多未来文本与潜状态接口](progress/2026-09-08-multinarration-and-latent-future-interface.md)：核对 VANS 随机文本解码、FUTURE-L1 完整推演采样及潜状态导出；讨论离线候选生成与直接潜状态条件，保持模式覆盖、目标轨迹信息和教师用途为待验证问题。
- 2026-09-08 [回查事件多样性教师生成](progress/2026-09-08-event-diverse-teacher-generation-recap.md)：按 narration 基准澄清潜状态的相对价值，回查联合枚举、互补生成、语义比较和多样性训练；明确尚缺针对教师生成过程的事件级鼓励机制，候选覆盖与模式权重保持分别讨论。
- 2026-09-08 [讨论冻结 L1 的语义前瞻解码](progress/2026-09-08-frozen-l1-semantic-lookahead-decoding.md)：记录用户优先考虑 L1 且少调或不调权重的偏好；核查 SemDiD、NLI 重采样及分支搜索，提出事件级前瞻与分支反馈候选，保留潜状态配对和推理适配问题，尚未实现或测试。
- 2026-09-08 [审阅 SemDiD 与 L1 主动潜状态调整](progress/2026-09-08-l1-semantic-latent-steering-review.md)：读取共享讨论，核查冻结权重下文本目标反向调整潜状态的依据；区分文字可控性、未来视觉信息、候选发现与协同分支，保留两版为待验证候选，未实现或运行模型。
- 2026-09-08 [澄清 L1 联合多分支生成与分叉位置](progress/2026-09-08-l1-joint-branch-generation.md)：记录一次输入联合维护多个候选的目标；核对 SemDiD 的逐步搜索与 L1 连续状态更新，提出在对应潜状态形成前探索文字事件分叉的讨论方向，尚未确定或验证。
- 2026-09-08 [核查 L1 首个 reason 的实际内容](progress/2026-09-08-l1-first-reason-content-audit.md)：逐一检查图 15–18 及训练/提示模板；四个展示案例首段均以观测概括及活动背景为主，暂缓将首段作为默认未来事件分叉点，已向此前记录追加修正，未运行模型。
- 2026-09-09 [讨论 L1 潜状态分叉与事件前瞻反馈](progress/2026-09-09-l1-latent-branch-search-options.md)：核查 LatentTTS 论文、采样及搜索代码和 GTS；提出冻结 L1 的 latent 提议与事件级前瞻搜索候选，保留小型学习式采样器为备选，尚未验证视觉未来分支效果。
- 2026-09-09 [暂定随机扰动主动版并梳理方法缺口](progress/2026-09-09-active-l1-selection-and-method-gaps.md)：记录用户暂定采用 L1 latent 随机扰动与事件前瞻反馈；梳理模式图、阶段接口、额外轨迹监督、评判器、输出数值及数据评估六项未决内容，未新增架构或开展实验。
- 2026-09-09 [审阅主动版评价并核对 MotionMap 迁移](progress/2026-09-09-active-review-and-motionmap-transfer.md)：保留随机提议和事件前瞻，指出自动增噪及似然认证等推断过强；核实 MotionMap 以最大值合并高斯目标、codebook 平均与解码适配，明确可借用建图流程，表示与监督迁移仍待确定。
- 2026-09-09 [复核当前进展与代码测试就绪度](progress/2026-09-09-current-state-and-test-readiness-audit.md)：确认冻结 FUTURE-L1 主动 latent 分支仅为暂定候选，MotionMap 与第二阶段仍有关键缺口；本地尚无研究实现，并发现 Python/Conda 随机源、NVML/CUDA 与 Git 仓库状态问题。
- 2026-09-09 [验证 AutoDL 连接并确定首个 L1 实验](progress/2026-09-09-autodl-connectivity-and-l1-test-target.md)：SSH 登录、远程命令与退出均通过；记录无卡模式环境、依赖和网络状态，并将 FUTURE-L1 latent 扰动有效性确定为首个远程模型测试。
- 2026-09-09 [配置 AutoDL 网络加速与数据盘 Conda 环境](progress/2026-09-09-autodl-network-turbo-conda-environment.md)：验证 network_turbo 后 GitHub、Hugging Face 均返回 HTTP 200；在数据盘创建 FUTURE-L1 实验环境并绑定模型、数据、结果和缓存目录。
- 2026-09-09 [核查 MotionMap 表示与平均 codebook 的训练依据](progress/2026-09-09-motionmap-representation-and-codebook-evidence.md)：恢复目标图前的模式判别器位置；核实运动重建对表示的间接约束、当前观测条件及平均向量解码微调，区分训练依据与经验支持，未运行模型。
- 2026-09-09 [确认训练约束路径并回顾模式评判器](progress/2026-09-09-training-constraints-and-mode-critic-recap.md)：将表示与平均条件问题纳入训练及验证事项；回查模式评判器的输入、职责、接受度输出和先验来源，具体模型与监督方案仍未选定。
- 2026-09-09 [讨论以 L1 生成先验省去独立模式评判器](progress/2026-09-09-l1-prior-without-separate-mode-critic.md)：核查未来帧对齐与生成训练依据；支持将无独立模式评判器作为待验证简化方案，区分生成先验与逐候选认证，未改变第二阶段评判器或运行远程测试。
- 2026-09-10 [讨论离线 L1 教师与模式图学生](progress/2026-09-10-offline-l1-teacher-and-mode-map-student.md)：核查 MotionMap 用观测预测未知未来条件的作用；明确教师预处理、学生训练与推理分工，区分离线蒸馏收益和二维 heatmap 的独有必要性，未测量加速或选定最终表示。
- 2026-09-10 [复核 L1 扰动实验并确认无图主线](progress/2026-09-10-l1-perturbation-results-audit.md)：核对115个源文件、33次运行、保存latent与输入/后续抽帧；30次非零latent改变、6次完整答案改变及1次截断，尚未展示清晰的新增事件分支；指出单baseline不足以验证重复一致性，模式图转为备选，未重跑模型。
- 2026-09-10 [汇总无模式图主线与剩余方法问题](progress/2026-09-10-direct-branch-method-status.md)：记录直接事件分支为主线、模式图为备选；区分已选框架、待验证扰动假设与分支接口、第二阶段监督、教师部署及轨迹评分等未决项，同步路线说明，未重跑实验。
- 2026-09-10 [复核新视频与 L1 反馈搜索实验](progress/2026-09-10-l1-complete-search-results-audit.md)：核查153个源文件、66个候选与693对比较；三组重复基线一致且输出完整，最终不同答案数为3/1/1；发现词集合比较、重复填满槽位、质量分恒零及缺反馈零扰动对照，尚未达到稳定多事件候选预期；未修改实验代码、未重跑模型。
- 2026-09-10 [确认 L1 latent 接口并核查 TrajLoom 适配](progress/2026-09-10-latent-interface-trajloom-feasibility.md)：记录在线无图主线、离线模式图备选及 Ego4D/SSv2 数据候选；核实 TrajLoom 为二维轨迹且训练代码待发布，解释离线无图学生、MotionForesight 目标坐标、帧数与三维伪标签有效性，保留三维适配和轨迹数值为待讨论项，未运行模型。
- 2026-09-11 [核验服务器迁移与 Future-L1 round3 状态](progress/2026-09-11-remote-migration-and-round3-status-audit.md)：通过新端点 `connect.bjb1.seetacloud.com:19874` 核对 GPU、环境、进程、代码和结果；确认 event-aware v2 已完成 3×200 候选、高 sigma stress 已完成 3×30 候选，哈希/有限性/重复 baseline 检查通过；按第一阶段范围，主要缺口是高层事件独立性和跨视频稳定性。
- 2026-09-11 [修正 Future-L1 round3 实验范围](progress/2026-09-11-round3-scope-correction.md)：依据项目对话确认本轮只评估高层事件模式候选生成与反馈筛选；三维 decoder、轨迹指标和物理合理性属于后续阶段，不作为本轮门槛；当前候选多样性仍未达到预期。
- 2026-09-11 [比较前瞻语义模式序列与 L1+SemDiD](progress/2026-09-11-prospective-semantic-mode-sequence.md)：读取用户提供的分享对话并结合 round3 结果，确认“显式未来语义分布 + 顺序模式生成 + SemDiD 关系约束”在第一阶段目标上理论对齐性更强；语义表示、单真值监督、模式排序和概率校准仍待验证。
- 2026-09-11 [重评第一阶段组合架构](progress/2026-09-11-revised-first-stage-architecture.md)：明确分享方案是 MotionForesight 前瞻化、ObjectForesight 多结果生成和 ModeSeq 顺序互补解码的组合，不是单独替换为顺序概率公式；该结构比随机 L1 latent 分支更贴合模式生成目标，但语义目标、实际 ModeSeq 接入、无序模式训练和概率解释仍需验证。
- 2026-09-11 [评估第一阶段组合方法的实现定位](progress/2026-09-11-composite-stage1-method-assessment.md)：根据补充说明修正架构表述；新主线是前瞻语义适配、条件多结果生成和生成期互补解码的组合，L1+SemDiD 随机分支保留为基线；确认该组合在第一阶段目标上理论对齐性更强，但模式覆盖、单真值监督、概率含义、无序集合训练和模块接口仍待验证。
- 2026-09-13 [提炼第一阶段 VLM 先验讨论并核查方法依据](progress/2026-09-13-vlm-prior-stage1-discussion-audit.md)：读取补充分享，核查前瞻适配、单 GT 的 ModeSeq、GePSAn 文本先验迁移及相关 VLM 预测工作；明确组合仍属工作假设，纠正通用评分器、SemDiD 等同 NLI、完整有序多 GT 前提等过度推断；未改实验代码、未训练或重跑模型。
- 2026-09-14 [分析前瞻 VLM、token sampling 与 ModeSeq 的未决问题](progress/2026-09-14-vlm-sampling-modeseq-open-questions.md)：记录当前未定倾向，核查已有多样性机制；明确模式定义、记忆与采样输出一致性、语义匹配后的语言训练及候选合理性问题，区分 VLM 能力依据与组合效果验证；未运行模型。
- 2026-09-14 [核查 ModeSeq 机制继承与第一阶段文本接口](progress/2026-09-14-modeseq-inheritance-and-text-interface.md)：修正将特定记忆风险及已有单 GT 处理列为必要障碍的表述；核实 MolmoMotion 自然语言动作接口与 Ego4D 两类标注，确认 narration 可表达动作顺序，具体格式仍未定案；未运行模型。
- 2026-09-14 [审核双向蕴含、语义 EMTA 与 teacher forcing](progress/2026-09-14-nli-emta-teacher-forcing-audit.md)：确认双向蕴含有原文依据且旧 round3 已采用同一比较公式；说明低分最近分配属于待验证适配，核查条件文本损失可训练到模式模块的条件，并澄清时间戳支持片段监督但不等于精确动作边界；未运行模型。
- 2026-09-14 [检查第一阶段设计状态并比较先验模型](progress/2026-09-14-stage1-readiness-and-backbone-selection.md)：确认可进入第二阶段讨论，保留模式条件接入与文本时间接口为具体设计项；核查 Molmo2-8B、Qwen3-VL-8B-Instruct、Qwen3.5-9B，建议 Molmo2 为首个候选，尚未定案或运行模型。
- 2026-09-14 [实测 MolmoMotion 的动作文本类型](progress/2026-09-14-molmomotion-text-type-probe.md)：在三组官方完整输入上完成 5 类文本共 15 次推理及一次确定性复跑；自由句子与多步骤计划均能产出有效轨迹，但冲突文本的模式控制不稳定，另下载并核验 3 个 Ego4D FHO 短片。
- 2026-09-15 [用 Tracker3D 真实输入复核 MolmoMotion 文本控制](progress/2026-09-15-molmomotion-tracker3d-e2e.md)：从 Ego4D sewing clip 构造剪刀查询像素与三维历史并完成四类文本推理；输入 token 不同，但原始生成文本和轨迹完全相同且预测段静止，技术链路与复跑通过，文本控制目标未通过。
- 2026-09-15 [校正时间协议并完成 MolmoMotion 两视频复测](progress/2026-09-15-molmomotion-aligned-multivideo-retest.md)：以 15 fps 的 H3 历史、两个包含实际物体运动的 Ego4D 样本和离线 TrackCraft 未来伪标签完成 12 次预测；greedy 结果随文本改变且 6/6 精确复现，但 observed-action 未稳定提高精度，多步骤与反事实指令主要改变幅度，可靠动作遵循仍未通过。
- 2026-09-15 [澄清隐藏模式状态并回顾第二阶段](progress/2026-09-15-hidden-mode-state-and-stage2-recap.md)：确认 VLM 分支隐藏状态可承担模式表示、冻结先验与新增 LoRA 已有方向，采样和 teacher forcing 可共享条件；记录目标与时间沿用原则、不固定动作数，回查三方法分工和 MolmoMotion 最新复测，未运行新实验。
- 2026-09-16 [审核 MotionForesight 文本条件与生成式多未来改造](progress/2026-09-16-motionforesight-text-conditioned-generative-redesign.md)：核实 Wan 原生文本通路和当前 null context实现，建议以 cross-attention LoRA为主、文本 mask调制为消融；区分 mask与track latent，提出在 residual-track latent中训练条件 rectified flow，并保留生成空间、flow状态接入和概率解释为待验证问题。
- 2026-09-16 [复核第二阶段多样性设计](progress/2026-09-16-stage2-diversity-design-audit.md)：确认 MotionForesight 主干与 Wan 文本 cross-attention 方案的适用边界；否定仅对 mask latent 加噪作为完整多样性方法，提出以确定性文本条件预测为锚、在 future residual-track latent 上训练条件 flow 的候选，并记录单真实未来监督、轨迹 latent 可逆性、flow 状态接入和概率校准等未决问题；未运行新训练。
- 2026-09-16 [确定第一版完整方法流程](progress/2026-09-16-deterministic-stage2-complete-method.md)：将第二阶段多样性延期，冻结第一版为“VLM 多未来 narration + 文本条件 MotionForesight 确定性稠密三维轨迹”；整理研究动机、两阶段输入输出、narration 监督、ModeSeq/双向蕴含约束、Wan T5 cross-attention LoRA、三维伪标签和评估边界；明确该版本暂不声称连续概率分布或同一 narration 内多轨迹；未运行新训练。
- 2026-09-16 [完成 TrajLoom 文本条件矩阵实验](progress/2026-09-16-trajloom-text-conditioning.md)：在两个 RoboTAP 视频上完成 12 次推理与分析；固定噪声下文本改变会改变轨迹，同文本跨种子也产生变化，但文本效应跨视频不稳定，尚不能支持事件级多模态或概率校准结论。
- 2026-09-16 [生成 TrajLoom 文本条件轨迹可视化](progress/2026-09-16-trajloom-text-visualization.md)：为两个视频生成无文本、一致、静止和冲突文本四条件的未来轨迹叠加图，供定性检查使用。
- 2026-09-16 [复核 TrajLoom 文本条件运动方向](progress/2026-09-16-trajloom-directional-analysis.md)：定量计算三种文本条件的位移方向和夹角，确认冲突文本未造成方向反转，视觉运动先验仍占主导。
- 2026-09-16 [评估视觉模态削弱与文本条件](progress/2026-09-16-visual-masking-text-conditioning.md)：基于 TrajLoom 两视频 probe 评估第二阶段随机 mask 思路；建议采用保留目标几何与近期运动的视觉模态 dropout，区分文本敏感性提升和三维精度损失，并给出完整/部分 mask/冲突文本实验矩阵。
- 2026-09-16 [澄清 ModeSeq 的文本迁移机制](progress/2026-09-16-modeseq-text-transfer-clarification.md)：核实原 ModeSeq 前序分支传递连续 mode embedding，并固定双向蕴含 EMTA、无匹配回退、选中分支 teacher forcing 与其余分支置信度标签；项目迁移接口已于 9 月 18 日改为实际 narration token memory。
- 2026-09-18 [审核当前方法的学术合理性并设计快速实验](progress/2026-09-18-current-method-academic-audit-and-fast-tests.md)：明确当前版本是语义多分支到确定性三维轨迹的验证系统，尚不满足连续概率主张；识别置信度语义、mode embedding 与随机文本、NLI 校准、文本可辨识性、阶段差异和数据对齐风险，并给出由数据/NLI 审计到小样本消融的验证顺序。
- 2026-09-18 [澄清 token 候选记忆、模态平衡与级联训练](progress/2026-09-18-token-memory-modality-balancing-and-cascade-training.md)：将当前目标改述为固定预算下的长尾模式覆盖；采用实际 narration token/hidden states 作为后续分支记忆；调研 ModDrop、Gradient Blending 与 OGM-GE；明确冻结第一阶段后只用与真实未来匹配的预测 narration 配对训练第二阶段。
- 2026-09-18 [完成 TrajLoom 事件文本条件复测](progress/2026-09-18-trajloom-event-text-probe.md)：在两个视频上完成正确/相反事件文本和三个配对种子的 12 次推理；文本对目标轨迹有稳定但弱于随机噪声的影响，尚未控制宏观事件，TrajLoom 保留为二维基线或结构参考。
- 2026-09-20 [独立审核当前两阶段方法](progress/2026-09-20-independent-method-review.md)：基于正式方法记录、ModeSeq 原机制和实际文本条件实验，重新评估 narration 模式假设、ModeSeq 迁移、双向蕴含、单 GT 监督、文本可辨识性、级联协议和数据对齐，并给出分组快速验证方案。
- 2026-09-18 [审查 MotionForesight 官方 Demo 的失效现象](progress/2026-09-18-motionforesight-demo-failure-analysis.md)：逐项检查 20 个 SSv2 与 20 个 OOD 对比样本；主要问题集中在宏观结果与幅度、刚体/关节约束、旋转接触动力学和单未来不确定性表达，MolmoMotion 无语言版本还频繁退化为近静止预测。
- 2026-09-16 [规划当前两阶段方法的低成本可行性测试](progress/2026-09-16-two-stage-feasibility-tests.md)：按数据接口、第一阶段语义候选、第二阶段文本通路和端到端接口分解四层设计测试；明确正确/空/置换文本对照与 GT/top-1/oracle/空文本诊断，并保持第一阶段独立评价不使用轨迹指标。
- 2026-09-21 [完成 MolmoMotion 原始视频文本遵从复测](progress/2026-09-21-molmomotion-raw-video-gpu-probe.md)：迁移并校验官方 `model.pt`、DA3 与 TrackCraft3R 链路，在饮料瓶 8 点历史上完成四条指令推理与可视化；输入 token 不同，但四个 `(8,30,3)` 输出逐字节相同且预测期静止，当前公开 AR checkpoint 未通过该样本的文本遵从检查。
- 2026-09-21 [诊断 MolmoMotion 饮料瓶静止轨迹](progress/2026-09-21-molmomotion-static-diagnosis.md)：在当前服务器复跑旧剪刀/浇水壶输入并恢复非静止文本敏感输出；修正饮料瓶7.5 fps协议错误，测试15 fps、短动作句、动作后移窗口与交叉提示词后仍近静止，排除通用实现故障，定位为公开AR检查点在当前视觉/三维输入上的零运动退化。
- 2026-09-21 [审查点间表面关系预测方案](progress/2026-09-21-surface-relation-method-review.md)：核对用户对话记录与 MotionForesight、TrackCraft3R、TrajLoom、MolmoMotion 原始资料；确认关系建模方向可验证，同时指出 query--latent 对齐、关系头监督冗余、差分噪声、固定表面图和结构指标等问题，并给出先做 decoded-coordinate relation loss 的最小验证版本。
- 2026-09-21 [二次复审点间表面关系修正版](progress/2026-09-21-surface-relation-revision-review.md)：确认修正版已采用 query readout、未来关系状态主监督和分阶段消融；指出 readout 后缺少返回 frozen decoder 或 point output 的接口，并补充边关系全局可实现性、监督图与消息图区分、有效性加权、query-set 稳定性和旋转等变性问题；建议按直接关系损失、稀疏 query residual refinement、latent writeback 的顺序验证。
- 2026-09-21 [三次复审关系约束图投影方案](progress/2026-09-21-surface-relation-graph-projection-review.md)：确认 graph projection 补全了 relation 到最终坐标的接口，核验闭式方程、反对称构造和零初始化；指出投影严格保持连通分量质心、锚定项收缩结构修正、学习权重塌缩、query density、梯度边界和时间抖动等限制，并给出冻结基线的最小实现及必要消融。
- 2026-09-21 [复审固定规范图关系整合方案](progress/2026-09-21-surface-relation-gauge-fixed-review.md)：确认移除 mu I 并采用零质心约束可消除正确结构修正的系统性收缩；指出单一 gauge 仅适用于有效图连通、保持的是 query-set mean、无正则整合可能放大伪标签噪声、固定边权不解决错误拓扑，以及 Relation Head 应显式接收 baseline edge relation。
- 2026-09-21 [复审修正版点间表面关系方案](progress/2026-09-21-surface-relation-method-rereview.md)：确认修正版已引入 query--latent readout、未来相对位置主监督和软边门控；仍发现 graph-to-decoder 接口、边关系可积性、局部图的全局漂移、gate 退化和 query 图采样等未闭合问题，建议先用冻结 decoder 的残差坐标修正验证关系收益。
- 2026-09-22 [复审组件感知固定规范关系整合方案](progress/2026-09-22-surface-relation-latest-review.md)：确认组件级 gauge、面积加权、显式 baseline relation、robust integration 与时间损失等修正基本成立；指出 IRLS 更新必须保留约束、面积权重与边置信度需分工、断连 patch 的能力边界、固定 K 不保证 query-subset invariance，以及鲁棒损失不能解决系统性关系偏差。
- 2026-09-22 [完成点间关系整合方案合格性复核](progress/2026-09-22-surface-relation-final-qualification-review.md)：判定组件级 gauge-fixed L2 核心方案可进入原型；进一步区分 raw relation proposal 与最终投影关系的时间损失，指出 Huber/IRLS 仍需带约束求解、弱连接谱稳定性不能由鲁棒损失替代、面积 gauge 只是规范选择，并补充 observed-only graph builder 与训练有效性 mask 的推理边界。
- 2026-09-22 [审计点间表面关系方案从第一版到当前版的修改](progress/2026-09-22-surface-relation-v1-to-current-audit.md)：还原 latent Surface-Relation Adapter 到 query-level relation correction 与 graph integration 的演化；判定 query readout、输出接口、correction-space gauge、组件零空间和显式 baseline relation 为必要修正，将面积加权 gauge、Huber/IRLS、anchor graph、global motion head、learned gate 和额外物理模块移出第一版，并给出固定观测图加组件级 uniform-gauge L2 integration 的精简方案。
- 2026-09-22 [规划点间表面关系方案的小型可行性实验](progress/2026-09-22-surface-relation-feasibility-experiments.md)：设计 E0 求解器单测、E1 现有真实轨迹 oracle/噪声实验、E2 query graph 比较、E3 Relation Head 可学习性 probe、E4 小规模训练消融和可选 E5 IRLS 决策；明确 E0--E2 可立即利用本地 TrackCraft3R/MolmoMotion 数值轨迹开展，并设置逐级停止条件。
- 2026-09-22 [完成点间表面关系 E0--E1 可行性实验](progress/2026-09-22-surface-relation-e0-e1-results.md)：实现组件感知 uniform-gauge L2 求解器并在本地与 AutoDL CPU 复跑；E0 数值、置换和梯度检查全部通过，E1 在 3 个底层视频上 oracle 恢复 100% RVE、5% Gaussian 边噪声下恢复约 75.1%，同时确认异常边和断连会显著降低恢复率。
- 2026-09-22 [审查官方 ViPE+AllTracker 替换可行性](progress/2026-09-22-vipe-alltracker-compatibility-audit.md)：确认官方管线替换是排除坐标、尺度和跟踪分布差异的正确 A/B 设计；本轮因 GPU 服务器 SSH 在密钥交换阶段被远端关闭，且本地没有官方 ViPE/AllTracker 代码、权重和转换器，尚未完成替换推理，不能宣称第三点已排除。
- 2026-09-22 [复核点间关系 E0--E1 并判断后续实验门槛](progress/2026-09-22-surface-relation-e0-e1-readiness-review.md)：固定种子复跑与原始本地 JSON 逐字节一致，跨本地与 AutoDL 的 E1 数值一致；判定 E0 充分通过、E1 可支持进入 E2-lite，同时发现饮料瓶组噪声恢复不足、样本仅 3 个源视频、时间对齐未验证及 structural-drift AUC 重复定义等限制，暂缓 E3/E4 与 IRLS 主线化。
- 2026-09-22 [复核 E2-lite 并判断后续阶段](progress/2026-09-22-surface-relation-e2-lite-readiness-review.md)：本地脚本重跑与原始结果逐字节一致，AutoDL 数值差异仅为浮点误差；判定 2D symmetric kNN `k=3` 可作为原型默认图并进入正式 E2 数据扩展，但 2D/3D 差距很小、三视频宏平均不足以证明最优性，E3 需等待 MotionForesight `P0,H`，E4 与 IRLS 继续暂缓。
- 2026-09-22 [明确正式 E2 的研究决策与代码执行分工](progress/2026-09-22-surface-relation-formal-e2-responsibility-split.md)：确认正式 E2 的核心图方案已固定，研究负责人需要确认独立视频单位、query 采样、运动类别、30→9 时间映射、主判据和纳入规则；我负责数据与方法学审查及阶段放行，代码 AI 负责预检器、manifest、时间映射报告器、正式 runner 和复现检查。
- 2026-09-22 [冻结正式 E2 的数据源、query 采样与时间协议](progress/2026-09-22-surface-relation-formal-e2-protocol-decision.md)：决定主实验只使用 Ego4D，目标 6 个、最低 5 个独立视频；每视频先生成 64 个 mask 内 deterministic FPS query，主结果使用 48 点并分析 32/64 密度；固定 15 fps、2.0 秒窗口、3 帧历史和 9 步未来时间网格，明确有效性、图选择、统计聚合和进入 E3 的预注册门槛。
- 2026-09-22 [完成点间表面关系 E2-lite 图比较](progress/2026-09-22-surface-relation-e2-lite-results.md)：比较 2D、observed 3D 与 hybrid-lite 的 `k=2/3/4`，对每个噪声条件运行 20 个种子并改用逐 horizon RVE 的时域梯形 AUC；hybrid 在 2/5 个 `k=3` 图配置断连且无稳定收益，暂定原型采用全部连通、无桥边的 2D symmetric kNN `k=3`，同时保留饮料瓶强噪声失败和正式 E2 数据不足的限制。
- 2026-09-22 [审计正式 E2 数据扩展条件](progress/2026-09-22-surface-relation-formal-e2-data-audit.md)：核对新 AutoDL 数据盘与本地样本，确认仍只有 3 个源视频和 8 点 query；实现正式 E2 输入审计，明确 5--10 个独立视频、32--64 query、时间元数据和投影一致性要求，当前不把现有 prompt/窗口变体扩写成正式独立样本。
- 2026-09-22 [完成正式 E2 数据准备与 runner 代码](progress/2026-09-22-surface-relation-formal-e2-code-preparation.md)：实现 15 fps 时间网格、真实时间戳匹配、polygon 腐蚀、32/48/64 deterministic FPS query、2D/3D k=3 正式 runner、可变 query smoke test 和 6 视频 manifest 模板；当前旧 `.npz` 因缺少真实时间字段被正式 runner 拒绝，等待新增 Ego4D 视频与 mask。
- 2026-09-22 [完成 MolmoMotion 官方 ViPE+AllTracker A/B 实验](progress/2026-09-22-molmomotion-vipe-alltracker-test.md)：固定饮料瓶视频、8 个查询点和四条指令，完成官方 ViPE、AllTracker、3D lift 与 MolmoMotion 推理；四条输出仍完全相同且预测段恒定，路径约 `0.000832 m`，因此 DA3+TrackCraft3R 的坐标/尺度/跟踪差异不能单独解释近静止退化。
- 2026-09-22 [归档 MolmoMotion 官方 ViPE+AllTracker 实验](progress/2026-09-22-molmomotion-vipe-alltracker-archive.md)：将结果、日志、可视化、报告和运行器复制到 `archive/experiments/molmomotion/MolmoMotion-vipe-alltracker-2026-09-22/`，原始 `experiences_results` 目录继续保留。
- 2026-09-22 [完成正式 E2 数据与模型预检](progress/2026-09-22-surface-relation-formal-e2-data-and-model-preflight.md)：清理中断的 Ego4D 下载并保留 metadata，下载 2 个可复用 clip 与 4 个按固定 metadata 顺序选择的候选 clip，完成 16/48/64 sampler smoke；官方 MolmoMotion-4B-H3-F30 processor 明确拒绝超过 8 点的 `points_3d_history`，因此正式 E2 的 48/64 点模型接口仍被阻塞，未用 8 点替代协议。
- 2026-09-22 [更新正式 E2-G/E2-M 执行边界](progress/2026-09-22-surface-relation-formal-e2-g-boundary-update.md)：保留 32/48/64 query 协议；将密集图与积分器辅助实验独立标记为 E2-G，并在 runner 和 manifest 中显式区分受控扰动、TrackCraft3R pseudo-GT 与 MolmoMotion prediction；E2-M 继续等待可变 query 接口。
- 2026-09-22 [确定正式 E2 模型接口闸门与执行边界](progress/2026-09-22-surface-relation-formal-e2-model-gate-decision.md)：保持 32/48/64 dense-query 协议，不降级为 8 点；将正式 E2 分为模型无关的 E2-G 图积分验证与需可变 query 模型的 E2-M 接口验证，明确 `future_3d` 来源命名和代码 AI 可继续工作的范围。
- 2026-09-23 [复核正式 E2-G 结果并调整图选择流程](progress/2026-09-23-surface-relation-formal-e2-g-review-and-decision.md)：确认 E2-G 数据与工程闸门通过，但 2D `k=3` 的噪声 RVE AUC 比 3D 高约 14.56%，而 3D 在 4/5 视频断连；暂停“2D 失败即切换 3D”规则，保留 2D/3D 为候选与对照，新增一次受限的 3D 局部图加 2D 组件连接边实验，并补做 48→64 query-density 检查。
- 2026-09-23 [复核 E2-G.1 并确定图候选层级](progress/2026-09-23-surface-relation-formal-e2-g1-review-and-candidate-hierarchy.md)：确认修复 3D 图在 shared-union RVE 上通常优于 2D，但依赖 4 条长 bridge 且 component-centered error 上升；保留 2D `k=3` 为主工程候选、修复 3D 为后续模型侧次级候选、原始 3D 为诊断对照，结束当前无模型图结构搜索并转向模型接口与 `P0,H` 准备。
- 2026-09-23 [复核 E2-G.1 轻量诊断并结束无模型图阶段](progress/2026-09-23-surface-relation-formal-e2-g1-diagnostic-validation.md)：确认真实目标 bbox、连接边尺度和 20 种子异常诊断，当前环境确定性复跑与原 JSON 逐字节一致；保留 bridge 特异性缺少局部边匹配对照的解释边界，并决定下一阶段优先检查 MotionForesight observed-only `P0,H` 接口，E2-M 不作为 E3 的强制前置。
- 2026-09-23 [续接正式 E2 服务器环境准备](progress/2026-09-23-surface-relation-formal-e2-server-preparation.md)：重新核验 V100S AutoDL 服务器，安装并通过 `DepthAnything3` API 导入检查，确认 TrackCraft3R 仓库与 checkpoint 仍在；DA3 checkpoint、polygon、三帧 pointmap 和真实时间对齐 pseudo-GT 尚未完成，正式 E2-G 继续等待数据审计。
- 2026-09-23 [完成正式 E2-G 数据构建与辅助实验](progress/2026-09-23-surface-relation-formal-e2-g-execution.md)：在 V100S Conda 环境用精确 query-axis 分块解决 TrackCraft3R SDPA OOM，生成 5 个 Ego4D 视频的 64-query、3 帧历史和 9 步 pseudo-GT bundle并通过正式审计；2D `k=3` 在 5/5 视频连通且无桥边，但共享评估边集下主指标比 3D 高约 14.6%，默认保留闸门未通过，3D 又在 4/5 视频断连，因此图选择保持未决，E2-M/E3 继续阻塞。
- 2026-09-23 [完成正式 E2-G.1 连通修复图实验](progress/2026-09-23-surface-relation-formal-e2-g1-connectivity-repair.md)：保留 3D `k=3` 局部边，并按确定性 Kruskal 规则从 2D `k=3` 加入跨分量边；4/5 视频各加入 1 条 bridge 后全部连通。修复图的 graph-native RVE 接近原始 3D，但 shared-union RVE 回升约 4.8%--10.9%，说明原始 3D 优势部分来自断连后减少跨分量约束；48→64 未见系统性恶化，正式默认图继续不冻结。
- 2026-09-23 [完成 E2-G.1 连接边归一化与异常敏感性诊断](progress/2026-09-23-surface-relation-formal-e2-g1-review-and-candidate-hierarchy.md)：修正诊断脚本，改从正式 query manifest 读取目标 `bbox_xyxy`；4 条连接边归一化 2D 长度为 `0.0831--0.1143`，3D 长度约为局部边中位数的 `3.40--7.35` 倍，并完成 20 个固定种子的 bridge 异常扰动统计。修复 3D 图继续列为模型侧次级候选，无模型图结构阶段结束。
- 2026-09-23 [完成 MotionForesight E3 前置接口检查](progress/2026-09-23-surface-relation-e3-motionforesight-preflight.md)：在 5 个正式 Ego4D 视频上完成 observed-only 7 帧输入、48-query 身份映射、`P0` 与两类 `H` 缓存、时间/坐标/无未来信息审计和视频级划分；模型接口 5/5 通过，固定输入复跑逐字节一致。现有 E2 pseudo-GT 与 observed-only DA3 序列存在明显坐标登记误差，浇水壶一致 pseudo-GT 原型虽将中位归一化残差从 25.2% 降至 3.9%，dense identity 仍有较多失配，因此 Relation Head 训练暂未启动。
- 2026-09-23 [复核 E3 前置检查并固定监督质量闸门](progress/2026-09-23-surface-relation-e3-preflight-review-and-gates.md)：确认 MotionForesight `P0,H` 接口已可用于后续 probe，但一致 pseudo-GT 的 `p90`、异常 query 和 dense anchor 误差仍未满足训练条件；要求用 past-only 12 帧显式建立跨序列 query 身份，采用留出 query 登记审计、逐点 validity 与有效图连通性检查，五视频审计通过前不训练 Relation Head。
- 2026-09-24 [运行 E3 严格 pseudo-GT 身份审计](progress/2026-09-24-surface-relation-e3-strict-pseudogt-audit.md)：实现 past-only/监督双 12 帧窗口、共同 `t0` 全局一对一匹配和四折 query cross-fitting。首个视频 48/48 query 有效且 2D 图连通，但留出 3D 登记误差 `p90=34.52%`、最大 `59.15%` bbox 对角线，gate 失败；其余四个视频已进入远端顺序审计，Relation Head 保持暂停。
- 2026-09-24 [完成五视频 E3 严格 pseudo-GT 审计](progress/2026-09-24-surface-relation-e3-strict-audit-review.md)：五个视频均为 `48/48` 有效 query，2D anchor 与 `k=3` 图连通，但 3D 留出登记 gate 全部失败，p90 为 `7.40%--34.52%`、最大误差为 `39.39%--59.15%`；补充直接 UV、真实 bbox、邻近匹配和两两距离诊断后，将原因表述为跨序列三维登记失败，query identity 仅是候选原因之一，Relation Head 继续暂停。
- 2026-09-24 [冻结跨上下文 pseudo-GT 监督路线并完成单上下文关系审计](progress/2026-09-24-surface-relation-supervision-route-freeze.md)：修正 query 间距字段命名，冻结跨 DA3/TrackCraft3R 上下文的绝对 3D pseudo-GT 监督；在浇水壶单一 16 帧坐标链上完成离线边位移和规范化形状目标审计，仍未启动 Relation Head 训练。
- 2026-09-24 [完成五视频单一上下文关系目标质量审计](progress/2026-09-24-surface-relation-single-context-quality-audit.md)：五个视频均通过 48-query anchor validity 与 2D 图连通性 gate，但未来两两距离漂移从 p90 `4.09%` 到 `157.39%`，说明单一上下文目标仍需刚体/受控数据对照，尚不能生成真实视频 Relation Head 训练缓存。
- 2026-09-24 [修正单上下文质量审计 t0 索引问题](progress/2026-09-24-surface-relation-single-context-quality-audit-review.md)：确认上一版五视频报告误用 12 帧窗口的索引 2，不能代表正式 16 帧窗口；修正脚本从时间戳计算 t0。现有四个视频缺少正式 16 帧输出，浇水壶 16 帧结果 t0=6 但 anchor p90 `5.27%`、最大 `29.91%`，gate 失败，真实视频训练继续暂停。
- 2026-09-24 [完成 E3 监督定位实验并冻结 vector pseudo-GT 路线](progress/2026-09-24-surface-relation-e3-locator-experiments.md)：同步四份正式 16 帧 JSON；`9069...` 与浇水壶的 t0 首帧教师登记分别为 p90 `14.03%`、`40.02%`，均失败。浇水壶 `process_res=392` 下 anchor p90 恶化至 `35.42%`，排除分辨率差异为主要原因；停止当前真实视频 vector relation pseudo-GT 路线。
- 2026-09-24 [完成 E3 受控 Relation Head 工程测试](progress/2026-09-24-surface-relation-e3-controlled-head-tests.md)：在合成 oracle 数据上通过 zero correction、梯度有限性、单样本过拟合和 query permutation；edge head 与 node residual 参数量约匹配。结果仅支持接口与优化可行性，不支持真实视频关系可学习性结论。
- 2026-09-24 [复核 E3 受控 Relation Head 测试并放行下一阶段](progress/2026-09-24-surface-relation-controlled-head-review.md)：确认 zero correction、反向传播、单样本过拟合和同步 query permutation 均通过；参数量对照基本匹配，但未测试真实 `H_output` 信息量或 edge head 优势。工程阶段结束，下一步转向 MotionForesight 同坐标监督链复现。
- 2026-09-24 [复核同坐标监督链与 head 对照审计](progress/2026-09-24-surface-relation-same-coordinate-audit-review.md)：确认源码与缓存的坐标/时间接口检查通过，但原始 `*_sparse.npz` 缺失、缓存仍来自 TrackCraft3R，因此 `relation_loss_ready` 保持为 false。10 种子随机单样本拟合中 edge 与 node 的绝对最终损失无稳定优势，暂不冻结结构选择。
- 2026-09-24 [完成 MotionForesight 官方监督来源审计并调整后续流程](progress/2026-09-24-surface-relation-official-supervision-source-audit.md)：核对官方仓库、预处理脚本、SparseTrackDataset 和论文后，确认公开代码只提供 sparse 读取协议与 user-NPZ 预处理，未提供训练 sparse 标签生成器。当前重建文件仅是 schema smoke test；下一步只允许按 checkpoint 的 `7+15`、完整 22 帧协议做一次有界标签链重建审计。
- 2026-09-24 [复核 E3 定位实验并确定监督路线边界](progress/2026-09-24-surface-relation-locator-decision-review.md)：确认 t0 首帧教师与分辨率 A/B 均未通过登记质量门槛；冻结当前 TrackCraft3R/DA3 vector pseudo-GT 监督链，但保留“点间存在可利用相关运动”和图积分数学可用性这两个研究假设。Relation Head 后续仅开展合成、oracle 与 E1 受控工程测试，真实训练需等待 MotionForesight 同坐标标签链恢复。
- 2026-09-24 [复核正式单上下文关系目标闸门](progress/2026-09-24-surface-relation-formal-single-context-gate-review.md)：确认修正后的 16 帧 t0/future 协议正确；浇水壶实际 t0 anchor 与未来距离漂移仍失败，不能生成真实视频训练缓存。其余四个视频只允许进行一次有界的 16 帧数据审计，不能用旧 12 帧结果放行或筛选样本。
- 2026-09-24 [决定五视频正式单上下文闸门后续流程](progress/2026-09-24-surface-relation-five-video-formal-gate-decision.md)：正式 16 帧审计中 2/5 仅通过 t0 anchor 闸门，未来标签质量和与 observed-only `P0,H` 的坐标接口仍未通过；决定不训练真实 Relation Head，只做浇水壶 `process_res=392` 敏感性 A/B，以及一个 PASS/一个 FAIL 样本的 t0 首帧直接 query 教师定位实验。
- 2026-09-24 [复核点间关系方法假设与后续决策](progress/2026-09-24-surface-relation-method-assumption-review.md)：保留“同一目标点间存在可利用相关运动”这一研究假设，否定硬刚体约束和跨上下文绝对 3D 登记假设；单上下文关系目标仍受 `67.70 px` anchor 异常与大形状漂移限制，要求先做逐点质量屏蔽、刚体样本检查和合成/oracle Relation Head probe。
- 2026-09-24 [复核单上下文关系目标质量审计](progress/2026-09-24-surface-relation-single-context-quality-audit-review.md)：发现质量脚本将 t0 错用为第 3 帧索引 2，而正式 16 帧关系窗口的 t0 位于索引 6，未来漂移也使用了错误切片；五视频 anchor gate 与漂移结果暂不能作为正式证据，真实 Relation Head 训练继续暂停。
- 2026-09-24 [复核 E3 严格身份审计结果](progress/2026-09-24-surface-relation-e3-strict-audit-review.md)：五个视频的 2D anchor、有效 query 和图连通性均通过，3D 留出登记全部失败；直接 UV 与跨点距离诊断支持跨 DA3 上下文几何不一致，但不能唯一归因于 TrackCraft3R identity。发现辅助脚本的“最近邻/间隔”字段实际是 query 间距，要求修正命名或重新计算 dense 候选匹配距离；停止继续调整历史窗口放行当前监督，转向单一上下文或关系不变目标的最小检查。
- 2026-09-24 [复现 MotionForesight 同坐标监督链并完成 Relation Head 受控对照](progress/2026-09-24-surface-relation-motionforesight-same-coordinate-audit.md)：核对 `SparseTrackDataset` 的最后观测相机坐标变换、共享归一化和 validity mask；同步五个 `P0/H` 审计缓存，完成 decoded-coordinate 关系审计；合成 10 种子下 edge Relation Head 与等规模 node residual 均可优化，真实视频监督仍未放行。
- 2026-09-24 [查找并重建 MotionForesight sparse 标签格式链路](progress/2026-09-24-surface-relation-motionforesight-same-coordinate-audit.md)：确认官方仓库、发布 checkpoint 和 TrackCraft3R 子模块均未提供 `*_sparse.npz` 生成器或训练标签下载入口；新增重建适配器并在一个 48-query dense TrackCraft 样本上验证字段格式，结果仅作为工程链路审计，不能替代官方监督。
- 2026-09-24 [执行 MotionForesight 22 帧 7+15 协议获取尝试](progress/2026-09-24-surface-relation-motionforesight-same-coordinate-audit.md)：依据论文公开的 SSv2 22 帧流程，从远端完整 15 fps RGB 序列生成 `9069...` 的 22 帧 DA3 输入并通过时间/坐标字段审计；V100 上 22 帧 TrackCraft3R dense 推理因显存/进程退出未产出轨迹，未将失败解释为标签质量结论。
- 2026-09-24 [复核 22 帧协议获取尝试并停止资源调参](progress/2026-09-24-surface-relation-22-protocol-decision.md)：确认完整 `7+15` 输入链路通过，但 TrackCraft3R 两次均未生成完整 dense 轨迹；同时区分未来 RGB 未传入 MotionForesight 与 DA3 可能使用完整窗口上下文。停止当前 V100 上无边界调参，`relation_loss_ready=false` 和真实训练暂停保持不变。
- 2026-09-24 [监督来源缺失后的路线决策](progress/2026-09-24-surface-relation-supervision-source-decision.md)：确认代码 AI 本轮未完成要求的完整 `T=22`、`7+15` 协议兼容审计，只重复报告公开官方标签缺失。停止继续无边界搜索官方 sparse 文件；保留重建适配器为 schema 测试，真实 Relation Head 和 decoded-coordinate relation-loss pilot 继续暂停。
- 2026-09-24 [整理点间关系方案当前实验状态](progress/2026-09-24-surface-relation-current-status.md)：汇总 E0--E3-controlled、E2 图阶段、MotionForesight 接口和 22 帧监督获取尝试；确认工程/数学链路可行，但真实视频学习可行性仍受监督来源、坐标一致性和完整未来标签阻塞。
- 2026-09-24 [修正方法假设并重启 TrackCraft3R query 获取决策](progress/2026-09-24-surface-relation-current-status.md)：确认固定硬刚体约束不是本方法假设；此前失败主要针对 22 帧全图 dense 获取，不能推出 query-only 推理失败。要求沿用既有成功环境优先尝试 22 帧、48-query 输出；代码任务当前尚未产生新产物。
- 2026-09-24 [完成 t0→frame-0 映射版 query-only 12/16/22 smoke](progress/2026-09-24-surface-relation-query-only-12-16-22-smoke.md)：使用已有 observed-only dense TrackCraft 映射固定 48 个 frame-0 query；12/16 帧分别通过 `(12,48,3)`、`(16,48,3)` 形状和 finite 检查；22 帧在 `predictor.predict()` 的 DiT self-attention 阶段 CUDA OOM，未生成完整 `(22,48,3)`。
- 2026-09-26 [复核点间关系方案的现有证据与调整范围](progress/2026-09-26-surface-relation-method-evidence-review.md)：保留时变软关系修正及组件级 L2 积分核心；将 2D 图限定为工程候选，改以可审计的同坐标标签作为监督条件，区分九步探索与完整 7+15 复现。当前无真实 Relation Head 收益证据；仅分析，未运行实验。
- 2026-09-26 [核对实验前点间关系方法的已证实依据](progress/2026-09-26-surface-relation-preexperiment-evidence-validation.md)：仅依据实验前方法定义整理证据；E0 支持图积分与组件规范，E1 支持存在可恢复的内部结构误差及有限噪声容错，E2 支持观测图的工程可用性，E3 支持 MotionForesight 接口和受控优化。真实关系学习收益仍未证实。
- 2026-09-26 [制定点间关系方法后续可行性实验计划](progress/2026-09-26-surface-relation-next-experiment-plan.md)：将后续验证分为同坐标监督闸门、真实 baseline 结构漂移审计、关系信息 probe、最小端到端消融、图稳健性和异常边鲁棒性；明确各阶段停止条件，未运行新实验。
- 2026-09-26 [估算 F0 同坐标监督链显存需求](progress/2026-09-26-surface-relation-f0-vram-estimate.md)：依据 22 帧 query-only OOM 日志估算当前 SDPA 路径峰值至少约 34.22 GiB；建议 48 GB 级显存作为稳妥下限，80 GB 更安全，未运行新实验。
- 2026-09-26 [完成 F0 query-only TrackCraft3R 获取](progress/2026-09-26-surface-relation-f0-query-only.md)：在双 V100 远程环境按完整 7+15、22 帧协议和 48 query 成功生成 `(22,48,3)` 轨迹；随后用同一 query 完成 MotionForesight observed-only 配对，t0 登记误差相对联合点云包围盒对角线 p90 `2.43%`、最大 `3.90%`，同输入坐标接口 pilot 通过；未来边关系差异仍需后续结构漂移审计。
- 2026-09-26 [完成 F1 并复核 F2 可学习性 pilot](progress/2026-09-26-surface-relation-f1-f2-pilot-review.md)：单视频 F1 中去质心点误差、边向量相对误差与边长形状误差均随 horizon 增长；但脚本 gate 与实际放行逻辑不一致。F2 exploratory 结果中 H+geometry edge head 未优于未训练 baseline，自动优势 gate 漏比 baseline，node residual 的平移指标也不可公平比较；F3 暂停，F2 结果未验证。
- 2026-09-26 [完成 F2 修正版单视频 pilot](progress/2026-09-26-surface-relation-f2-redesigned-pilot.md)：采用 zero-init、统一去质心/边向量损失、训练时域标准化和验证集选 checkpoint；在最后 5 个未来槽位的测试中，H edge 边相对误差 `21.15%` 高于 baseline `19.00%`，node residual 结构误差较低但 raw ADE 明显恶化，F2 单视频 pilot 未放行 F3；多视频 F2 仍等待额外同坐标 F0 bundle。
- 2026-09-26 [完成 F2 多轮变体与 query 留出复核](progress/2026-09-26-surface-relation-f2-iterative-variants-review.md)：交互式 SSH 认证再次失败；在单视频上完成 8/2/5、10/2/3、6/4/5 三种时域切分和 32/16 query 留出。短时域覆盖时出现正结果，长时域和未见 query 均未超过 baseline，说明当前关系修正主要是已见时域拟合；Relation Head 与 node centered residual 暂不具备泛化证据，F3 保持暂停。
- 2026-09-26 [完成 F2 统一分辨率多视频复核](progress/2026-09-26-surface-relation-f2-res518-final-review.md)：确认此前 `process_res=392` 与 query manifest 长边 518 不一致，按 `process_res=518` 重建四个可用同坐标 F0 bundle；四视频 LOO 中 baseline 的去质心误差中位 `0.1152`、绝对边向量误差 `0.1100`，优于 H edge 的 `0.2138`、`0.1821`。同视频时间留出仅显示局部拟合收益，未形成稳定放行；oracle 边位移积分误差接近 `1e-16`，因此当前瓶颈是关系信息预测/跨视频不变性，不是积分器。F3、真实视频 Relation Head 和复杂 gate 继续暂停。
- 2026-09-26 [诊断 F2 对点间关系模型设计的意义](progress/2026-09-26-surface-relation-f2-design-diagnosis.md)：四视频留出中 H edge 的去质心与边向量误差均在每个视频上高于 baseline；F2 逐边 MLP 缺少原方案中的初始边关系和边可信度，尚未完整检验原方法。优先补齐观测期关系与共享物体状态，保守预测时变关系残差；不据此否定软关系假设或引入硬刚体约束。
- 2026-09-26 [完成 F2 观测关系、共享状态与保守修正实验](progress/2026-09-26-surface-relation-f2-observed-relations-experiment.md)：四视频留出、五种子对照中，history/shared/safe 在原对齐监督上均为 `0/4` 视频优于 baseline。发现三个视频的教师 t0 边偏差与未来修正目标几乎等大且同向；改用 t0 位移锚定的诊断性监督后，`6516` 的 history 结构指标改善，但其余三视频退化，三种方法均为 `1/4`。当前主要缺口是可靠地判断何时需要修正及监督坐标链质量，暂不能单独否定冻结输出端接口。
- 2026-09-26 [制定冻结输出端关系接口的预注册取舍实验](progress/2026-09-26-surface-relation-interface-decision-experiment.md)：采用同一渲染链三维真值的 PointOdyssey 作洁净主试验、TAPVid-3D 作真实外部确认；固定视频级分割、原始 MotionForesight、冻结关系头/节点残差、可训练预测模块关系头/节点残差五组及机会、安全、收益门槛。结果事先映射为保留冻结接口、转向可训练关系建模、采用简单节点残差或停止当前关系扩展；本轮只制定方案，未执行。
- 2026-09-26 [完成 PointOdyssey 冻结接口数据与坐标预检](progress/2026-09-26-pointodyssey-interface-data-preflight.md)：官方两独立样本的 7+15、48-query GT 投影 p90 均 <0.4 px；RGB-only DA3、观测 TrackCraft3R 和冻结 MotionForesight 已在单 mask 接口跑通，输出 15 帧 `P0/H`。整物体目标的 GT→DA3 t0 留出登记 p90 为 34.1%、21.0%，超过 10% 闸门；远端 SSH 随后中断，五组训练与方法裁决均未执行。
- 2026-09-26 [恢复服务器并尝试 PointOdyssey 完整 22 帧输入](progress/2026-09-26-pointodyssey-full22-da3-oom.md)：上传两个已审计目标所需的 22 帧 RGB，`r4_new_f` 在 DA3 `process_res=518` 阶段因单卡约 32 GiB 显存不足而退出，未生成 TrackCraft 标签；按执行规范暂停自动低分辨率重试，主实验仍未启动。
- 2026-09-26 [完成 PointOdyssey 双卡低分辨率 F0--F2 诊断](progress/2026-09-26-pointodyssey-full22-f0-f2-diagnostic.md)：双 V100 分别运行两个目标的 `process_res=392` 完整 22 帧链路；F1 两目标均观察到结构漂移，双视频 LOO F2 中 baseline 在全部主要指标上优于 geometry edge、node centered 与 H edge。低分辨率改变了 `P0/H` 数值，因此结果仅作同输入诊断，不能替代正式分辨率裁决。
- 2026-09-26 [完成 PointOdyssey 分辨率稳健性检查](progress/2026-09-26-pointodyssey-resolution-robustness.md)：对比 `process_res=392` 双视频与 `process_res=518` 四视频 F2，确认特征数值不等价，但 baseline 优于 geometry edge、node centered 与 H edge 的相对方向一致。`392` 可继续作为诊断配置，不能替代统一分辨率正式主试验。
- 2026-09-26 [分析 F2 失败原因并提出点间关系重设计](progress/2026-09-26-surface-relation-f2-cause-and-redesign.md)：证据指向无条件修正、监督坐标偏差和冻结表示跨视频迁移不足；保留软关系假设，改为 baseline-preserving 的对象中心状态、低维形变残差和对象级门控，待同坐标真实标签上验证。
- 2026-09-27 [澄清 F2 两次 TrackCraft3R 坐标差异](progress/2026-09-27-surface-relation-coordinate-mismatch-explanation.md)：说明关系头比较的是点对相对向量；共同平移会被抵消，真正的风险是跨上下文运行产生的非一致局部几何误差。原始 MotionForesight 的统一坐标监督链与 F2 的伪标签拼接链不同。
- 2026-09-27 [说明 F2 伪标签坐标对齐机制](progress/2026-09-27-surface-relation-pseudogt-alignment-detail.md)：确认相似变换由伪标签预处理脚本用 48 个 t0 对应 query 拟合，并统一应用到未来教师轨迹；MotionForesight 本身不读取教师坐标，也未执行逐点平移对齐。
- 2026-09-27 [进一步澄清坐标对齐与监督链差异](progress/2026-09-27-surface-relation-coordinate-alignment-clarification.md)：确认全局坐标变换不能消除局部 t0 几何残差；若 MotionForesight 使用同样的双 TrackCraft3R 伪标签链也会受影响，官方链路的区别在于观测与未来标签本来就在统一坐标中。
- 2026-09-27 [修正 MotionForesight 原始训练边界描述](progress/2026-09-27-surface-relation-motionforesight-training-boundary-correction.md)：确认官方训练确实使用完整视频伪轨迹监督、只输入观测 RGB/几何；其公开加载器对观测 `Pj` 与完整 tracks 使用同一最后观测相机变换和归一化。F2 另行加入的观测期 TrackCraft3R 轨迹属于额外坐标链，不能把两类误差混为一谈。
- 2026-09-27 [澄清 MotionForesight 与 F2 两次 TrackCraft3R 链路](progress/2026-09-27-surface-relation-motionforesight-two-trackcraft-clarification.md)：确认 MotionForesight 的观测输入是 DA3 pointmap，不是只看过去的 TrackCraft3R 轨迹；完整 TrackCraft3R 仅用于离线伪标签。F2 额外引入 observed-only TrackCraft 与 full-video TrackCraft 的跨上下文登记，因此逐点 t0 残差不能当作普通坐标变换。
- 2026-09-27 [评估仅用 DA3 pointmap 构造观测关系特征](progress/2026-09-27-surface-relation-da3-observed-relation-route.md)：判断该路线可以移除双 TrackCraft 跨上下文登记误差；推荐先使用最后观测帧 DA3 几何构造静态边特征，再与 observed-only TrackCraft 版本和 baseline 做相同划分对照。
- 2026-09-27 [澄清 t0 边残差锚定及既有 F2 证据](progress/2026-09-27-surface-relation-edge-t0-residual-assessment.md)：核对三维边向量的锚定公式及 `anchored_displacement` 已有四视频结果；说明它能移除静态 t0 差异却未令当前关系头稳定超过 baseline，并更正上一条建议中 frame-0 query 无法直接按同像素采样最后观测帧 DA3 点图的遗漏。
- 2026-09-27 [界定 t0 位移锚定能够排除的误差](progress/2026-09-27-surface-relation-edge-t0-residual-assessment.md)：复核训练脚本和四视频留出结果，确认三维边向量的等价点位移锚定精确消除 t0 双运行静态偏移，不需要观测期运行生成未来轨迹；撤回此前将缺少该轨迹作为未排除跨运行问题的说法。三种 head 在另三个视频过修正，验证集门控均选全量修正。
- 2026-09-27 [确定 t0 锚定后的关系头诊断决策](progress/2026-09-27-surface-relation-next-decision-after-anchoring.md)：暂不启动新实验或继续改伪标签；建议在已有四视频留出上区分预测修正的方向错误与幅度/门控错误，再按结果决定是否保留冻结输出端关系头，正式结论仍以预定的独立视频实验为准。
- 2026-09-27 [区分坐标系对齐与点位置登记](progress/2026-09-27-surface-relation-alignment-vs-registration.md)：明确全局变换只统一坐标表示，不能保证每个 query t0 重合；若对齐后仍有残差，未来伪标签是近似监督目标。训练应使用观测输入与变换后的未来标签，不能把不可用的完整视频 t0 教师点当作推理输入。
- 2026-09-27 [完成 F2 修正方向与幅度诊断](progress/2026-09-27-surface-relation-f2-direction-gate-diagnostic.md)：60 组锚定监督复跑与原 F2 指标完全一致；留出视频事后最佳幅度在三视频远小于原选的全量修正，表明当前门控跨视频失准。方向存在有限信息，但事后选择使用测试伪标签，不构成可部署收益或接口有效性证明。
- 2026-09-27 [完成 F2 保守门控跨视频与逐点对照](progress/2026-09-27-surface-relation-f2-gate-transfer.md)：四视频训练、两个 PointOdyssey 转移视频的固定 `g=.25` 与可拒绝门控均已运行；门控在两段均选择修正，关系头仅一段改善，另一段结构与 ADE 恶化，逐点对照两段均小幅改善。预设判据未通过；转移集 `392/518` 分辨率不一致且 DA3 使用完整视频上下文，不能据此裁决冻结输出端接口。
- 2026-09-27 [评估点间关系子方法的结构性重设计](progress/2026-09-27-surface-relation-architecture-rethink.md)：F2 指向未来边修正预测不稳，保守门控未补上缺失的信息；建议停止阈值微调，改为 query 对齐、跨点交互的联合时空轨迹解码，保留软关系损失和观测图，并以可靠同坐标数据和等容量逐点消融检验；未训练新模型。
- 2026-09-28 [发现 DAVIS query 时间对齐错误](progress/2026-09-28-davis-query-time-alignment-audit.md)：query 实际按 t0=源帧20 mask 采样，但被写入 TrackCraft3R 默认 frame-0 query 接口，并在源帧14可视化；`soccerball` 在源帧14为 0/48、源帧20为 48/48，说明当前四视频轨迹和质量结论需在修正 query frame index 后重做。

- 2026-09-28 [生成修正后的 DAVIS frame-0 query 预览](progress/2026-09-28-davis-corrected-frame0-query-previews.md)：按源帧14（模型输入 frame 0）目标 mask 重新确定性采样48点，四段均生成854×480静态确认图；尚未重新运行TrackCraft3R或MotionForesight，等待query位置确认。

- 2026-09-28 [完成 DAVIS frame-0 直接 query 四视频实验与质量复核](progress/2026-09-28-davis-frame0-direct-four-video-audit.md)：四段均完成7帧仅观测、22帧离线教师和冻结 MotionForesight 推理，远端58个产物与本地逐字节核验；四段质量闸门均失败（0/4），未启动整视频留一训练。修正汇总脚本的四视频全通过条件，记录磁盘清理及旧预览作废状态。

- 2026-09-28 [定位 DAVIS TrackCraft3R 首帧偏移的坐标链根因](progress/2026-09-28-davis-trackcraft-frame0-coordinate-diagnosis.md)：对 `car-turn` 和 `drift-turn` 只读核对 mask/query、DA3 深度/K/w2c、模型 resize/整数取样、两次 TrackCraft3R 输出和可视化；Pj 起点投影正确且 48/48 在 mask 内，偏移来自解码 frame0 residual 未被强制为零。提出保留 raw 输出的 Pj0 常量锚定修复，未覆盖原产物或启动训练。
- 2026-09-28 [完成 DAVIS 四视频官方交错 TrackCraft3R 与 Pj0 锚定审计](progress/2026-09-28-davis-official-interleaved-pj0-audit.md)：四段均完成 23 帧 DA3、两条官方 12 帧交错 query-only 推理、Pj0 锚定、坐标链与未来 mask 诊断；frame0 均恢复 48/48 mask 命中，但四段固定质量 gate 全部失败，未来教师质量不足，未启动整视频留一训练。

- 2026-09-29 [完成 DAVIS 四视频联合关系解码器教师审计](progress/2026-09-29-davis-ab-teacher-experiment.md)：在 RTX 6000D 上完成四段 48-query、7 帧 observed-only、22 帧 TrackCraft3R query-only 替代教师链；四段均生成 `(22,48,3)` 且无显存阻断，但 Route A/B 质量闸门均为 `0/4 PASS`，因此未启动整视频留一训练。远端结果已同步，本地归档及 53 个文件 SHA256 全部一致。
