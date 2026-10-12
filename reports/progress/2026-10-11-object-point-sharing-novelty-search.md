# 同物体点共享信息用于未来三维预测：针对性查新

- 日期：2026-10-11，Asia/Shanghai。
- 完成内容：扩展关键词检索；沿主要论文参考文献追查粒子物理预测、视觉物体预测与点轨迹生成；核查最接近候选的方法章节、公式、图注和消融范围。
- 使用 academic-research-suite 区分论文事实、思想重合与具体实现；使用 research-progress-logger 保存查新记录。
- 本轮无远端连接、GPU、模型训练、实验协议变更或对实验线程发消息。

## 1. 判定范围

用户定义的思想：通过结构或模型改造，让同一物体上的多个点共享信息，以提高未来三维点轨迹预测能力。

核查四项：

1. 预测尚未观测到的未来，而非在完整视频中跟踪。
2. 更新或输出物体上的多个三维点，而非只有单个中心或对象位姿。
3. 明确对象归属，或建立对象内层级/关系，避免把一般全场景注意力自动算成对象级设计。
4. 共享结构实际参与未来点预测。

机器人动作条件、物理状态输入、仿真环境与单目视频之间的区别单独记录。它们影响任务和方法可比性，不自动取消思想重合。没有添加“必须使用 MotionForesight”之类限制。

“思想重合”检验研究主张是否已有先例；“具体实现相同”还需比较已确定的网络、输入、训练目标及输出。当前本项目的新共享架构尚未定型，不能对尚未定义的实现宣布相同或独创。

## 2. 有明确思想重合的主要工作

### HRN：Flexible Neural Representation for Physics Prediction，NeurIPS 2018

[官方论文](https://proceedings.neurips.cc/paper_files/paper/2018/file/fd9dd764a6f1d73f4340d570804eacc4-Paper.pdf)，§3、§4.1–4.3，Figures 2–4。

物体由三维粒子及对象根节点组成；信息经粒子→上级、同级交换、上级→粒子传播。根节点预测全局运动，其他节点预测相对父节点运动，再组合未来点位置。支持刚体和可变形物体；另监督未来相邻粒子直线距离。

判定：思想重合程度高。对象层级共享、全局与局部运动组合、抑制物体点散开都有直接先例。输入为粒子状态、关系及外力，采用物理仿真训练，与本项目的单目视频接口不同。这里的距离监督不等同于物体表面路径距离。

### DPI-Net：Learning Particle Dynamics for Manipulating Rigid Bodies, Deformable Objects, and Fluids，ICLR 2019

[作者论文](https://arxiv.org/pdf/1810.01566)，§3.2，pp.4–5。

每个需要长程耦合的物体内聚类粒子、增设簇根节点，依次执行粒子间、粒子到根、根间、根到粒子的信息传播。刚体聚合粒子特征预测共同旋转与平移，再更新全部点；不同材料采用不同规则。

判定：物体内跨点传播及共享运动用于未来三维预测，属于思想重合。输入粒子物理状态，应用于模拟与操控；硬刚体处理属于它的分支，不是本项目所有目标的前提。

### VGPL：Visual Grounding of Learned Physical Models，ICML 2020

[官方论文](https://proceedings.mlr.press/v119/li20j/li20j.pdf)，§3.1–3.3，式(1)–(9)，Table 2、Figure 4。

视觉模块从图像序列推断三维粒子及对象分组。图边编码两粒子是否同属一个物体，消息传递后依据历史预测未来位置；同一刚体点共享中心平移和旋转，非刚体使用相应点更新。

判定：最接近本项目宽泛主张的视觉先例之一，兼具视觉输入、对象归属、跨点共享和未来三维点预测。物理环境、粒子表示及物性推断机制与当前自然视频 query 任务不同。它的消融含刚性处理，不能解释为单独隔离了全部信息共享的收益。

### AdaptiGraph，RSS 2024

[作者论文](https://arxiv.org/html/2407.07889v1)，§III-A–B，式(1)、(3)–(6)。

对象点与机器人点构成三维粒子图；关系描述同物体或不同物体，迭代传播特征，再解码未来点状态。物性条件共同进入点特征，支持绳、布、刚体和颗粒材料。

判定：对象内关系共享用于未来三维点预测已有机制先例。创新重点为物性条件与在线适应，输入包含三维状态及机器人动作；不能直接等同于单目视频自主未来预测。

### HOPNet：Physics meets Topology，2024 预印本；2025 期刊发表

[核查版本](https://arxiv.org/html/2411.11467v1)，§2、§5.2，式(12)–(16)；[期刊元数据](https://www.nature.com/nature-index/article/10.1038/s41467-025-62250-7)。期刊题名为 Integrating physics and topology in neural networks for learning rigid body dynamics。

点、网格面和物体级单元交换信息，对象聚合的信息再传给所属节点，用于预测节点加速度及未来位置。§3.3 移除物体级单元后位置 RMSE 增加44.4%；该消融改变表示及通信结构，不能视为只改变某一共享操作。

判定：明确物体级结构改善未来三维节点预测的先例。使用网格与物理状态，面向刚体模拟。方法核查基于公开预印本，未把期刊正文访问失败的内容当作已阅读全文。

## 3. 部分思想重合与相关工作

| 工作及原始来源 | 核查位置与分类依据 | 分类 |
| --- | --- | --- |
| [SE3-Nets，ICRA 2017](https://arxiv.org/pdf/1606.02378) | §III，式(1)–(4)：学习对象/运动分组及共同三维变换，变换层生成各输入点的未来位置；输入深度点云与动作，偏向刚体 | 共享对象运动的思想先例；不等同于一般可形变点关系设计 |
| [ParticleFormer，2025](https://arxiv.org/html/2506.23126v1) | §3.1–3.3：从双目图像分割任务对象、编码物性，点集合通过三层普通注意力预测未来三维位移；未给出专门对象根节点或同实例通信限制 | 对象点联合预测的近邻工作；宽泛“让对象点交换信息”已有先例，专门对象级通信机制不同 |
| [3PoinTr，2026](https://arxiv.org/html/2603.08485v1) | §III-A：普通 Transformer 预测未来点轨迹；明确避免对象中心掩码等先验 | 相关：三维未来点联合预测，缺少明确同物体归属结构 |
| [PointWorld，2026](https://arxiv.org/html/2601.03782v1) | §3.1：RGB-D与未来机器人动作条件，全场景点流预测；强调不假设对象归属/材质先验 | 相关：动作条件三维点预测，不能据一般交互能力判为本项目对象内结构重合 |
| [PointZero，2026](https://arxiv.org/html/2609.19142v1) | §3–4：已给少数完整未来轨迹，补全稠密未来三维轨迹；普通DiT共享点/动作/视觉条件 | 相关：稀疏轨迹条件传播，推理条件中已有部分未来，任务不同 |
| [MolmoMotion，2026](https://arxiv.org/html/2606.18558v1) | §2.2：同对象多个点联合输出，坐标相对首个query锚点；AR或普通DiT解码 | 同任务近邻：共享坐标原点及联合生成，未据此认定已实现专门对象内关系传播机制 |
| [MotionForesight，2026](https://arxiv.org/html/2607.16192v1) | §3.2–3.5及附录A.1：视觉/点图潜变量进入视频DiT，解码未来对象点轨迹 | 当前基线；已有跨时空潜变量注意力，不能声称基线完全没有共享 |
| [MoRe4D，arXiv:2512.05044v1](https://arxiv.org/html/2512.05044v1) | §4.2：点轨迹生成、深度条件及逐token运动特征调制；核查版本未显式建立同对象点分组通信 | 相关：几何与运动一致的三维生成；不能把场景几何一致性直接等同对象内共享结构 |
| [Gen-Points](https://gen-points.github.io/) | 官方架构说明：点间/时间/图像注意力；此前核查未建立明确对象归属结构 | 相关：通用联合点预测 |
| [TrajLoom](https://arxiv.org/html/2603.22606v1) | §3.2.1：二维轨迹与多尺度邻点相对运动监督 | 相关：维度与对象级组织均不同 |
| [CoTracker](https://co-tracker.github.io/)、[SpatialTracker](https://arxiv.org/html/2404.04319v1) | 已见视频内联合跟踪；后者包含局部刚性关系 | 相关：跟踪阶段共享，不构成本项目未见未来预测组合 |
| [ObjectForesight](https://objectforesight.github.io/) | 对象中心几何条件与未来刚体位姿输出 | 相关：输出位姿，不直接构成一般多点共享预测机制 |

表中 Gen-Points、TrajLoom、CoTracker、SpatialTracker、ObjectForesight 延续前一轮原始来源核查；本轮重点扩展物理粒子与近期三维预测分支。

## 4. 检索记录与证据限制

检索日期为2026-10-11。检索入口为网页搜索，方法事实采用作者/会议/期刊原文与官方项目，不使用第三方解读定性。沿HRN、DPI-Net、VGPL和PointZero的参考文献扩展候选；按题名/arXiv号合并重复入口。本轮是针对性查新，不是穷尽性系统综述或专利检索。

使用的主要原始检索式：

```text
"3D" "point trajectory" "forecasting" "object" "graph"
"point trajectory prediction" "object-centric" shared
"Learning Particle Dynamics" rigid bodies hierarchical graph
"3D motion forecasting" "object" "coherence" points
"3D point" "forecasting" "object-centric"
"point trajectory" "object" "shared" prediction 3D
"object-centric" "motion prediction" "graph" "RGB"
"Integrating physics and topology in neural networks for learning rigid body dynamics"
"Flexible Neural Representation" "physical" Mrowca objects
"3D" "forecasting" "instance" "rigid" "scene flow"
"point trajectories" "object-level" "prediction"
"3D point" forecasting "object" "coherence"
"3D" "trajectory prediction" "intra-object"
"3D" "forecasting" "object-aware" points
"Geometry-Aware Single-Image 4D Synthesis" trajectory object
"DynBridge" "interaction dynamics" arxiv
"AdaptiGraph" material graph rigid hierarchy
"SE3-Nets" "rigid body motions"
"3D point trajectory" "object-level" forecasting shared architecture
"Pose-Transformed Equivariant Network for 3D Point Trajectory Prediction"
```

辅助定位包含官方域名与候选题名的查询。

候选处理边界：

- DynBridge、Forecasting Future Videos from Novel Views的正文读取部分失败；摘要不足以确定专门同对象共享结构，不计入确认重合，也不宣布已排除。
- PT-EvNet的CVF摘要确认三维点预测与全局/局部运动分解；未获得足够方法正文支持明确对象归属，不列为确认重合。
- HOPNet期刊正文访问失败后使用作者预印本核查，两个题名作为同一工作记录。
- 搜索摘要对arXiv:2512.05044使用MoGe4D题名；实际可读v1标题为MoRe4D，本报告采用核查版本。没有将二者算作两篇。
- 人体/分子预测、分割、检测、已见帧场景流、轨迹控制视频生成不作为本项目完整思想重合的证据。未以这些候选扩展任务。
- 无检索条目总数、覆盖率或“全部论文已排除”的虚构统计；AI辅助核查不记作用户人工全文阅读。

## 5. 完成本轮后的判断

1. 用户定义的宽泛思想有明确先例，HRN、DPI-Net、VGPL、AdaptiGraph、HOPNet构成实质证据；这些论文并非仅预测对象中心或仅做二维跟踪。
2. HRN尤其接近“对象共同运动信息＋局部点变化＋未来关系监督”，且支持可形变物体。不能通过强调本项目不要求硬刚体来排除这一先例。
3. VGPL已经连接视觉输入与对象归属三维粒子未来预测；“从视频预测”本身也不足以完全避开已有思想。
4. 本轮未确认有工作实现本项目尚待确定的具体共享网络。这个结论不证明无人做过；结构未定型也无法进行逐模块同一性判定。
5. 研究问题仍有价值，独立创新需要具体的共享机制与收益证据。“同物体点应共享信息”“增加物体节点”“全局＋局部”本身均不宜作为独创贡献。更换骨干或数据集不会自动产生机制创新。

## 输出与后续状态

- 新记录：本文件。
- 更新：`reports/progress_log.md`；前一轮分类记录增加完成核查链接，保留原记录。
- 指标与Transformer对照保持待讨论，未追加实验或选定架构。
