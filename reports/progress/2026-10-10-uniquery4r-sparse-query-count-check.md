# 稀疏query数量与当前MOVi-F点集核查

日期：2026-10-10

使用academic-research-suite区分官方数据准备规则与本地实现，research-progress-logger保存记录。本轮只读核查，没有远端连接、GPU、模型推理或实验代码修改。

## 证据与结论

- WorldTrack原始论文补充材料C.1的数据准备说明：query均匀下采样至每序列约1,000个。其后首帧可见、画面内及动态点筛选可能进一步改变实际评分点数。本轮未读取原始200个文件，因此没有逐序列的最终query数统计。
- 当前MOVi-F准备JSON复算：18段合计71,111点，范围860–7,697，平均3,950.61点/段。这些点覆盖指定目标物体的每个可见表面像素，没有空间下采样，属于目标区域稠密点集。
- 官方约1,000点是序列级点集，不等于每个物体固定1,000点；当前全部点集中在一个物体，空间分布亦有区别。即使某段总数接近1,000，也不能声称其采样方式相同。
- 上轮将“全部已有稀疏标注点”替换为“目标掩码每个像素”，两种点集并不等价。当前只能确认评分函数及部分筛选逻辑已同步；不能宣称官方稀疏点集采样规则已同步。
- 没有证据证明当前稠密点集能解释原精度差距，亦没有新的推理结果。本轮未改变现有准备文件。

## 来源和限制

- [St4RTrack ICCV2025补充材料C.1](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Feng_St4RTrack_Simultaneous_4D_ICCV_2025_supplemental.pdf)：官方原始资料的网页搜索索引直接提供约1,000点及均匀下采样段落。直接PDF访问返回403，未据此宣称已完整读取PDF。
- [作者发布的完整论文](https://st4rtrack.github.io/files/St4RTrack_paper.pdf)：同一说明在该版本B.1中，搜索索引可读；直接读取因文件过大失败。
- [UniQuery4R论文](https://arxiv.org/html/2608.17283v1)：采用WorldTrack评测；未找到为MOVi-F指定每物体全部像素的评测规则。
- 本地计数：`experiences_results/uniquery4r-movif-protocol-sync-2026-10-10/preparation.json`。

下一步状态：本次数量与采样定义核查完成，实验仍未启动；采样方案未在本轮擅自修改。
