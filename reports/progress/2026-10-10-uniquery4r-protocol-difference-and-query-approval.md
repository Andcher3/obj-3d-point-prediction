# 当前模拟公开评测协议的变化与query确认

日期：2026-10-10

## Material Passport

- Origin Skill: academic-research-suite（协议证据核查）；research-progress-logger（工作记录）
- Origin Mode: protocol comparison, read-only code inspection
- Verification Status: ANALYZED（配置与代码比较，无新模型推理或性能结果）

## 已完成

用户确认当前query取样没有问题。记录对18场景新采样预览的批准；GPU授权保持false，未修改输入、GT或预测数组，未连接远端或启动实验。

对照原`uniquery4r_movif_experiment1_2026-10-09.json`与当前`uniquery4r_movif_density_query6_2026-10-10.json`，核查实际采样、评分代码和准备JSON。

| 项目 | 原始实验1 | 当前准备的比较规则 |
| --- | --- | --- |
| 采点 | mask内收缩2px、最远点采样48点 | 约1000点/整幅图的固定间距网格，原图8.095px；网格分支不收缩mask。有效点不足48精确沿用原48点，保留其旧收缩方式 |
| 点数 | 每场景固定48 | 13场景50–134点，5场景原48，共1280 |
| 动态点 | 选择观测期发生运动的物体，未逐点用完整GT动态阈值筛选 | 累计GT三维位移长度>0.01场景单位。当前1280点全部满足，未因此排除点 |
| 主评分尺度 | t0点对长度比中位数，只用t0确定整体尺度 | 按公开评分对动态子集计算整段GT/预测点到原点距离的中位数之比；整段共用一个尺度。之前已在旧缓存上重算过类似公式，该变化并非本轮首次出现 |
| 主评分指标 | 未来点间三维向量/bbox误差、点间长度相对误差 | 加入并用于论文风格比较的EPE与四阈值平均命中率；原关系指标保留作方法诊断 |
| 时间评分 | 重点报告未来15帧，也有观测与t0辅助结果 | 论文风格主评分使用全部22帧；未来15帧仍可另报 |
| 遮挡 | 已保留隐藏点的GT，并报告可见/遮挡分组 | 继续保留后续遮挡点。该项原先已具备，不能写为新修改 |

模型、权重、输入RGB内容、22帧时间索引、第6帧query、256→252像素处理、query归一化、真值来源和首帧相机坐标均不变。取点变化可能改变误差统计的点分布，评分尺度变化可能改变指标数值；两者均未构成模型改善证据。

当前仍有差别：MOVi-F与论文WorldTrack不同、22/64帧不同、query6/query0不同、指定单物体点集与基准标注点集不同、场景单位与已认证米制不同、18场景等权与四数据集等权不同；作者完整评测入口未公开。采样密度是依据图像大小和约1000点推算的近似，不能写为已复用作者原始点集。

本轮没有将历史24帧/query0/逐像素稠密候选作为当前实际协议。旧48点缓存用论文尺度的CPU重算已完成；新采样点集尚未模型推理。

## 文件与来源

- 批准记录：`experiences_results/uniquery4r-movif-density-query6-2026-10-10/query-approval.json`
- 当前配置：`configs/uniquery4r_movif_density_query6_2026-10-10.json`
- 准备证据：同结果目录`preparation.json`及`logs/official-preprocessing-cpu.json`
- 评分代码：`scripts/uniquery4r_worldtrack_protocol.py`
- [公开WorldTrack评分实现](https://github.com/HavenFeng/St4RTrack/blob/0f9a3f44a7ebac76600cd31ec9eea5228ad7db91/dust3r/track_eval_util.py)
- [UniQuery4R论文](https://arxiv.org/html/2608.17283v1)

状态：query已获批准，等待用户明确开启GPU；本轮仅核查与记录。
