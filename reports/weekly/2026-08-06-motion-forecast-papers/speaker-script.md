# MotionForesight 与 Generative Point Tracking and Forecasting 发言稿

## Page 01｜MotionForesight 封面

这一部分先介绍 MotionForesight。论文研究的问题很直观：只观看一段普通单目视频的前半段，不提供动作名称、语言指令或机器人状态，模型能否提前判断被操作物体接下来会在三维空间中怎样运动。它最终输出物体表面若干点的三维轨迹。论文在 2026 年 7 月发布于 arXiv，目前属于预印本。

[Sources]
- https://arxiv.org/html/2607.16192v1

## Page 02｜MotionForesight：动机与任务定义

作者首先讨论预测结果应该用什么表示。生成未来视频需要同时处理纹理、光照和背景变化，计算量很大；一个刚体的六自由度位姿虽然紧凑，却难以表示抽屉开合、软物体形变和物体局部运动；二维轨迹又缺少真实的空间距离。三维点轨迹场直接回答“物体上每个点之后会去哪里”，能够统一表示平移、旋转、关节运动和局部形变。

[Sources]
- https://arxiv.org/html/2607.16192v1#S1

## Page 03｜MotionForesight：训练监督构建

模型真正看到的是前 7 帧 RGB 和由单目深度得到的 pointmap。SAM 在目标物体上产生掩码，模型从掩码内选择查询点。训练标签来自另一条离线流程：完整 22 帧视频经过掩码传播、深度与相机运动估计，再由 TrackCraft3R 恢复这些点的完整三维轨迹，最后截取未来 15 帧作为伪标签。完整未来只用于制作标签，从未作为预测模型的输入，避免训练时偷看答案。

[Sources]
- https://arxiv.org/html/2607.16192v1#S3.SS1
- https://arxiv.org/html/2607.16192v1#S3.SS2

## Page 04｜MotionForesight：模型结构

TrackCraft3R 原本是回溯式跟踪器，需要看完整视频后解释点已经怎样移动。MotionForesight保留它的两条输入流：一条是随时间变化的 RGB 与 pointmap 条件，另一条反复询问“第一帧的这个点在当前时刻在哪里”。在未来 15 个时间位置上，真实图像和几何被换成可学习的未知标记；时间 RoPE 告诉模型每个位置对应哪个未来时刻。大型视频骨干、原跟踪适配器和编解码器保持冻结，只训练新的 LoRA、输入输出投影、预测头和未知标记。

[Sources]
- https://arxiv.org/html/2607.16192v1#S3.SS3
- https://arxiv.org/html/2607.16192v1#S3.SS4

## Page 05｜MotionForesight：实验结果

训练损失直接比较预测点和伪标签点的三维坐标，并用有效性掩码忽略无效位置。未来部分的权重为 1，观测重建权重为 0.25。模型在 SSv2 未见视频上的 ADE 为 4.47 厘米、FDE 为 6.23 厘米，76% 的预测位于 5 厘米阈值内；在独立手机视频上对应结果为 9.31 厘米、14.88 厘米和 54%。在论文列出的五种对比设置中，MotionForesight 在 SSv2 与手机视频的六项指标上均取得最佳数值。定性结果还覆盖抬起、平移、旋转、受约束滑动和局部非刚性运动，说明冻结视频先验并训练轻量适配模块可以产生明确的三维轨迹输出。

[Sources]
- https://arxiv.org/html/2607.16192v1#S3.SS5
- https://arxiv.org/html/2607.16192v1#S4
- https://arxiv.org/html/2607.16192v1#S5

## Page 06｜Generative Point Tracking and Forecasting 封面

第二篇是 CVPR 2026 的 Generative Point Tracking and Forecasting。作者提出一个统一观点：点跟踪和未来预测都在回答“这个查询点在各个时刻位于哪里”。差别只在于当前时刻有没有图像可以参考。论文因此训练一个因果的视频条件 Flow Matching 模型，让同一网络同时完成二维点跟踪和未来轨迹生成。

[Sources]
- https://openaccess.thecvf.com/content/CVPR2026/papers/Lu_Generative_Point_Tracking_and_Forecasting_CVPR_2026_paper.pdf

## Page 07｜Generative Point Tracking and Forecasting：统一任务定义

这篇工作的第一步，是把点跟踪和未来预测写成同一个条件生成问题。图中的 P 表示二维点轨迹，V 表示点是否可见，Q 是初始查询点，I_C 表示当前能够使用的视频帧。对于已经观察到的时刻，I_C 中存在对应图像，模型可以依据图像匹配点的位置，这就是点跟踪；进入尚未发生的时刻后，未来图像被屏蔽，模型只能根据历史轨迹和学到的运动规律继续生成，这就是未来预测。这样，两项任务的输出完全相同，变化的只有图像条件是否存在。论文进一步学习轨迹的条件分布，因此同一段历史可以采样出多种合理运动。任务被统一以后，接下来需要回答的是：一个模型怎样同时处理点、时间和有时存在、有时缺失的图像条件。

[Sources]
- https://openaccess.thecvf.com/content/CVPR2026/papers/Lu_Generative_Point_Tracking_and_Forecasting_CVPR_2026_paper.pdf

## Page 08｜Generative Point Tracking and Forecasting：模型结构

模型把每个查询点在每个时刻的位置表示成一个轨迹 token。训练输入并非直接使用真实坐标，而是在坐标上加入噪声，再让网络逐步恢复真实轨迹；DINOv3-S 同时从可用视频帧中提取图像特征。图中三种注意力各自解决一种关系：空间注意力让同一时刻的不同点交换信息，因果时间注意力让同一点读取当前和过去的运动，跨模态注意力让点 token 从图像特征中寻找视觉证据。观测时刻提供真实图像特征，未来时刻用 [M] 表示图像条件缺失，因此同一结构可以自然地从跟踪切换到预测。网络最终输出点坐标的速度场和可见性。模型结构说明了信息怎样流动，下一页再看它如何通过训练同时学会三种条件组合，以及怎样生成超过单个窗口长度的未来。

[Sources]
- https://openaccess.thecvf.com/content/CVPR2026/papers/Lu_Generative_Point_Tracking_and_Forecasting_CVPR_2026_paper.pdf

## Page 09｜训练与推理

训练包含三个相互配合的设计。第一，条件 Flow Matching 把高斯噪声与真实坐标沿一条概率路径混合，网络学习指向真实轨迹的速度场；可见性由二元交叉熵单独监督。第二，每条训练样本随机选择视觉截止帧：全程保留图像时训练点跟踪，只保留首帧时训练未来预测，截止位置位于中间时训练先跟踪、再预测。模型由此学会第7页中的不同条件组合。第三，单个窗口长度有限，长序列推理采用重叠滑动窗口：上一窗口末尾的历史轨迹成为下一窗口的前缀，窗口按照步长 S 持续前移，直到覆盖完整时间范围。至此，论文已经给出了从条件分布生成轨迹并延伸到长未来的完整过程，最后需要通过实验检查预测、跟踪和生成式训练是否都有效。

[Sources]
- https://openaccess.thecvf.com/content/CVPR2026/papers/Lu_Generative_Point_Tracking_and_Forecasting_CVPR_2026_paper.pdf

## Page 10｜实验结果

实验按照前面的方法设计分成四部分。首先，Kubric MOVi-A 检查未来轨迹分布：预测专用模型把 Best-of-K 从此前方法的 127.0 降到 95.6，统一模型为 98.6，并取得最低的 LRTL 13.9，说明生成结果与真实运动分布更加接近。其次，Physics101 检查真实物理运动，模型的总体 MSE 为 21.41，对比方法分别为 28.62 和 30.08。第三，TAP-Vid 检查同一网络的跟踪能力，Kinetics 与 DAVIS 上的 AJ 分别为 49.8 和 59.9。最后，消融实验直接比较训练目标，Flow Matching 的 AJ 为 53.7，高于 L1 回归的 43.3 和 L2 回归的 30.2。四组结果分别对应未来分布、真实运动、点跟踪和训练目标，支持论文提出的统一生成式点模型。

[Sources]
- https://openaccess.thecvf.com/content/CVPR2026/papers/Lu_Generative_Point_Tracking_and_Forecasting_CVPR_2026_paper.pdf
