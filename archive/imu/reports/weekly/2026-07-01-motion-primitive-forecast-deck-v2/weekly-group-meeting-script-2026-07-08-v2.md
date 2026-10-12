# 2026-07-08 组会汇报发言稿 v2

说明：页码只用于和 PPT 对齐，不建议在发言中读出。正文按连续口播组织，PPT 负责把数据流、模型结构和实验信息可视化。括号中的“备答”内容只在被问到时使用。

## Page 1

各位老师同学好，这次汇报围绕手机视频场景下的移动端行为识别展开。这里的核心问题是：只使用手机 IMU，也就是加速度计和陀螺仪信号，去同时理解两类状态，一类是使用者的人体动作，另一类是手机或相机本身的运镜行为。

传统 HAR 主要回答“人正在做什么”，但手机视频场景里，画面变化往往来自人体行为和手机运动的共同作用。例如画面中的晃动，可能是人在走路，也可能是手部在摇动，或者两种运动同时存在。因此我目前关注的是联合建模、识别和预测这两条运动流。

这个任务希望输出的不只是当前类别，还包括未来一段时间的延续趋势和切换趋势。这样可以服务两类场景：在视频缓存模式中，可以利用更多 IMU 片段提高识别精度；在实时模式中，则需要只根据历史信息进行推理，尽量提前判断行为是否会发生变化。

## Page 2

任务输入是一段六轴 Smartphone IMU 序列，包括三轴加速度和三轴角速度。模型只接收过去到当前时刻 t 的历史 IMU，输出从 t 到 t+n 的四条状态序列，分别是人体动作、人体阶段、相机运动和相机阶段。

人体动作可以包含 standing、walking、sitting、turning 这类类别；相机运动可以包含 static、pan、tilt、roll、shake 或 reposition。这里相机运动先定义为 dominant-motion single-label，也就是每个时间片只给一个主导运动类别。phase 只区分 stable 和 transition，用来描述当前是否处在动作或相机状态的切换过程中。

当前状态直接从历史编码后的 g_t 读取。未来状态则由模型根据历史上下文和 future queries 预测出未来上下文状态，再接同一组分类头得到。切换时间采用一个直接的定义：未来序列里第一个不同于当前标签的位置，就是第一次切换位置；再乘以 patch duration，就得到 time-to-switch。

（备答：当前状态识别不使用未来平滑结果，避免未来预测反向影响当前实时识别。平滑只用于趋势展示和切换时机读取。）

## Page 3

数据形式上，原始输入是一条连续六通道 IMU 流。首先按照固定长度把连续信号切成 patch，每个 patch 表示一个短时间片段。这样历史序列可以表示为 L 个 patch、P 个采样点、C 个通道，其中 C 等于 6。

标签也按照 patch 序列对齐。每个 patch 同时对应人体动作标签、人体 phase 标签、相机运动标签和相机 phase 标签。这样的组织方式有两个好处：当前状态可以从 t 位置读取，未来延续、切换时机和下一状态也可以直接从 t 到 t+n 的标签序列里读取。

这里我把 phase 定义得比较克制，只保留 stable 和 transition。原因是现阶段更关心稳定状态和切换状态的区分，如果把阶段定义得过细，会明显增加标注难度，也可能让模型学习到不稳定的边界。

## Page 4

整体方法可以分成 Student 主路径和 EMA Teacher 预训练路径。Student 在训练和推理时都只使用历史 IMU。历史 IMU 先被切成单通道 patches，再转换成 Local Motion Tokens；之后经过通道独立的因果编码器，建模每个通道自己的时间依赖；最后在同一时间点融合六个通道，得到历史上下文序列 G。

当前状态预测从 g_t 出发，接四个预测头，分别输出当前人体动作、人体 phase、相机 motion 和相机 phase。未来预测使用历史上下文 G 和一组 future queries，通过 Parallel Future Context Predictor 得到 t+1 到 t+n 的未来上下文状态，再接同一组四个预测头，得到未来状态序列。

无标签预训练阶段额外引入 EMA Teacher。Teacher 看到完整的 1 到 t+n IMU，用 EMA backbone 产生真实未来位置对应的上下文目标；Student 只能看到 1 到 t 的历史，并尝试预测这些未来上下文目标。微调和推理阶段 Teacher 都会被丢弃，只保留 Student。

## Page 5

Local Motion Token 的作用，是把每个通道、每个短时间片段转成 Transformer 可以处理的连续表示。当前版本采用连续 local motion token，VQ codebook 和离散 primitive index 只作为文献中的参考设计。

具体做法是：对第 i 个 patch、第 c 个通道的原始波形片段先计算均值和标准差，然后做实例归一化。归一化后的局部波形送入轻量 1D CNN，学习短时间内的局部运动形状。与此同时，均值和标准差被显式保留下来，用于表示幅值偏移和波动强度。

除了局部波形和统计量，token 中还加入通道元信息，也就是 sensor type 和 axis identity，用来区分加速度计、陀螺仪以及 x、y、z 轴；最后再加入 patch 的时间位置编码。经过拼接和投影后，每个单通道 patch 都得到一个统一维度的 local motion token。

（备答：实例归一化有利于减少不同人、不同设备、不同携带方式造成的幅值差异；显式保留均值和标准差，是为了避免归一化把强度信息完全抹掉。）

## Page 6

当前状态预测依赖历史编码。这里采用 channel-independent causal Transformer：把六个通道看作六条单独的时间序列，将通道维并入 batch 维，用共享参数分别建模每个通道的时间动态。

因果注意力保证第 i 个 patch 只能看到 1 到 i 的历史信息，符合实时推理条件。每个通道编码完成后，再恢复通道维度，并在同一时刻把六个通道的表示拼接起来，通过 MLP 做 channel fusion，得到统一的上下文向量 g_i。

这样做的考虑主要有两点。第一，计算复杂度更可控。通道独立处理时，attention 的总复杂度约为 C 乘以 t 平方，比直接把所有通道和时间点拼成一个长序列更稳。第二，物理含义更清楚：单通道内部先学习时间依赖，跨通道关系在同一时间点融合。

当前识别只用 g_t 的 raw head output。也就是说，模型对当前时刻的判断不会依赖未来预测结果，也不会被平滑过程改变。

## Page 7

无标签预训练面向的是这个领域里比较现实的问题：IMU 数据容易连续采集，但精确标注人体动作、相机运动和 transition phase 的成本很高。因此预训练阶段不使用标签，而是让模型先学习历史运动如何约束未来运动上下文。

Student 输入历史 IMU_{1:t}，经过 Local Motion Tokenizer、Shared Causal Encoder 和 Channel Fusion，得到历史上下文 G。随后加入 future queries，通过 Parallel Future Context Predictor 预测未来上下文状态 G_hat。Future queries 是固定 horizon 的未来槽位，它们之间使用非因果 self-attention，可以彼此交互；但 cross-attention 的 memory 只能来自历史 G，因此不会看到真实未来。

Teacher 输入完整 IMU_{1:t+n}。Teacher backbone 是 Student backbone 的 EMA 副本，包含 Local Motion Tokenizer、Shared Causal Encoder 和 Channel Fusion。Teacher 输出最终层融合后的上下文状态，取 t+1 到 t+n 作为目标，并通过 stop-gradient 固定下来。

训练损失是 future context alignment：让 Student 预测的未来上下文状态接近 Teacher 产生的未来目标。目标侧经过无仿射 LayerNorm，主要是降低尺度差异对余弦相似度的影响。EMA 更新只作用在 backbone 参数上，Future Context Predictor 不进入 Teacher。

（备答：这里没有采用多层 hidden states 平均，也没有采用 data2vec 式多层目标聚合；当前目标固定为 Teacher 最终层的 fused contextual states。）

## Page 8

有标签微调阶段使用预训练后的 Student backbone。当前分支从 g_t 输出四个当前标签；未来分支使用历史上下文和 future queries 得到未来上下文状态，再用同一组四个预测头输出未来标签。这样 t 到 t+n 的人体动作、人体 phase、相机 motion 和相机 phase 都能由同一套读出方式得到。

监督损失由四个交叉熵组成，分别对应人体动作、人体 phase、相机 motion 和相机 phase。当前设定中，预训练只优化 future context alignment，微调只优化有标签监督损失；EMA Teacher 在微调结束后不参与推理。

推理时，当前状态直接读取 t 位置 raw output。未来延续趋势则看未来预测标签是否保持当前类别；切换时机看平滑后的预测序列里第一个不同于当前标签的位置；下一状态就是第一次变化后的新标签。平滑只用于减少孤立错误 patch 对趋势判断的影响。

（备答：如果未来序列里没有出现不同标签，可以把该窗口解释为当前状态在预测 horizon 内保持延续。）

## Page 9

支撑这个思路的文献主要有三篇。第一篇是 MoPFormer，它关注 wearable-sensor HAR 中的 motion primitive 表示。它的出发点是，IMU 是连续数值序列，直接解释比较困难，而且不同数据集的传感器位置、通道和采样设置差异明显；很多活动又可以看作一些局部运动模式的组合。

MoPFormer 将多通道 IMU window 划分成更短的 channel-wise segments，做实例归一化后通过 VQ codebook 得到 primitive token，再叠加统计特征、传感器元信息和位置编码，输入 Transformer。训练上，它先做 masked primitive reconstruction，再在下游 HAR 数据集上微调。

这篇工作的价值在于，它把 IMU 的局部运动片段显式表示成 token，并且用无标签预训练提升迁移能力。当前方案保留了“局部运动单元”和“统计量显式保留”的思想，但没有沿用离散 VQ codebook。因为当前目标更偏未来上下文预测，连续表示与 EMA teacher-student 对齐更直接。

## Page 10

第二篇是 P2LHAP。它把 wearable sensor 数据组织成 patch-level label sequence，并在同一个框架里处理 HAR 识别、HAS 分割和 HAF 未来预测。它的关键特点是输出一整段按时间排列的标签序列，而不是停留在单个窗口的分类结果。

在数据管线上，P2LHAP 先对多通道传感器流做通道处理、归一化和 patch 划分，再用共享线性投影、位置编码、通道独立 Transformer encoder 和 decoder 输出标签序列。它还引入 label smoothing 和 boundary refinement，用来减少孤立错误 patch，使活动边界更稳定。

这篇工作对当前方案的启发主要在输出形式上。transition、未来活动和切换边界可以从 label sequence 中自然读取，这比单窗口分类更适合描述行为延续和状态切换。当前方案同样采用 patch-level 序列输出，但进一步加入无标签 future context distillation，希望在标注不足时先学习 IMU 的未来动态结构。

## Page 11

第三篇是 FUTR，它研究视频中的 long-term action anticipation。它给定已经观察到的视频片段，预测未来较长时间内的动作序列。它的关键设计是 fixed ordered queries：每个 query 对应未来序列中的一个预测槽位，并行输出未来动作标签和持续时间。

这种并行预测方式避免了自回归逐步生成中的误差累积，也允许未来 query 之间进行双向交互，从而建模未来序列的整体结构。实验上，它相比自回归版本速度更快，长期预测效果也更稳定。

当前方案中的 Future Context Predictor 借鉴了这一点。Future queries 被看作未来 horizon 的 latent slots，因此 query self-attention 采用非因果形式；真正限制信息来源的是 cross-attention，它只能关注历史上下文 G。这样模型可以并行预测未来上下文状态，同时满足只用历史信息推理的要求。

综合来看，目前这套方案的主线是：输入端用连续 Local Motion Token 表示短时 IMU 运动单元；历史主干用通道独立因果编码和同一时刻通道融合；无标签阶段用 EMA Teacher 做未来上下文蒸馏；有标签阶段用四个任务头输出人体动作、人体阶段、相机运动和相机阶段。接下来需要进一步确认数据和标签来源，并做一个最小实现来验证这个思路是否可行。
