# 2026-07-08 组会汇报发言稿 v3（5 分钟版）

说明：页码只用于排练时和 PPT 对齐，不建议读出。“主讲”用于 5 分钟汇报；“备答”只在老师追问时补充。

## Page 1

主讲：
各位老师同学好，这次汇报的主题是面向手机视频的移动端行为识别。这里希望只使用手机 IMU，同时理解人体动作和手机相机运动。手持拍摄中，画面变化会同时受到人走路、身体转向、手部晃动和运镜影响，所以需要把这两类运动放在同一个时序问题中建模。

备答：
这个方向后续可以服务视频防抖、智能拍摄、运动状态感知和移动端实时理解等场景。

## Page 2

主讲：
因此任务输入是一段历史六轴 IMU，包括三轴加速度和三轴角速度。期望输出是从当前时刻到未来一小段时间的状态序列，其中包含人体动作、人体阶段、相机运动和相机阶段。这样既能回答“现在是什么”，也能进一步描述当前状态是否会延续、是否快要切换，以及切换后更可能进入什么状态。

备答：
人体和相机各自有 action / motion 标签，也各自有 stable / transition 阶段标签。切换时机可以从未来标签序列第一次变化的位置读取。

## Page 3

主讲：
数据形式上，连续 IMU 会被切成固定长度的 patches。每个 patch 对应一个短时间片段，标签也按 patch 对齐。这样输入和输出都变成按时间排列的序列，动作延续、状态切换和边界变化都可以直接在序列中表达。

备答：
输入可记为 `X_{1:t} ∈ R^{L×P×C}`，其中 `L` 是 patch 数，`P` 是每个 patch 内采样点数，`C=6`。

## Page 4

主讲：
整体结构包含 Student 主路径和 EMA Teacher 预训练路径。Student 只看历史 IMU，先构造 Local Motion Token，再经过因果编码和通道融合得到历史上下文。当前状态从最后一个历史上下文读出；未来状态通过 future queries 预测未来上下文后读出。

预训练时，Teacher 看到完整的历史加未来 IMU，产生未来上下文目标；Student 只看历史，学习预测这些目标。微调和推理时只保留 Student。

备答：
Teacher 是 Student backbone 的 EMA 副本，包含 Local Motion Tokenizer、Causal Encoder 和 Channel Fusion，不包含 Future Context Predictor。

## Page 5

主讲：
Local Motion Token 用来把每个单通道短片段变成连续表示。它包含三类关键信息：局部波形形状、均值和标准差、通道身份与时间位置。局部形状由轻量 1D CNN 从归一化片段中提取；均值和标准差保留幅值偏移和波动强度；通道与时间信息帮助模型区分传感器、坐标轴和 patch 顺序。

这样做的动机是，IMU 的局部形状、幅值强度、物理通道和时间顺序都对识别有用，需要在 token 中一起保留。

备答：
当前版本不使用 VQ/codebook，也不使用离散 primitive index。公式可写成 `z_{i,c}=f_fuse([f_motion(x^{norm}_{i,c}); μ_{i,c}; σ_{i,c}; m_c])+p_i`。

## Page 6

主讲：
历史编码采用 channel-independent causal Transformer。每个通道先独立建模自己的时间变化，并且只能看当前和过去，满足实时推理条件；之后在同一时间点融合六个通道，得到统一上下文向量。当前识别从 `g_t` 接四个分类头，输出人体动作、人体阶段、相机运动和相机阶段。

这种结构可以先学清楚单通道动态，再融合跨通道关系，同时降低注意力计算压力。

备答：
通道独立处理的 attention 复杂度约为 `O(C·t²)`；直接拼接通道时会接近 `O((Ct)²)`。

## Page 7

主讲：
无标签预训练的核心，是让模型先学习“历史 IMU 如何暗示未来运动上下文”。Student 只输入 `IMU_{1:t}` 并行预测未来上下文；Teacher 输入完整 `IMU_{1:t+n}`，用 EMA backbone 得到真实未来片段的上下文表示，作为训练目标。

损失采用上下文对齐，让 Student 预测的未来上下文接近 Teacher 目标。这样可以利用大量无标签 IMU 数据，先学习未来运动趋势，再进入有标签微调。

备答：
Teacher target 取最终层 channel fusion 后的 fused contextual states；target 侧 stop-gradient，并使用无仿射 LayerNorm。Future queries 的 self-attention 是 non-causal，但 cross-attention 只能看历史上下文 `G^S_{1:t}`。

## Page 8

主讲：
有标签微调阶段使用预训练后的 Student。当前分支从 `g_t` 输出当前四个标签；未来分支从预测的未来上下文输出未来四条标签序列。训练时用四个交叉熵分别监督人体动作、人体阶段、相机运动和相机阶段。

推理时，当前识别直接读当前输出；未来延续和切换趋势从预测标签序列中读取。局部平滑只用于趋势展示和切换时机读取。

备答：
平滑不反向影响当前状态识别。当前状态使用 raw head output at `t`。

## Page 9

主讲：
第一篇参考文献是 MoPFormer。它把 IMU 中短时间的局部运动模式表示成 motion primitive token，再结合统计信息、传感器元信息和位置编码输入 Transformer。它对当前设计的启发是：先形成局部运动单元，会比直接处理整段连续波形更容易建模。

备答：
MoPFormer 使用 VQ codebook 得到离散 primitive token，并通过 masked primitive reconstruction 做预训练；当前方法只保留局部运动单元思想，表示形式改为连续 token。

## Page 10

主讲：
第二篇是 P2LHAP。它把传感器识别、分割和未来预测统一成 patch-level label sequence 输出。这个思路对当前任务很重要，因为窗口分类、未来标签延续和边界变化都可以放在同一种序列形式中处理。

备答：
P2LHAP 更偏有监督 patch-to-label；当前方法加入了无标签 future context distillation，目的是缓解标注不足。

## Page 11

主讲：
第三篇是 FUTR。它在视频动作预测中使用 fixed ordered queries 并行预测未来动作序列。当前方法中的 future queries 借鉴了这个思路，每个 query 对应未来 horizon 中的一个预测位置。

串起来看，当前方法就是用 Local Motion Token 表示短时 IMU 运动，用因果编码和通道融合建立历史上下文，再通过 EMA Teacher 学习未来上下文，最后读出人体和相机两条状态序列。

备答：
当前 future queries 属于并行 horizon slots，不按自回归方式逐步生成；信息约束来自 cross-attention 只能访问历史上下文。
