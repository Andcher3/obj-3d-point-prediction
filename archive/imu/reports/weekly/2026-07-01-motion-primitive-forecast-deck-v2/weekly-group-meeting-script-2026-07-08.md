# Weekly Group Meeting Script - 2026-07-08

说明：页码只用于和 PPT 对齐，不建议在发言中读出。正文按连续口播组织，PPT 只承担图示和信息补充。

## Page 1

各位老师同学好，这次汇报围绕手机视频场景下的移动端行为识别。研究目标是只使用手机 IMU 中的加速度计和陀螺仪信号，同时理解两类状态：使用者的人体动作，以及手机或相机本身的运镜行为。

传统 HAR 更多回答“人正在做什么”，但在手机拍摄场景里，画面变化同时来自人体移动和手持设备运动。例如同样是画面晃动，可能是人在走，也可能是手在摇，或者两者同时发生。因此这里需要把人体行为和相机运动放在一个联合建模框架里。

这个任务的输出也不只限于当前分类，还需要估计未来一段时间内行为是否延续、何时切换、下一状态可能是什么。这样才能服务于视频缓存模式和实时模式：缓存模式可以利用更多 IMU 信息提高识别精度，实时模式则只能基于历史信息预测未来趋势，同时满足移动端轻量部署要求。

## Page 2

任务输入是一段六轴 Smartphone IMU 序列，包括三轴加速度和三轴角速度。历史序列先被划分成按时间排列的 patches，模型需要输出从当前时刻 t 到未来 t+n 的四条状态序列：人体动作、人体阶段、相机运动、相机阶段。

人体动作和相机运动都是离散类别。人体动作可以包括站立、行走、坐下、转身等；相机运动可以包括静止、平移、俯仰、滚转、抖动和重新构图。phase 只保留 stable 和 transition，用来描述状态是否处于稳定延续或切换过程。

当前时刻的识别直接来自历史编码后的 g_t。未来 t+1 到 t+n 的状态来自模型预测出的未来上下文状态。切换时机采用一个直接定义：预测序列中第一个与当前标签不同的位置就是切换位置，再乘以 patch 时长，得到 time-to-switch。这样输出序列同时支持当前识别、延续趋势、切换时机和下一状态读取。

## Page 3

数据形式上，原始输入是连续六通道 IMU 流。模型先按固定长度切成 patch，每个 patch 对应一个短时间片段；后续的人体动作、人体 phase、相机运动和相机 phase 都按 patch 对齐。

这种形式的好处是，模型输出天然是时间序列。当前状态从 t 位置读取；延续趋势看未来标签是否保持不变；切换时机看平滑后序列第一次变化；下一状态就是第一次变化后的 label。

这里最关键的数据要求是连续时间对齐。human 和 camera 两条流需要分别标注，phase 只区分 stable 与 transition，避免把阶段定义做得过细，增加不必要的标注难度。

## Page 4

整体结构分为 Student 主路径和 EMA Teacher 预训练路径。Student 只看到历史 IMU：从 raw IMU 到单通道 patches，再转成 Local Motion Tokens。随后用通道独立的因果编码器处理每个通道的时间依赖，并在同一时刻融合六个通道，得到历史上下文序列 G。

当前识别接在 g_t 上，输出四个当前状态头。未来预测使用历史上下文 G 加一组 future queries，通过并行 Future Context Predictor 得到未来上下文状态，再接同一组四个预测头，输出未来状态序列。

无标签阶段引入 EMA Teacher。Teacher 看到完整的 1 到 t+n IMU，用 EMA backbone 生成真实未来对应的上下文目标；Student 只用 1 到 t 的历史预测未来上下文，并对齐 Teacher 的 stop-gradient target。微调阶段只使用有标签交叉熵。推理阶段丢弃 Teacher，只保留 Student。

## Page 5

Local Motion Token 的作用，是把每个通道、每个短时间片段变成可供 Transformer 处理的连续表示。当前版本采用连续 token，去掉了 VQ codebook 和离散 primitive index。

每个 token 包含四类信息。第一类是归一化后的局部波形表示，由轻量 1D CNN 学习，主要表达局部运动形状。第二类是均值和标准差，用来保留实例归一化移除掉的幅值偏移和波动强度。第三类是 sensor 和 axis 身份信息，用 one-hot 区分加速度、角速度以及 x、y、z 轴。第四类是 patch 的时间位置编码。

这样的设计让 token 同时保留“形状、强度、通道身份、时间位置”四类信息。后续 Transformer 处理的是短时间运动单元序列。

## Page 6

历史编码采用 channel-independent causal Transformer。实现上，把六个通道看作六条独立时间序列，将通道维并入 batch 维，共享同一个因果编码器参数。每个通道内部只能访问当前及过去 patches，因此满足实时推理约束。

编码完成后，再恢复通道维，并在同一时间点融合六个通道，形成统一的上下文向量 g_i。这样做主要有两个考虑。第一，复杂度更可控：通道独立处理后，总复杂度约为 C 乘以 t 平方。第二，物理含义更清晰：通道身份已经保留在 token 中，跨通道关系在 same-time fusion 阶段再融合。

当前状态预测只使用 g_t 的 raw head output。未来预测和平滑不会反向影响当前识别，这样可以减少延迟，也避免未来序列中的修正改变当前判断。

## Page 7

无标签预训练针对的是连续 IMU 数据“易采集、难标注”的问题。预训练不依赖动作标签，训练目标是让 Student 从历史 IMU 中预测未来上下文表示。

Student 输入 1 到 t 的历史 IMU，经过 Local Motion Tokenizer、Causal Encoder 和 Channel Fusion 得到历史上下文 G，再由 Future Context Predictor 输出 t+1 到 t+n 的未来上下文状态。Teacher 输入完整的 1 到 t+n IMU，用与 Student backbone 同构的 EMA 副本生成未来目标。

这里需要强调 Teacher 的范围。Teacher backbone 包含 Local Motion Tokenizer、Shared Causal Encoder 和 Channel Fusion；EMA 更新只发生在这些 backbone 参数上。Future Context Predictor 属于 Student 的预测分支，不进入 Teacher。

损失使用余弦对齐：Student 的预测未来状态与 Teacher 的 stop-gradient target 对齐，Teacher 目标经过无仿射 LayerNorm，降低尺度差异带来的干扰。这个预训练目标希望模型先学到“历史如何决定未来运动上下文”，再进入有标签任务微调。

## Page 8

有标签阶段使用预训练后的 Student backbone。当前分支从 g_t 输出人体动作、人体 phase、相机运动和相机 phase；未来分支用 future queries 和历史上下文生成未来状态，再用同一组四个 head 输出 t 到 t+n 的四条序列。

监督损失由四个交叉熵组成，分别对应人体动作、人体 phase、相机 motion 和相机 phase。这里预训练和微调目标保持分离：预训练只优化 future context alignment，微调只优化有标签监督损失，EMA Teacher 在微调和推理阶段都不再使用。

推理读数分两类。当前状态直接读 t 位置的 raw output。未来趋势、切换时机和下一状态从预测序列中读取。为了减少孤立错误标签，可以只在趋势展示和切换读取时做局部平滑；这个平滑不参与当前识别。

## Page 9

后面三篇文献是这套设计的主要参考。MoPFormer 的价值在于，它明确把 IMU 中的局部运动模式建模为 motion primitive。它将多通道 IMU window 划分为更短的 channel-wise segments，做实例归一化后通过 VQ codebook 得到 primitive token，再叠加统计特征、传感器元信息和位置编码，输入 Transformer 做分类和 masked primitive reconstruction。

它带来的启发有两点。第一，局部运动单元比直接输入整段连续波形更可解释，也更适合迁移到不同传感器数据。第二，无标签预训练能够提升下游 HAR 表现。

我们的方案保留了“局部运动单元”和“统计量显式保留”的思想，但没有沿用离散 VQ codebook。当前目标需要预测未来上下文状态，连续 Local Motion Token 更适合做 EMA teacher-student 对齐。

## Page 10

P2LHAP 的启发主要来自任务形式。它将传感器序列组织为 patch-level label sequence，用统一框架处理识别、分割和未来预测。模型先对多通道传感器流做通道拆分、归一化和 patch 划分，再用共享投影、位置编码、通道独立 Transformer encoder 和 Transformer decoder 输出标签序列。

这篇工作的价值在于，transition 和 future activity 可以自然地从 label sequence 中读出。它还使用 label smoothing 和 boundary refinement 来减少孤立错误 patch，使边界更平滑。

我们的输出形式与它接近，同样关注 patch-level 的未来状态序列。但 P2LHAP 更偏 supervised patch-to-label，我们的方案进一步加入无标签 future context distillation，希望在标签不足时先从大量 IMU 序列中学习未来动态结构。

## Page 11

FUTR 参考的是 future action anticipation。它给定已观察到的视频片段，预测未来较长时间的动作序列。关键点是用 fixed ordered queries 并行预测未来动作和持续时间，避免自回归逐步生成。

这种并行 query 设计有两个优势。第一，可以避免自回归误差逐步累积。第二，future queries 之间可以相互建模，获得更完整的未来序列结构。我们的 Future Context Predictor 采用类似思想：future queries 是未来 horizon 的 latent slots，因此 query self-attention 使用 non-causal；真正的信息约束来自 cross-attention 只能访问历史上下文 G。

从这三篇文献回到当前方案，可以看到方法的主线已经比较明确：输入端用连续 Local Motion Token 表示短时 IMU 运动单元；历史主干用通道独立因果编码和同一时刻通道融合；无标签阶段用 EMA Teacher 做未来上下文蒸馏；有标签阶段用四个任务头输出人体动作、人体阶段、相机运动和相机阶段。后续实验需要进一步确认连续 IMU、双流标签和 phase 标签的可获得性，再做最小实现和验证。
