# P2LHAP / MoPFormer / FUTR 发言稿

> 说明：这里把用户口中的 PL2HAP 按论文正式名称写作 P2LHAP。

## 1. P2LHAP：Patch-to-Label Seq2Seq Transformer

这篇论文要解决的是可穿戴传感器人体活动感知里一个很实际的问题：识别、分割和预测往往被当成三个任务分开做。识别是判断当前是哪类动作，分割是判断动作从哪里开始、到哪里结束，预测是判断接下来会发生什么。传统做法常常先用滑动窗口切数据，再对每个窗口分类，但这个设定有一个天然问题：真实活动的持续时间并不固定，一个固定长度窗口不一定刚好装下一个完整动作，也可能把两个动作混在一起。所以作者的动机是，能不能不要先把任务拆开，而是把连续传感器流直接变成一个按时间排列的标签序列。

P2LHAP 的核心思想就是 Patch-to-Label。它把连续的多通道传感器数据切成一个个短 patch，每个 patch 作为一个输入 token，然后让 Seq2Seq Transformer 输出 patch 级别的活动标签序列。这样一来，三个任务就统一了：如果看每个 patch 的标签，就是活动识别；如果把相邻同类 patch 合并起来，就得到活动分割边界；如果让模型继续输出未来若干 patch 的标签，就得到活动预测。

方法上，论文没有简单把所有传感器通道混在一起，而是使用 channel-independent Transformer。也就是说，不同传感器通道先相对独立地切 patch 和编码，但共享 embedding 和 Transformer 权重。这样做的理由是，多变量传感器序列里不同通道的物理含义不同，直接混合可能会引入干扰；而共享权重又能控制参数规模，避免每个通道都训练一套大模型。

这篇论文还有一个很重要的后处理设计，就是基于周围 patch 标签的平滑。因为模型逐 patch 输出标签时，可能会出现一长串 stand 中间夹了一个错误的 sit 这种孤立错误。如果直接根据原始预测切边界，就会产生很多碎片化的过分割。P2LHAP 用周围标签来修正这类孤立错误，让边界更稳定。

所以它效果好的原因可以概括为三点：第一，patch 比固定长窗口更灵活，一个动作可以由任意数量 patch 组成；第二，输出本身就是时间顺序标签，因此识别、分割、预测可以从同一个序列里自然读出；第三，channel-independent Transformer 加上标签平滑，分别缓解了多通道干扰和边界抖动。对我们最有启发的是，所谓“动作切换趋势”不一定要单独设计很多切换类别，而可以先预测未来标签序列，再从标签首次变化的位置和置信度里读出切换时机与延续概率。

## 2. MoPFormer：Motion-Primitive Transformer

MoPFormer 关注的是另一个问题：IMU 信号本质上是连续数值序列，虽然模型可以直接输入这些数值做分类，但这样的表示既不直观，也不容易解释；同时，不同数据集的采样率、传感器位置、通道设置都可能不同，导致跨数据集泛化困难。作者的出发点是，人类复杂活动往往可以看成更基础局部运动模式的组合。比如走路、上楼、下楼，都可以拆成一些短时间内反复出现的局部运动片段。于是论文提出，把连续 IMU 片段先离散化成 motion primitives，也就是运动原语，再交给 Transformer 学习它们的组合关系。

具体做法可以分成两层。第一层是 Motion Primitive Module。模型先把多通道 IMU 按通道切成短片段，并做 instance normalization，减少不同个体、设备和幅值尺度带来的影响。然后使用 VQ codebook，把每个短片段量化成一个离散 codeword，也就是一个 motion primitive。同时，模型还保留片段的统计特征，例如均值和方差，因为离散原语主要描述局部形状，而统计特征能补充幅值和分布信息。

第二层是 Context-Aware Embedding Module。每个输入 token 不是单独的 primitive embedding，而是由四部分相加构成：primitive embedding、statistical embedding、sensor metadata embedding 和 positional encoding。这里的 metadata 很重要，它告诉模型这个 token 来自哪个身体部位、哪个传感器、哪一个轴。最后，模型把这些 token 连同 CLS、MASK 等特殊 token 输入 Transformer Encoder。

训练上，MoPFormer 使用 masked motion modeling。也就是随机 mask 一部分 motion primitive，让模型根据上下文重建被遮住的 VQ index。这一点和 BERT 或 masked autoencoder 的思路很接近，但预测对象从文字 token 或图像 patch 变成了 IMU 的运动原语。预训练完成后，再用 CLS token 接分类头，针对具体 HAR 数据集微调。

它的创新点在于，把连续、难解释的 IMU 数值变成了离散、可观察的 motion primitive vocabulary；又通过统计特征和传感器元数据把幅值信息、通道信息补回来；最后用自监督 masked reconstruction 利用大量无标签数据。为什么这样可能有效？因为 motion primitive 把局部运动模式抽象成稳定单元，Transformer 不再直接处理原始噪声很强的连续曲线，而是学习“这些局部运动原语如何组合成一个活动”。这对跨数据集泛化和可解释性都有帮助。对我们来说，MoPFormer 直接支持了“构造 IMU token / motion token”的想法，也说明 future token 不应只包含一个离散原语，还应该带有统计成分、通道元数据和时间位置。

## 3. FUTR：Future Transformer for Long-Term Action Anticipation

FUTR 来自视频领域，任务是 long-term action anticipation，也就是给定已经观察到的一段视频，预测未来较长时间内会发生哪些动作。虽然它不是 IMU 论文，但它和我们讨论的“未来标签序列怎么预测”高度相关。传统方法经常使用自回归解码：先预测第一个未来动作，再把这个预测结果作为条件预测第二个，再预测第三个。这个方式的问题很明显，如果第一步错了，错误会一路传下去；而且它必须一步一步生成，推理速度也慢。

FUTR 的核心动机是，未来动作之间并不是完全独立的，它们有全局关系。比如做饭视频里，拿杯子、倒水、搅拌这些动作之间有顺序和依赖。如果只用自回归方式，后面的未来动作只能看到前面已经生成的结果，很难充分建模整个未来动作序列的整体结构。因此作者提出用一组有序 action queries 并行预测未来动作段。

模型结构上，FUTR 先用 Transformer Encoder 编码已经观察到的视频帧特征，得到 past memory。然后 Decoder 里放入一组 learnable ordered action queries，每个 query 对应未来序列里的一个动作段。与自回归 decoder 不同，这些 query 之间没有 causal mask，因此每个未来 query 可以和其他未来 query 做双向 self-attention；同时它们又通过 cross-attention 读取过去视频的 memory。每个 query 最后接两个输出头：一个预测 future action label，另一个预测 normalized duration，也就是这个动作段持续多久。如果未来动作段数量不足，还可以用 NONE 类表示这个 query 不对应真实动作。

FUTR 的输出不是逐帧标签，而是一个段级序列：每个元素是“动作类别 + 持续时间”。这个设计比直接预测每一帧更紧凑，也更贴合长期动作预测，因为长期预测里精确到每一帧通常很难，先预测未来会发生哪些动作、各自大概持续多久更稳定。论文还显式比较了自回归版本、带 mask 的 query 版本和完整并行版本，结果表明并行预测不但更快，而且准确率更好。

它有效的原因主要有两点。第一，并行解码避免了自回归的错误累积，前一个预测错了不会直接作为下一个预测的输入。第二，不加 causal mask 的 future queries 可以彼此交流，因此模型能学习未来动作之间的整体结构，而不是只从左到右贪心生成。对我们来说，FUTR 给出的启发很明确：如果我们要预测未来一段 IMU token 或未来 patch label，不一定要自回归逐步生成；可以使用 parallel query decoder，一次性预测固定 horizon 内的未来序列，再从这个序列里读取下一动作、延续概率和切换趋势。
