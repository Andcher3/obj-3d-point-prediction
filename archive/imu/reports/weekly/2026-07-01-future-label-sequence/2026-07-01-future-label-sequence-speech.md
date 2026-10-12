# 2026-07-01 组会发言稿：Future Patch Label Sequence 方法设想

预计时长：约 5 分钟

## Slide 1：本周进展

这周我主要从前面阅读的几篇工作出发，开始把方向一的任务形式往一个更具体的方法设想上收束。

原来我们讨论的是手机 IMU 上的人体动作识别和手机运动识别。如果只是做单个窗口的 HAR 分类，输出就是当前窗口属于哪一类动作。但方向一里还有“未来是否延续”“什么时候切换”“下一个动作是什么”这些要求，所以我把问题重新整理成连续 IMU 流上的 future label sequence prediction。

也就是说，本周的重点不是已经完成实验，而是明确一个近期可落地的方法框架：用过去的 IMU patch 序列预测当前和未来一段 patch 的动作标签序列。

## Slide 2：任务定义

这里最关键的一个选择是：我暂时不把 `<sit-2-stand>` 这样的动作切换作为主标签。

这样做的原因是，如果每种动作到另一种动作都单独作为 transition class，类别会随着动作组合数增加，而且这些过渡片段本身很少，容易出现严重样本不平衡。

所以更自然的形式是预测未来标签序列。比如当前是 sit，模型预测未来是 sit、sit、stand、stand，那么第一次从 sit 变成 stand 的位置就是切换时机，stand 就是下一动作。延续概率和切换概率也可以从每个未来位置的分类置信度中计算出来。

这个形式也更接近 P2LHAP、FUTR、ActFusion 这类近期相关工作。

## Slide 3：模型结构

模型结构上，我计划使用 Transformer encoder-decoder，而不是只用一个 MLP 直接输出。

输入是过去 L 个 IMU patches。每个 patch 包含一小段 acc 和 gyro 的多通道时间序列，先经过 patch embedding 变成 token，然后送入 Transformer Encoder 得到历史 IMU 的 memory 表示。

Decoder 部分使用 parallel query decoder。这里有一组 learnable queries：`q0` 表示当前标签，`q1` 到 `qH` 分别表示未来第 1 到第 H 个 patch 的标签。Decoder 对 encoder memory 做 cross-attention，并行输出 `o0` 到 `oH`。最后每个输出 token 通过共享 MLP classifier，得到当前和未来每个 patch 的 label distribution。

选择 parallel query decoder 的原因不是单纯工程方便，而是参考了 FUTR 这类 long-term action anticipation 工作。它们发现并行预测可以避免自回归方式的错误累积，并且推理更快。

## Slide 4：训练策略

训练策略分两阶段。

第一阶段是无标签 IMU 预训练。因为真实带标签数据比较稀缺，但手机 IMU 原始数据相对更容易收集，所以可以先做 self-supervised pretraining。第一版我倾向于 masked patch reconstruction：随机 mask 一些 IMU patch，让 encoder 学习从可见部分重构被遮住的 IMU 信号。

第二阶段是有标签微调。保留预训练好的 encoder，加上 label sequence decoder 和分类头，用连续 HAR 数据训练当前和未来 patch 的标签预测。

需要注意的是，这里图里的 mask 只对应第一阶段的 reconstruction。第二阶段的 label query decoder 采用 parallel query decoder，不使用 causal mask。

## Slide 5：切换输出如何得到

这个图展示的是为什么不需要额外设计很多 transition class。

模型对未来每个 patch 都输出一个 label distribution。假设当前标签是 sit，未来预测前三个 patch 仍然是 sit，第四个 patch 变成 stand，那么：

第一，延续概率可以直接看未来每个 patch 上当前标签 sit 的概率。

第二，切换时机就是第一次 argmax label 不等于当前标签的位置。

第三，下一个动作就是第一次变化后的 label，也就是 stand。

所以“动作是否延续”“什么时候切换”“下一个动作是什么”都来自同一个未来标签序列，而不是额外强行造 transition label。

## Slide 6：下一步实验计划

下一步我想先按三个层次推进。

第一是数据和任务。UCI HAR 这种窗口级数据可以先用来做 encoder 和分类 sanity check，但它不适合真正验证切换预测。真正的 future label sequence prediction 需要连续 HAR 数据，或者后续自采连续手机 IMU 数据。

第二是结构消融。需要比较 early fusion、P2LHAP-style channel independent，以及我更倾向的 hybrid 方案，也就是先做通道独立前端，再做跨通道融合。

第三是训练消融。比较从头训练、masked reconstruction 预训练、future IMU regression 预训练，看无监督预训练是否真的能提升少标签条件下的未来标签预测。

最终方向一还要从单一人体动作标签，扩展到人体动作序列和手机/相机运动序列的联合预测。这样才能更贴近课题里“人和手机动作行为联合识别与预测”的目标。

我的阶段性结论是：目前最合理的近期落地形式不是枚举所有动作切换类别，而是 future patch label sequence。它既能表达当前动作识别，也能自然给出延续概率、切换时机和下一动作。

