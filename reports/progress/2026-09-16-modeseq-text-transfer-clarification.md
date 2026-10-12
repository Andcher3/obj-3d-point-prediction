# ModeSeq 文本迁移机制澄清

日期：2026-09-16

> 2026-09-18 更新：下文对 ModeSeq 原机制的核查仍成立；本项目的具体迁移现改为使用实际生成的 narration token/hidden states 作为后续分支 memory，以消除独立 mode embedding 与随机文本可能不一致的问题。见[后续澄清](2026-09-18-token-memory-modality-balancing-and-cascade-training.md)。

## 完成工作

- 对照 ModeSeq 3.3--3.6 节及附录的 Match 定义，复核跨分支 memory、EMTA、回退选择、置信度监督和未选中分支处理。
- 修正完整方法记录中“语义摘要”“关系类别”和“未匹配分支仍待设计”等过宽表述。

## 关键结论

ModeSeq 在第 `k` 步将此前生成的连续 mode embeddings
`Omega[k-1]=[m[1],...,m[k-1]]` 作为 Memory Transformer 的 key/value；当前 mode query 读取该 memory，再读取场景上下文。原方法不把已经解码的轨迹或轨迹摘要直接提供给后续分支。

对应到第一阶段，前序 narration 的生成条件 `m[k]` 进入后续分支 memory。它可以作为 VLM 的 soft prefix 或 cross-attention 条件。另行生成语义摘要、拼接原始 narration token、或重新编码实际输出文本都属于可选文本扩展，不是 ModeSeq 的必要步骤。

文本版 EMTA 固定为：

```text
s[k] = min(P(N_hat[k] => N_GT), P(N_GT => N_hat[k]))
G = {k | s[k] >= tau}
k* = min(G)          if G is nonempty
     argmax_k s[k]   otherwise
```

只有 `k*` 接受 GT narration 的 teacher-forced token loss；`k*` 的置信度标签为 1，其余分支默认为 0，并使用 Binary Focal Loss。若多个候选都匹配 GT，最早匹配者为正分支，后续匹配者仍为负置信度样本。原文另有忽略正分支之前低质量模式置信度损失的变体，主要用于没有 mode rearrangement 的设置；它不是默认规则。

双向蕴含替换的是原 ModeSeq 中“候选与 GT 的 Match 判定”和无匹配时的候选--GT 距离。原文没有显式的候选间 pairwise distance、margin 或 repulsion loss。模式互补性由连续 mode memory、EMTA 的 earliest-match 标签分配、多层置信度重排和共享训练间接形成。候选--候选 NLI 可以统计语义重复；若用于显式排斥，需要作为新增损失单独定义。

单 GT 训练已经由 ModeSeq 的分配规则覆盖，不构成第一阶段的结构缺口。它仍不为未发生的替代未来提供独立正标签，因此模式覆盖和额外候选合理性属于实验评价边界。

## 本地输出

- 更新 `reports/progress/2026-09-16-deterministic-stage2-complete-method.md`。
- 新增本澄清记录并更新 `reports/progress_log.md`。

## 下一步

在选定 VLM 后确定 `m[k]` 的具体张量位置、soft-prefix/cross-attention 接口、NLI 模型及阈值；基础版本先沿用 ModeSeq 默认标签规则，不额外加入候选间排斥损失。
