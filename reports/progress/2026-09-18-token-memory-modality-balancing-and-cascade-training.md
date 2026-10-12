# Token 候选记忆、模态平衡与级联训练澄清

日期：2026-09-18

## 完成工作

- 修正当前方法的概率定位：第一阶段不输出 narration 的真实发生概率，当前假设是固定候选预算下改善罕见模式和长尾事件覆盖。
- 将第一阶段跨分支条件从独立 mode embedding 改为实际生成的 narration token/hidden states。
- 调研 ModDrop、Gradient Blending 和 OGM-GE，确定视觉削弱和优化平衡可作为第二阶段训练增强。
- 明确先冻结第一阶段、再使用其预测 narration 训练第二阶段时的语义配对规则。

## 第一阶段 token memory

```text
V_obs + target
  -> <ALT_1> N_1 </ALT_1>
  -> <ALT_2> N_2 </ALT_2> | N_1
  -> ...
```

前序 token 与实际输出严格对应；候选结束位置的 hidden state 可接置信度头。训练选中分支时，使用前面实际采样候选作为固定上下文，再对 `N_GT` 做 teacher forcing。后续分支通过共享参数和在其他样本上被 EMTA 选中获得文本监督。该设计是 ModeSeq-inspired 文本迁移，不能声称复制了原论文的连续轨迹 mode embedding。

需要防止模型将多个候选理解成动作时间序列；每个候选必须标明是从同一观察截止点出发的替代未来。长上下文、复制前序文本和离散 token 路径不可微是实现代价。

## 长尾定位

顺序生成和 EMTA 可能减少高频模式重复占据候选槽，从而提高罕见事件召回。它没有重加权长尾数据，也不能恢复训练和预训练先验均未覆盖的事件。验证应在固定 `S` 下比较独立采样、token memory、token memory+EMTA 的 head/medium/tail recall、GT match@S、unique-mode coverage 和重复率。

## 第二阶段模态平衡候选

- ModDrop 随机丢弃模态通道以学习跨模态相关性并适应缺失信号。
- Gradient Blending 根据模态的过拟合行为混合梯度。
- OGM-GE 根据各模态对目标的贡献差异在线调节优化。

这些工作支持“模态竞争可以通过训练策略缓解”，不能直接证明视频--文本轨迹预测中的效果。当前优先方案是混合完整视觉与视觉削弱样本，保留目标 mask、query/history、pointmap 和近期运动；可附加轨迹--文本匹配损失，用同一真实轨迹的正确 narration 作正对、动作/对象/顺序不同的 narration 作 hard negative。梯度调制保留为更复杂的后续消融。

## 级联训练

1. 完成第一阶段训练并冻结参数与生成配置。
2. 在第二阶段训练集的观察前缀上自由生成 `{N_hat_k}`。
3. 使用双向蕴含与真实 future narration 匹配，选择最早达到阈值的 `N_hat_k*`。
4. 只用 `(V_obs, target/history, N_hat_k*, X_GT)` 训练第二阶段。
5. 无匹配时回退 GT narration 或跳过；不把低分候选和全部替代候选配给同一个 `X_GT`。
6. 推理时对第一阶段的全部候选分别生成确定性轨迹。

该流程减小 GT 文本与预测文本的风格和错误分布差异。真实未来未被第一阶段覆盖、训练/测试泛化和替代候选缺少当前场景轨迹真值仍然存在，必须分别报告 GT narration、matched predicted narration 和完整 predicted candidates。

## 来源

- ModeSeq: https://arxiv.org/html/2411.11911v2
- ModDrop: https://arxiv.org/abs/1501.00102
- Gradient Blending: https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_What_Makes_Training_Multi-Modal_Classification_Networks_Hard_CVPR_2020_paper.html
- OGM-GE: https://openaccess.thecvf.com/content/CVPR2022/html/Peng_Balanced_Multimodal_Learning_via_On-the-Fly_Gradient_Modulation_CVPR_2022_paper.html

## 本地输出

- 更新完整方法和学术审核记录。
- 新增本澄清记录并更新进展索引。

## 下一步

先用顺序提示代理检查 token memory 是否降低重复并提高 tail recall；第二阶段先完成完整视觉文本基线，再逐项加入视觉模态 dropout 和轨迹--文本匹配辅助损失。
