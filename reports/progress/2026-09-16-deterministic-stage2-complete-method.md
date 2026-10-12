# 第一阶段与确定性第二阶段的完整方法版本

日期：2026-09-16

## 方法版本定位

用户决定暂不为第二阶段加入 flow、噪声采样或其他模式内多样性机制。当前版本定义为可训练、可评估的第一版完整流程：第一阶段输出多个未来 narration，第二阶段对每条 narration 生成一条确定性稠密三维轨迹。该版本用于验证语义条件到三维运动的有效性，不宣称已经实现连续概率校准或同一 narration 下的多轨迹分布。

## 研究问题与动机

观察视频往往不足以唯一决定未来：同一物体可能被拿起、放下、移动或继续当前交互。确定性单轨迹回归容易把多个未来平均成一个模糊结果。视频模型还可能过度依赖局部视觉运动，无法利用高层交互意图；文本条件模型也可能出现视频信号压过文字的现象。

本项目希望预测指定物体的未来三维点轨迹，并让高层未来语义控制宏观事件，MotionForesight 的视频与几何先验负责把语义转换为物理细节。直接生成未来 RGB 会引入背景、纹理和光照等无关变量，目标点的稠密三维轨迹更贴近任务接口。

## 总体接口

```text
V_obs + M_target
    -> Stage 1: prospective VLM
    -> {N_1, ..., N_K} 未来 narration 与分支隐藏状态

(V_obs, M_target, query/history, N_k)
    -> Stage 2: text-conditioned MotionForesight
    -> z_track^k
    -> frozen TrackCraft3R decoder
    -> X_future^k 稠密三维点轨迹
```

第一阶段的隐藏状态只负责候选模式的形成、记忆和互补解码；当前第二阶段通过 narration 文本接收语义条件，不直接消费第一阶段的隐藏向量。

## 第一阶段设计

### 目标

从观察视频和目标信息预测未来事件的自然语言概括。输出采用自由长度 narration，可以包含多个有顺序的动作；不固定动作数量，也不要求输出动作类别列表。第一阶段只评价事件级语义，不做三维轨迹解码、轨迹指标或物理可行性判断。

### 模型与适配

- 选择具有世界知识和视频理解能力的 VLM 作为先验。
- 参考 MotionForesight 的前瞻化思想，将回顾式视频理解/总结适配为观察前缀到未来 narration 的预测任务。
- 冻结大部分预训练主体，训练 LoRA、输入输出适配器和必要的模式记忆模块；具体层位和 rank 仍需按候选 VLM 确定。
- 第一版使用实际生成的 narration token 作为跨分支记忆；每条候选结束位置的 VLM 隐藏状态可作为该候选的连续表示和置信度头输入，不再另设可能与采样文本脱节的独立模式摘要。

### 多候选与模式区分

对相同观察条件生成多个未来 narration。第 `k` 个分支读取此前实际生成的 token 序列，并用明确的 `<ALT_i>...</ALT_i>` 边界表示它们是同一观察截止点下的替代未来，避免被理解为同一未来中的连续步骤：

```text
V_obs + target
  -> <ALT_1> N_1 </ALT_1>
  -> <ALT_2> N_2 </ALT_2> conditioned on N_1
  -> ...
```

候选结束 token 的隐藏状态 `h_k` 与实际 `N_k` 一一对应，可作为置信度预测和候选表示。该结构借用 ModeSeq 的“后续模式读取已生成模式”思想，但 memory 由文本 token/hidden states 承担，不声称与原论文的连续 trajectory-mode embedding 完全相同。若后续需要更接近原 ModeSeq，可让新的 branch query 对前序 token hidden states 做 memory cross-attention。

候选与 GT 使用双向蕴含分数：

```text
sim(A, B) = min(P(A => B), P(B => A))
```

定义 `IsMatch[k] = [sim(N_hat[k], N_GT) >= tau]`。若存在匹配，EMTA 选择解码顺序最早的匹配分支；若没有匹配，选择 `sim` 最高的分支，等价于使用 `1-sim` 作回退距离。选中分支接受 GT narration 的 teacher-forced token loss，并得到正置信度标签；其余分支不接受该 GT 的 token loss，并按 ModeSeq 默认规则得到负置信度标签，以 Binary Focal Loss 训练。候选间的双向蕴含可用于重复率评估；将其额外写成分支间排斥损失属于后续扩展，不是 ModeSeq 原机制。

`tau` 的具体数值和 NLI 模型仍需在任务数据上验证。当前基础版本只使用“语义等价/不等价”的二元匹配，不增加蕴含、矛盾、兼容等关系类别。

### 监督

- 观测输入只包含未来开始前的视频、目标信息和可用历史状态。
- 训练真值来自与未来时间窗对齐的 Ego4D narration；时间戳用于确定监督片段，但不自动提供精确动作边界。
- 真实未来不可在预测时输入。完整视频或未来帧只用于离线整理 narration 和第二阶段三维伪标签。
- 单个样本的一条真实 narration 足以执行 ModeSeq 式稀疏监督：按上述 EMTA 规则只对选中分支使用 language-model teacher forcing，其余分支通过置信度标签、顺序 memory 和共享参数受到训练。未发生未来没有独立文本 GT 是监督边界，不再列为第一阶段训练机制尚未解决的问题。

## 第二阶段确定性设计

### 输入与目标

第二阶段输入为观察 RGB 视频、目标 mask 或等价目标指定、参考帧采样的 query points、其历史轨迹/几何条件，以及第一阶段的一条 narration。输出为这些 query points 在未来窗口中的稠密三维轨迹和可见性信息。

目标 mask、查询点采样、坐标锚定和时间采样原则沿用 MotionForesight。点图、相机对齐和 TrackCraft3R 轨迹可由观测或完整视频离线处理得到；完整未来只参与训练伪标签构造，不能进入预测条件。

### 文本通路

MotionForesight 主体来自 Wan2.1-T2V DiT，保留原生 text cross-attention。第一版恢复冻结的 Wan T5 encoder，将 narration 编码为 token-level context，并在 DiT 的 cross-attention 中使用：

```text
N_k -> frozen Wan T5 -> E_text^k
(RGB latent, pointmap latent, future mask, E_text^k)
    -> Wan DiT + forecasting adapter + cross-attention LoRA
    -> future residual-track latent
    -> frozen TrackCraft3R decoder
    -> dense 3D trajectory
```

保留 MotionForesight 原有 forecasting self-attention LoRA、future mask latent、输入输出投影和预测头。文本调制 future mask latent 只作为后续辅助消融；mask 的基本含义仍是未来观测缺失槽。

### 确定性训练

当前不加入 flow state、随机起点、latent noise、轨迹 VAE 或多步 ODE。先训练并冻结第一阶段，再在训练集观察前缀上用与推理一致的自由生成配置产生候选 `{N_hat_k}`。候选与真实 future narration 做双向蕴含匹配，只将达到阈值的最早匹配候选 `N_hat_k*` 与该样本三维真值配对：

```text
(V_obs, target/history, N_hat_k*) -> z_track_hat
z_track_hat -> frozen decoder -> X_future_hat
X_future_GT / residual-track target -> regression and visibility losses
```

训练时 T5、主视频先验和轨迹 decoder 默认冻结，更新预测 LoRA、cross-attention LoRA、mask/投影/预测头等适配参数。冻结 decoder 不阻断损失向上游适配器传递梯度。若第一阶段没有候选达到匹配阈值，该样本使用 GT narration 回退或暂不进入 predicted-narration 子集；不能把语义不匹配的候选与同一 `X_future_GT` 强行配对。全部替代候选也不能共享该样本唯一轨迹真值。

这种训练显著减小第二阶段看到的文本风格和错误分布与推理时的差异，但仍需分别报告 GT narration、matched predicted narration 和完整 predicted candidates，以测量第一阶段遗漏真实未来及训练集/测试集泛化造成的剩余差异。

### 推理

第一阶段得到 `{N_k}` 后，第二阶段逐条处理：

```text
for N_k in candidate narrations:
    X_k = MotionForesight(V_obs, target/history, N_k)
```

每条 narration 对应一条确定性轨迹。当前多轨迹来源是跨 narration 的语义分支，尚未包含同一 narration 内由随机变量产生的运动实现差异。

## 训练与数据边界

- 第一阶段语义监督主要依赖 Ego4D 的时间戳 narration。
- 第二阶段三维监督沿用 MotionForesight 的完整视频离线伪标签流程，候选数据集仍需在 Ego4D、SSv2 及许可条件之间最终确定。
- 同一观察前缀通常只有一个实际未来。未发生的反事实 narration 没有天然的三维 GT；当前版本只用真实 narration 训练轨迹映射，不把所有候选都配给同一个 GT。
- 阶段一与阶段二必须使用同一目标、同一未来时间窗和可对应的坐标约定。

## 约束与评估

### 第一阶段

评估未来事件目标一致性、事件级覆盖、语义重复率、互补性、跨视频稳定性和文本生成质量。双向蕴含只用于语义关系判断，不作为轨迹真实性证明。第一阶段不要求轨迹解码和三维轨迹评价。

### 第二阶段

评估稠密三维轨迹的 ADE/FDE 或等价点误差、可见点比例、端点和局部连续性，并进行同视频 narration 置换测试：改变文本时，宏观轨迹是否按事件语义改变；恢复真实文本时，是否保持 MotionForesight 的确定性基线性能。

### 当前版本暂不声称

- 不声称已经学习同一 narration 下的连续轨迹概率分布。
- 不声称第一阶段候选频率就是校准的模式概率。
- 不声称未匹配 narration 的轨迹一定具有反事实合理性。
- 不把 MolmoMotion 的小规模文本测试或 TrajLoom 的二维指标当作本方法效果证据。

## 仍需实验确认的事项

1. VLM 的具体选择、显存规模和 adapter 层位。
2. Ego4D narration 与 MotionForesight 三维伪标签的统一时间协议。
3. 双向蕴含 NLI 模型、阈值及其在事件等价判断上的任务校准；EMTA 分配和置信度训练规则已经确定。
4. MotionForesight 恢复文本后是否真正使用 narration，而非继续只依赖视频。
5. 前序 narration token memory 是否实际降低重复并提高长尾模式覆盖，以及未选中分支的额外候选质量；其基础训练规则已经确定。
6. 第一版确定性结果稳定后，是否需要引入 residual latent flow、条件 VAE 或其他多样性机制。
