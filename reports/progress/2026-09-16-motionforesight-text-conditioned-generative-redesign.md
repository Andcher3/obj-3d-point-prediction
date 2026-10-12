# MotionForesight 文本条件与生成式多未来改造审核

日期：2026-09-16

## 完成内容与状态

- 根据 MolmoMotion 小型文本控制测试的现象，重新评估第二阶段以 MotionForesight 为主干的理由，并核查 MotionForesight 论文、当前公开代码和 TrajLoom 机制。
- 核实 Wan DiT 的文本 cross-attention、文本投影和 T5 编码器仍在代码体系中；MotionForesight 当前缓存空提示的表示，在训练模型中移除文本编码器，并将可训练 LoRA 限制在 self-attention。
- 区分 future mask latent、future track latent、文本条件和随机变量的职责，提出分阶段的候选改造。该设计尚未训练或验证，本轮未访问远程服务器或运行模型。

## 1. 选择 MotionForesight 的理由

将 MotionForesight 设为第二阶段主干具有合理性：它直接输出目标物体的稠密、参考帧锚定、度量三维点轨迹，与项目最终接口一致；MolmoMotion 默认预测少量查询点，当前本地小样本也只验证了文本敏感性，没有验证可靠指令遵循。

需要修正两点：

- TrajLoom 生成的是稠密轨迹场，只是处于二维图像坐标；“其余两者均为稀疏轨迹”只适用于 MolmoMotion。
- MotionForesight 与 MolmoMotion 可在 MotionForesight 论文采用的共同协议下比较；TrajLoom 的二维任务、数据和指标不同，不能用跨论文指标直接认定 MotionForesight 性能更高。

因此，当前选择主要由输出表示、现有被动视频预测能力和代码可改造性支持。MolmoMotion 可降为数据与对照资源，不作为主干；这不表示已从小样本结果证明其普遍无效。

## 2. 文本条件入口

### 代码核查

[MotionForesight 代码](https://github.com/brains-bots-n-behavior/motionforesight)中的 Wan DiT block 保留 `cross_attn(norm3(x), context)`，并在进入 block 前用 `text_embedding` 投影 context。`WanSceneFlowPredictor` 用 T5 编码空字符串、缓存 `_null_context`；训练模型因只使用这一空表示而移除 text encoder。其配置注释还明确说明，当前 LoRA 训练跳过 cross-attention，因为它只读取空文本。

由此，恢复原生文本条件通路是有代码依据的最小改造：

```text
future narration N_k
    -> frozen Wan T5 encoder
    -> token-level text context E_k
    -> Wan DiT text cross-attention
    -> future residual-track latent
    -> frozen track decoder
    -> dense 3D tracks
```

### 建议训练范围

- 冻结 Wan 主干、TrackCraft3R tracking adapter、VAE/track decoder和 T5 编码器。
- 保留 MotionForesight 已有 forecasting self-attention LoRA、mask latent、I/O projection和 head 的训练路径。
- 新增一套面向 `cross_attn.{q,k,v,o}` 的小型 LoRA；必要时训练一个轻量 text projection 或门控。具体 rank 尚未确定。
- 首个文本基线只使用 cross-attention，检查同一视频换用不同未来叙述是否系统性改变宏观轨迹。

结构入口存在不代表模型一定使用文字。训练数据中视频通常已经强烈提示真实后续，只用匹配对的坐标回归损失仍可能得到忽略文本的局部最优。需要用文本置换敏感性、同类观测下不同动作的成组数据，以及经过语义筛选的 hard-negative / margin 检查来判断或加强条件利用；不能把所有未发生的替代未来标成错误轨迹。

### 文本调制 mask latent 的定位

MotionForesight 的 RGB/pointmap mask latent 表示未来观测缺失，并通过时间位置区分不同 future slot。可增加零初始化的文本投影偏移或门控，使未来槽从输入处携带语义；它适合作为 cross-attention 之后的辅助条件与消融。

不建议把 pooled text 压入 mask 作为唯一文本通路：它会把未知槽标记与事件语义混合，也丢失 token 级动作、对象和顺序信息。原生 cross-attention 在每一层都能读取完整文字，更符合 Wan 的预训练接口。

## 3. 多样性机制

### mask 噪声不足以定义生成分布

若只做 `m_future + epsilon` 并继续用单真实未来的 MSE/L1 训练，确定性网络可以通过忽略 `epsilon` 降低期望损失，或者把噪声变成不受控误差。已有 CVAE、Best-of-K / multiple-choice、stochastic conditioning、diffusion和 flow 方法都需要与随机变量相匹配的训练目标；“确定性网络输入加噪声”本身没有这些性质。

future mask latent 和 future track latent 的职责应分开：

```text
future mask latent   = 未观测的未来 RGB / geometry 槽
future track latent  = 要解码成未来三维运动的目标表示
text narration       = 宏观事件条件
random state         = 给定该事件后的运动实现不确定性
```

### 推荐候选：条件 rectified flow 生成 track latent

MotionForesight 当前在固定 regression timestep 上执行单步 latent 回归。其 Wan DiT 已具有时间步嵌入和原始 flow/diffusion 基础，更自然的候选是把它适配为条件速度场：

```text
z_1 = EncodeTrack(X_future_GT)
z_0 ~ p_0(. | observed boundary)
z_t = (1-t) z_0 + t z_1

v_theta(z_t, t | V_obs, P_obs, target, N_k)
    -> integrate from t=0 to 1
    -> z_track_future
    -> frozen decoder
    -> dense 3D trajectories
```

现有代码已经用冻结 VAE 编码 residual track target，并用冻结 decoder 还原坐标。这使“先测试现有 track latent 是否可作为生成空间”比直接另建 TrajLoom-VAE更有依据。不过，当前 row/query stream只接收重复的参考帧 RGB/pointmap latent，flow 改造必须给当前 noisy track state `z_t` 增加明确输入路径，例如新的 flow-state adapter或扩展输入流；只改变 timestep 和损失还不够。

TrajLoom 的历史条件源状态、boundary hints、时空一致性和 on-policy K-step fine-tuning可作为后续借鉴项。二维 offset field、TrajLoom-VAE和具体损失权重不直接照搬到三维 track latent。

### 是否需要新的轨迹 VAE

暂不预设。应先测试现有编码器—解码器对真实 future residual tracks 的重建误差、边界连续性、可见性和局部空间一致性：

- 若重建质量和 latent 插值/扰动行为足够稳定，直接在现有 track latent 上训练 flow。
- 若该空间只适合确定性回归，采样后容易产生解码伪影或局部不一致，再训练专用三维 trajectory autoencoder。

因此，AI 所说“不一定需要 TrajLoom-VAE”可以保留为候选；不能提前断言现有 track latent 天然适合高斯到数据分布的 flow。

## 4. 两阶段分工

当前方法可以形成分层条件分布：

```text
Stage 1: p(N_k | V_obs, target)        决定宏观事件
Stage 2: p(X | V_obs, target, N_k)     生成该事件下的连续运动
```

第一阶段不同 narration 提供跨事件模式，第二阶段对每个 narration 使用不同 flow 初值产生模式内运动变化。这样无需让一个噪声同时承担“杯子被喝水还是被放下”和“放下过程的速度、方向、终点差异”。

该结构能产生条件样本，但完整的 `p(X | V_obs, target)` 还依赖第一阶段模式权重与第二阶段条件分布如何组合；多个随机种子不自动提供校准概率。

## 5. 建议的验证顺序

1. 复现原 MotionForesight deterministic baseline，并保存现有 latent 编解码重建误差。
2. 恢复 T5 context与 cross-attention LoRA，先验证文本置换能稳定改变宏观轨迹，同时保持原基线精度。
3. 在没有显式文本的条件下建立 latent flow 基线，确认不同种子产生有效且连续的轨迹样本。
4. 合并文本与 flow，分别评估跨 narration 的事件差异和同 narration 内的轨迹差异。
5. 将 text-conditioned mask modulation、history-conditioned source、时空一致性和 K-step fine-tuning作为逐项消融，不在第一版同时加入。

## 输出

- 本记录、`research_roadmap.md` 当前主线补充及 `reports/progress_log.md` 索引。
- 本轮未进行新实验，所有结构效果仍待验证。
