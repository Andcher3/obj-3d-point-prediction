![[Pasted image 20260915142406.png]]

真值来源（构造数据）：
	**数据集**：SSv2
	**物体mask**：方法：用官方的动名词中的物体名称作为提示让SAM 在锚定帧识别对应的物体并分割，传播到整个视频
	**查询点**：在参考帧（第0帧）的mask上采样
	**点图**：DA3估计深度得到单帧点图
	**在整段视频上对齐的点图**：估计相机运动，把不同帧中的 3D 点转换到同一个三维坐标系（最后一帧观测帧）
	**提取三维轨迹**：将**对齐的点图**、**视频**和**查询点**送入TrackCraft3D生成完整视频的查询点轨迹


current:
![[Pasted image 20260915173144.png]]
RGB/mask latent -> VAE encoder
							        ->track latent -> frozen video DiT + forecasting LoRA  -> **residual track latent **  ->TrackCraft3R decoder
pointmap/mask latent-> VAE encoder 


修改：track latent->VAE/interface ->rectified flow-> future latent ->decoder
									↑
									文本条件（ pooled embedding、cross-attention 或条件 mask latent 注入）



你的选择有充分依据。以 **MotionForesight 作为第二阶段主体**，再分别解决“语义条件”和“多模态生成”，比继续围绕 MolmoMotion 改造更贴近我们最终的输出目标。不过有两处需要修正：

MotionForesight 确实预测 dense、reference-anchored 的 3D tracks；MolmoMotion 原生一次只预测 8 个 query points，属于 sparse 3D。**TrajLoom 本身也是 dense trajectory generation，只是它预测的是图像空间中的 dense 2D trajectory field**，因此“其余两者都是稀疏”这一点只适用于 MolmoMotion。([arXiv](https://arxiv.org/abs/2607.16192 "MotionForesight: Re-purposing Video Models for Future 3D Scene-Flow Prediction"))

性能方面，你的判断对 MotionForesight 与 MolmoMotion 的直接比较成立。MotionForesight 论文在统一 SSv2/OOD protocol 下，SSv2 ADE 为 4.47 cm，而 MolmoMotion 无语言为 5.66、带 GT language 反而为 5.93；OOD 也有类似结果。更重要的是，作者自己指出带语言的 MolmoMotion 获得了更多 test-time information。([arXiv](https://arxiv.org/abs/2607.16192 "MotionForesight: Re-purposing Video Models for Future 3D Scene-Flow Prediction")) 但 TrajLoom 做的是 dense 2D forecasting，指标体系不同，所以不能根据跨论文数字说 MotionForesight“性能高于 TrajLoom”。

---

## 文本条件：其实 MotionForesight 已经给我们留好了一个入口

这是我查代码以后觉得最值得利用的一点。

MotionForesight 的主体来自 **Wan2.1-T2V DiT**。Wan 原本就是 text-to-video 模型，所以每个 DiT block 天生就有：

Self-Attention→Text Cross-Attention→FFN.\text{Self-Attention} \rightarrow \text{Text Cross-Attention} \rightarrow \text{FFN}.

它的代码现在仍然保留：

x←x+CrossAttn⁡(x,context)x \leftarrow x+\operatorname{CrossAttn}(x,\text{context})

而 `context` 进入网络前还会经过 `text_embedding`。

**MotionForesight 没有删掉这条文本条件通路，只是把它废置了。**

它目前的代码做的是：

```text
context = null_text_context
        ↓
Wan DiT cross-attention
```

而且初始化时因为始终只使用 null prompt，直接把 T5 text encoder 从模型里丢掉了。代码甚至明确写着：cross-attention LoRA 没有训练，因为它只 attend 到 null text prompt。

因此我目前最倾向的第一版语言条件改造其实非常自然：

Nk→Wan T5 EncoderEktextN_k \xrightarrow{\text{Wan T5 Encoder}} E_k^{text}

然后把原来的

CnullC_{\rm null}

替换成

Ektext.E_k^{text}.

MotionForesight 主体保持：

Iobs→zrgbPobs→zgeofuture→mrgb,mgeo\begin{aligned} I_{\rm obs}&\rightarrow z_{\rm rgb}\\ P_{\rm obs}&\rightarrow z_{\rm geo}\\ \text{future}&\rightarrow m_{\rm rgb},m_{\rm geo} \end{aligned}

最后：

DiT⁡(zrgb,zgeo,mfuture,Ektext)→ztrackfuture\boxed{ \operatorname{DiT} ( z_{\rm rgb}, z_{\rm geo}, m_{\rm future}, E_k^{text} ) \rightarrow z_{\rm track}^{future} }

再经过原本 frozen decoder 得到 dense 3D trajectory。

这条路线的好处很明确：**文本直接进入原 Wan 在预训练阶段已经学过的 text→visual-dynamics cross-attention 通道**。我们不用自己凭空设计“文本到底应该怎么和空间 latent 融合”。

---

## 那么应该训练什么？

我目前认为比较合理的是恢复/保留 frozen T5 text encoder，然后增加一套新的 **cross-attention LoRA**。

MotionForesight 原本已经训练：

forecasting LoRA+mask+I/O projection+head.\text{forecasting LoRA} +\text{mask} +\text{I/O projection} +\text{head}.

现在把语言加入以后，可以变成：

forecasting LoRA+text cross-attention LoRA+mask+I/O/head.\text{forecasting LoRA} + \boxed{\text{text cross-attention LoRA}} + \text{mask} + \text{I/O/head}.

原因也很直接：原来的 Wan cross-attention 学的是

text→future video dynamics,\text{text}\rightarrow\text{future video dynamics},

我们现在需要把这种关联调整成

text→future 3D track dynamics.\text{text}\rightarrow\text{future 3D track dynamics}.

因此直接冻结 cross-attention 本身未必够，而全量训练又没必要。一个 fresh LoRA 是比较符合 MotionForesight 原论文“保留 video prior、用小 adapter 改任务”的哲学的。

训练数据就是：

(Vobs,NfutureGT,XfutureGT).( V_{\rm obs}, N_{\rm future}^{GT}, X_{\rm future}^{GT} ).

也就是观测视频 + Ego4D future narration + 由完整视频离线产生的 dense 3D track GT。

这和第一阶段、第二阶段接口也正好一致。

---

## 那“把文本写进 learnable mask latent”有没有价值？

**有，但我不建议把它作为唯一文本通路。**

现在的 mask：

mrgb,mpmm_{\rm rgb},m_{\rm pm}

在 MotionForesight 中代表的是：

> “这里是一个未知未来时刻，没有 RGB / pointmap。”

这些 mask 是共享的，时间差异依赖 RoPE；论文和代码都明确把它们定位成 future unknown slots，而不是未来运动本身。([arXiv](https://arxiv.org/abs/2607.16192 "MotionForesight: Re-purposing Video Models for Future 3D Scene-Flow Prediction"))

我们完全可以改成：

mrgb(k)=mrgb+Argb eˉktextm_{\rm rgb}^{(k)} = m_{\rm rgb} + A_{\rm rgb}\,\bar e_k^{text} mpm(k)=mpm+Apm eˉktext.m_{\rm pm}^{(k)} = m_{\rm pm} + A_{\rm pm}\,\bar e_k^{text}.

这样同一个视频：

```text
“drink from the cup”
        ↓
future mask A

“put the cup down”
        ↓
future mask B
```

模型从进入 future slots 的第一刻就知道“这一组未知未来是在什么语义假设下预测”。

我会把它理解为一种 **future-slot semantic initialization**。

它和 cross-attention 可以同时存在：

text→{DiT cross-attention全局持续条件future mask modulation未来槽显式条件\boxed{ \text{text} \rightarrow \begin{cases} \text{DiT cross-attention} & \text{全局持续条件}\\ \text{future mask modulation} & \text{未来槽显式条件} \end{cases}}

其中 cross-attention 是主通路，mask modulation 更适合做辅助设计和消融。这样比完全把文本信息压进一个 pooled vector 后塞进 mask 更稳妥，因为 token-level narration 信息仍然可以在每层 cross-attention 中被读取。

---

# 多样性这里，需要把你贴出的 AI 方案纠正一个关键概念

它有一个**非常值得保留的核心思想**：

> 不要在最终 3D 坐标上随便加噪声；应该把随机性放进生成 future motion 的 latent distribution 中。

这和 TrajLoom、TrajFlow 等生成式 trajectory forecasting 的思路是一致的。TrajLoom 本身就是先学习 dense trajectory latent，再在 latent space 用 flow matching 生成未来；TrajFlow 也直接用 flow matching 建模多模态未来。([arXiv](https://arxiv.org/abs/2603.22606?utm_source=chatgpt.com "TrajLoom: Dense Future Trajectory Generation from Video"))

但它把 MotionForesight 中两个 latent 混到了一起：

### `mask latent` 和 `track latent` 职责完全不同

MotionForesight 当前实际上是：

mfuture⏟输入：未来未知⟶DiT⁡⟶ztrackfuture⏟输出：未来运动⟶Dtrack.\underbrace{m_{\rm future}}_{\text{输入：未来未知}} \longrightarrow \operatorname{DiT} \longrightarrow \underbrace{z_{\rm track}^{future}}_{\text{输出：未来运动}} \longrightarrow D_{\rm track}.

代码也确实是把 observed RGB/Pj latent 与扩展后的 mask 拼起来送进 DiT，DiT 输出 `query_latent`，再分成 xyz/visibility latent 并解码。

所以我不赞成把：

mfuture+ϵm_{\rm future}+\epsilon

直接当成我们最终的 probabilistic model。

这等价于给一个**确定性回归器的输入 token 加随机扰动**。如果训练目标仍然是单一 GT 的 MSE/L1，模型最容易学会：

∂X^∂ϵ≈0\frac{\partial \hat X}{\partial \epsilon}\approx0

也就是把噪声忽略掉；或者产生无法控制的误差。我们之前 Future-L1 latent perturbation 的实验其实已经从另一个角度见过类似问题。

真正应该随机化的是：

ztrackfuture\boxed{z_{\rm track}^{future}}

代表的 future-motion distribution。

---

## 所以我现在会把两个问题这样分开

第一步，文本条件模型仍然可以保持完全 deterministic：

Vobs+Nk→text-conditioned MotionForesight→ztrackfuture→Xfuture\boxed{ V_{\rm obs} + N_k \rightarrow \text{text-conditioned MotionForesight} \rightarrow z_{\rm track}^{future} \rightarrow X_{\rm future} }

先确保同一视频：

NA=drinkN_A=\text{drink}

和

NB=put downN_B=\text{put down}

能够真的改变输出轨迹。

这一点尤其重要，因为你正在 MolmoMotion 上观察到的问题恰恰是：

video signal≫text signal.\text{video signal}\gg\text{text signal}.

如果这一问题没有先解决，再加 flow 只会得到“多条依然不听文本的轨迹”。

然后多样性模块才建模：

p(ztrackfuture∣Vobs,Nk).p(z_{\rm track}^{future}\mid V_{\rm obs},N_k).

最符合 TrajLoom 启发的形式是：

ϵ∼N(0,I)\epsilon\sim\mathcal N(0,I) ϵ→vθ(zt,t∣Cvideo,Ctext)ztrackfuture\epsilon \xrightarrow{ v_\theta(z_t,t\mid C_{\rm video},C_{\rm text}) } z_{\rm track}^{future} ztrackfuture→DtrackXfuture.z_{\rm track}^{future} \xrightarrow{D_{\rm track}} X_{\rm future}.

不同 ϵm\epsilon_m：

ϵ1,ϵ2,…,ϵM\epsilon_1,\epsilon_2,\ldots,\epsilon_M

产生：

Xk,1,Xk,2,…,Xk,M.X_{k,1},X_{k,2},\ldots,X_{k,M}.

于是变量的职责非常干净：

Nk决定“发生什么”mfuture表示未来观测缺失ϵm表示同一语义下的几何不确定性ztrackk,m表示具体未来运动Dtrack恢复 dense 3D trajectories\boxed{ \begin{array}{ll} N_k & \text{决定“发生什么”}\\ m_{\rm future} & \text{表示未来观测缺失}\\ \epsilon_m & \text{表示同一语义下的几何不确定性}\\ z_{\rm track}^{k,m} & \text{表示具体未来运动}\\ D_{\rm track} & \text{恢复 dense 3D trajectories} \end{array}}

我认为这个区分对我们后面的方法设计非常重要。

---

还有一个目前不能直接接受 AI 方案的地方：**TrajLoom 有专门训练的 TrajLoom-VAE，因此它明确知道自己的 latent 是一个适合生成轨迹的空间。MotionForesight 的 residual-track latent 是为了 deterministic tracking/regression 学出来的，我们目前没有证据证明它天然就是一个适合从高斯噪声做 flow matching 的生成空间。**

所以：

“不需要 TrajLoom-VAE，直接在 MF track latent 做 flow”\text{“不需要 TrajLoom-VAE，直接在 MF track latent 做 flow”}

目前是一个**很有吸引力的候选假设**，还不能直接当成成立的事实。

这可能恰恰会成为我们讨论“第二个问题：怎么生成多样输出”时最核心的技术问题。

因此就第一个问题而言，我现在的倾向已经相当明确：

Stage-1 narration→Wan T5→恢复 MotionForesight 原生 text cross-attention\boxed{ \text{Stage-1 narration} \rightarrow \text{Wan T5} \rightarrow \text{恢复 MotionForesight 原生 text cross-attention} }

并训练新的 cross-attention LoRA；**text-conditioned future mask** 可以作为第二条辅助条件路径。这个设计比单独修改 mask latent 更有原架构依据，也更容易解释为“在 MotionForesight 上恢复并重新适配其母体 Wan2.1 原本就存在的文本条件能力”。

而多样性暂时不要和 mask latent 绑定。更合理的研究对象是：

p(ztrackfuture∣Vobs,Nk)\boxed{ p(z_{\rm track}^{future}\mid V_{\rm obs},N_k) }

以及 **MotionForesight 的 track latent 到底能不能直接承担这个生成空间**。这才是我们接下来需要仔细研究的第二个问题。

我明白了，直接启用原生的text cross-attn确实会更简单和温和一些。我们可以以此为现有的文本条件添加方法。

看来我们需要解决的问题主要还是如何处理多样性的问题