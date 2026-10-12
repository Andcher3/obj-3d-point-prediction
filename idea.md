
验证数据集-那个完整性和概率性
概率性这一点还是待讨论
NEP

目前情况：
一阶段：
SemDiD+Future L1+未来latent扰动的效果不佳：
缺乏事件级变化

可能的新思路？
借鉴MotionForesight将对当前视频生成轨迹的模型转化成预测轨迹的模型那样，将对当前视频总结模式/意图的模型转化成预测模式/意图的模型，并借鉴ObjectForesight的思路生成多结果；同时，再以ModeSeq的方法确保输出在模式上的区分

推理：
如何指定物体：MotionForesight格式，SAM+文本形式输入？（物体名称？坐标？）
一阶段：
	第一步：
		将视频未来片段mask，+adapter将~~narration模型~~强VLM（提供世界知识先验）MolmoMotion？改造为未来narration预测模型；
		真值来源：
			1. EGO4D narration+timestamp([[存在标注不准确的问题]])
			2. 完整视频+~~原narration~~原VLM模型：
				同时给$t_c$
				+提示词：Describe only what happens after timestamp $t_c$.
				（[[teacher 是否真的能严格遵守时间边界]]需要做一下exp）
			3. [[不监督文本，监督具有时间结构的 semantic latent]]
			4. 用图像/视频—文本对齐模型比较未来视频和输出？（不能解决合并同模式输出的问题）
			5. 在合理潜空间/规则/变量（前提）下的最大化空间覆盖少量采样分支
	第二步：
			对自回归narration模型采用sampling，token sampling 本身就相当于随机性来源​
	第三步：
			借用ModeSeq的方式确保不同结果顺序输出并且能够根据上一个结果的信息来使这次的结果与上次的不同
			使用双向蕴含等工作来鉴别语义在事件级别的区别而不是单纯的token级别（连续数值置信度+阈值判断等价）by NLI
			待验证：NLI 是否按我们关心的事件差异分配分支，以及生成器是否实际利用了不同模式条件。
	第四步：
		
			
			
	 


二阶段：
TrajLoom+MotionForesight+MolmoMotion
待验证（TrajLoom）

实在不行放弃MotionForesight，只保留TrajLoom和MolmoMotion？

MotionForesight的输出形式、指标都更优，应当优先选取MotionForesight作为base
### 用文本调制未来 mask latent

MotionForesight 用 learned mask latents 表示“未来不可见”。可以让它们由文本条件生成：

$$  
m_{\text{rgb}}(a),\qquad m_{\text{pm}}(a),  
$$

而不是所有动作共享同一组 mask latent。这样“打开抽屉”和“关闭抽屉”会产生不同的未来条件。



### 把文本 token 注入视频 DiT 的 cross-attention

可以直接把文本作为条件送给MotionFoiresight，这是最直接的方案。使用文本编码器得到：

$$  
T_{\text{text}}=\mathrm{Encoder}(a),  
$$

然后让视频—几何 transformer 的每层同时 attend 到：

- 观测 RGB/pointmap latent；
- 文本 token；
- reference query latent。

可以写成：

$$  
H^{l+1} = \mathrm{TransformerBlock} \left( H^l,\, T_{\text{text}},\, T_{\text{visual-geometry}} \right).  
$$

这样文本会影响未来 track latent，最终通过冻结的 track decoder 生成稠密 3D 轨迹。

这相当于把 MotionForesight 的：

```text
视觉几何条件 → 未来轨迹
```

变成：

```text
视觉几何条件 + 动作文本 → 未来轨迹
```



借鉴自 SV2P:
有，而且最值得仿照的不是“在 MotionForesight 输出上随便加高斯噪声”，而是把它改成**带潜变量的条件预测器**。已有方法大致分两类：

1. **保留原来的确定性预测器，只增加一个描述未来模式的潜变量**：最接近你想要的 MotionForesight 改造；
2. **把未来轨迹本身放进扩散/去噪过程**：多样性更自然，但对原架构改动更大。

我会优先推荐第一类。

## 一、最接近的先例：SV2P 的潜变量方法

[Stochastic Variational Video Prediction（SV2P）](https://www.alphaxiv.org/abs/1710.11252) 的核心思想正是：保留一个原本可以确定性预测未来的网络，再引入一个随机潜变量 $z$，使不同的 $z$ 对应不同的未来模式。论文明确把未来建模为：

$$
p(x_{c:T}\mid x_{0:c-1})
=
\int p_\theta(x_{c:T}\mid x_{0:c-1},z)p(z)\,dz.
$$

生成器本身仍然可以是一个确定性的预测网络；随机性来自每次采样的 $z$。[Latent Variable](https://www.alphaxiv.org/abs/1710.11252?page=3)

把它迁移到 MotionForesight，可以写成：

$$
\hat X_{\mathrm{future}}
=
F_\theta\!\left(
V_{\mathrm{obs}},q,z
\right),
\qquad
z\sim p(z\mid V_{\mathrm{obs}},q),
$$

其中：

- $V_{\mathrm{obs}}$ 是观测 RGB 和 pointmap；
- $q$ 是参考物体查询；
- $F_\theta$ 是 MotionForesight 的未来 adapter 加冻结 track decoder；
- $z$ 表示“未来会采取哪一种合理运动模式”；
- $\hat X_{\mathrm{future}}$ 是整个物体的稠密 3D 轨迹。

例如相同的观察前缀下，不同的 $z$ 可以让模型生成：

- 向上拿起；
- 向右推动；
- 旋转；
- 保持静止；
- 不同幅度或不同时间节奏的运动。

这里的关键是：**$z$ 不是每个点单独一个噪声，而是一个共享的 scene-level 或 object-level latent。**这样整个稠密物体的点会共同选择一种运动模式，而不是每个点独立随机抖动。

## 二、训练时为什么需要 posterior？

如果只在训练和推理时都随机采样 $z\sim\mathcal N(0,I)$，模型很可能学会忽略 $z$，因为观测输入本身已经足够让它最小化平均重建损失。

SV2P 的做法是训练两个分布：

### 1. 先验：推理时能获得

先验只依赖观测：

$$
p_\psi(z\mid V_{\mathrm{obs}}).
$$

推理时从这个分布采样：

$$
z\sim p_\psi(z\mid V_{\mathrm{obs}}).
$$

### 2. 后验：训练时可以看到真实未来

训练阶段额外使用完整真实轨迹 $X_{\mathrm{future}}$，由 inference network 估计：

$$
q_\phi(z\mid V_{\mathrm{obs}},X_{\mathrm{future}}).
$$

然后从后验采样：

$$
z
=
\mu_\phi(V_{\mathrm{obs}},X_{\mathrm{future}})
+
\sigma_\phi(V_{\mathrm{obs}},X_{\mathrm{future}})\odot\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I).
$$

这个后验可以看到真实未来，所以能告诉生成器：

> 这条训练样本属于哪一种未来模式。

SV2P 使用重建项加 KL 项训练：

$$
\mathcal L
=
-\mathbb E_{q_\phi}
\left[
\log p_\theta
\left(
X_{\mathrm{future}}
\mid V_{\mathrm{obs}},z
\right)
\right]
+
\beta D_{\mathrm{KL}}
\left(
q_\phi(z\mid V_{\mathrm{obs}},X_{\mathrm{future}})
\;\|\;
p_\psi(z\mid V_{\mathrm{obs}})
\right).
$$

SV2P 的重要经验是：如果 inference network 也只看观测，那么 $z$ 中没有观测之外的未来信息，生成网络很容易忽略它；论文明确报告，后验需要在训练时访问未来帧，才能有效迫使预测器使用潜变量。[Posterior Training](https://www.alphaxiv.org/abs/1710.11252?page=4)

这对 MotionForesight 特别重要，因为它现在的训练设置本来就有完整视频和伪真值未来轨迹，只是未来帧不输入预测器。完整未来轨迹可以自然地用于训练 posterior。MotionForesight 当前正是用完整视频离线生成未来轨迹，再只把观测前缀输入模型。[Training Supervision](https://www.alphaxiv.org/abs/2607.16192?page=8)

## 三、如何具体改造 MotionForesight？

原始模型大致是：

```text
观测 RGB + 观测 pointmap + reference query
                  ↓
          视频 DiT + forecasting adapter
                  ↓
          residual track latent
                  ↓
          frozen track decoder
                  ↓
          一组确定性的未来 dense tracks
```

可以改成：

```text
观测 RGB + 观测 pointmap + reference query
                  ↓
         条件先验 p(z | observation)
                  ↓
              采样 z
                  ↓
          视频 DiT + forecasting adapter
                  ↓
          residual track latent
                  ↓
          frozen track decoder
                  ↓
          一组随机的未来 dense tracks
```

最简单的注入方式是把 $z$ 投影成一个 latent token，并拼到未来预测模块的输入中：

$$
e_z=W_z z.
$$

然后将它加入每个未来时间槽位：

$$
h_t^{\mathrm{future}}
=
h_t^{\mathrm{future}}
+
e_z,
\qquad t\ge T_1.
$$

或者把它作为 cross-attention 的条件：

$$
H_{\mathrm{future}}^{l+1}
=
\mathrm{TransformerBlock}
\left(
H_{\mathrm{future}}^l,
H_{\mathrm{obs}}^l,
e_z
\right).
$$

还可以用 FiLM 调制：

$$
\mathrm{FiLM}(h_t)
=
\gamma(z)\odot h_t+\beta(z).
$$

这几种方式中，我会优先尝试：

> **一个共享的 object-level latent $z$，通过 cross-attention 或 FiLM 注入未来 track latent，而不是把独立噪声加到每个点的坐标上。**

因为共享 latent 可以协调所有点的运动。

## 四、时间不变 latent 还是时间变化 latent？

这是 SV2P 对 MotionForesight 很有参考价值的一点。

### 时间不变 latent

对整个预测片段只采样一次：

$$
z\sim p(z\mid V_{\mathrm{obs}}).
$$

然后所有未来时间步共享它：

$$
\hat X_t=F_\theta(V_{\mathrm{obs}},z,t).
$$

这适合表达“这段未来属于哪一种动作模式”：

- 拿起；
- 推动；
- 旋转；
- 停止。

对于 MotionForesight，我会首先采用这种形式，因为物体的未来运动模式通常应该在整个预测窗口内保持一致。否则每个时间步独立采样噪声，可能出现物体先上升、下一帧突然横移、再突然停止的不连贯轨迹。

### 时间变化 latent

每个未来时间步采样一个 $z_t$：

$$
z_t\sim p(z_t\mid V_{\mathrm{obs}},t).
$$

它能表达每一帧还有新的不确定性，但更容易产生时间不连续。SV2P 研究了两者：时间变化 latent 在较长预测范围可能泛化更好，而时间不变 latent 在动作贯穿整个视频的场景中可能更合适。[Latent Timing](https://www.alphaxiv.org/abs/1710.11252?page=4)

对 MotionForesight，一个较稳妥的折中是：

$$
z_{\mathrm{global}}
\quad+\quad
z_t,
$$

其中：

- $z_{\mathrm{global}}$ 选择整体动作模式；
- $z_t$ 只负责局部时间细节或小幅扰动；
- $z_t$ 可以通过 temporal RNN 或 causal transformer 生成，而不是独立采样。

## 五、SV2P 的训练技巧很值得直接借鉴

SV2P 发现，直接端到端训练经常导致 latent 被忽略。因此它采用分阶段训练：

1. 先训练生成器，使其学会基本的确定性预测；
2. 再启用 posterior，让模型使用 latent 降低未来重建误差；
3. 最后逐渐增加 KL 权重，使 posterior 接近可采样的 prior。[Training Schedule](https://www.alphaxiv.org/abs/1710.11252?page=5)

迁移到 MotionForesight 可以是：

1. **阶段一**：保留当前确定性 MotionForesight，训练 adapter；
2. **阶段二**：加入 posterior $q_\phi(z\mid V_{\mathrm{obs}},X_{\mathrm{future}})$，暂时降低 KL 权重；
3. **阶段三**：逐渐增大 KL，使推理时从 prior 采样也能生成合理结果；
4. **阶段四**：冻结或部分冻结原有 track interface，只微调 latent 注入层和 forecasting adapter。

这比一开始随机加噪声更可靠。

## 六、为什么“直接给输出加噪声”通常不够？

比如你先得到确定性轨迹：

$$
\hat X=F(V_{\mathrm{obs}}),
$$

再加噪声：

$$
\hat X'=\hat X+\sigma\epsilon.
$$

这确实会生成不同输出，但通常不是多模态预测，而只是围绕同一条轨迹随机抖动。它不能自然地产生：

- “拿起”与“推开”两种不同模式；
- 抽屉沿滑轨运动与物体旋转；
- 不同的动作起始时间；
- 不同的运动方向和幅度。

更严重的是，独立的坐标噪声会破坏稠密点之间的几何一致性。一个物体表面上的相邻点可能被随机推向互相穿插的位置。

SV2P 的思想是把随机性放到**高层潜变量**中，而不是把独立噪声直接加到像素或输出坐标上。论文也指出，简单增加像素级噪声并不能自动得到有意义的随机预测；有效的随机性必须与潜在事件相关。[Structured Stochasticity](https://www.alphaxiv.org/abs/1710.11252?page=2)

## 七、另一条路线：直接对未来稠密轨迹做扩散

第二类可仿照方法是 [Motion Indeterminacy Diffusion（MID）](https://www.alphaxiv.org/abs/2203.13777)。它不是在一个确定性模型旁边加一个全局 $z$，而是把未来轨迹本身逐步加噪：

$$
y_k
=
\sqrt{\bar\alpha_k}\,y_0
+
\sqrt{1-\bar\alpha_k}\,\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I).
$$

训练一个网络从带噪轨迹 $y_k$、观测条件 $x$ 和噪声时间步 $k$ 中预测噪声：

$$
\epsilon_\theta(y_k,k,x).
$$

推理时从高斯噪声 $y_K$ 开始，逐步去噪到 $y_0$。每次从不同的 $y_K$ 开始，就能得到不同未来。[Diffusion Process](https://www.alphaxiv.org/abs/2203.13777?page=3)

它和 MotionForesight 的对应关系可以是：

- $y_0$：未来 dense track latent 或未来 dense 3D tracks；
- $x$：观测 RGB、观测 pointmaps、reference query；
- $\epsilon_\theta$：条件视频—几何去噪器；
- 最终的 $y_0$：交给 frozen track decoder，得到未来稠密轨迹。

流程是：

```text
真实未来 dense track latent y0
          ↓ 加噪
       noisy latent yk
          ↓
条件去噪网络(y k, k, observed geometry)
          ↓
       逐步去噪
          ↓
未来 dense track latent
          ↓
    frozen track decoder
          ↓
未来 dense 3D motion
```

这条路线的优点是：

- 多样性是模型目标的一部分；
- 可以自然表达多种未来；
- 不需要把所有模式压缩进一个低维 $z$；
- 可以对整个 dense track latent 联合建模，保持点间相关性。

缺点是采样要多次前向，成本明显增加。MID 论文报告其扩散采样在实验中使用约 100 步，并指出这会比普通轨迹预测昂贵很多。[Sampling Cost](https://www.alphaxiv.org/abs/2203.13777?page=8)

## 八、对 MotionForesight 来说，应该选哪一种？

如果你的目标是**尽量保留 MotionForesight 原架构**，我建议：

### 首选：SV2P 式 latent-conditioned MotionForesight

改动最小：

$$
\hat X
=
F_\theta(V_{\mathrm{obs}},q,z).
$$

训练：

$$
\mathcal L
=
\mathbb E_{q_\phi(z\mid V_{\mathrm{obs}},X)}
[
\mathcal L_{\mathrm{dense}}(F_\theta(V_{\mathrm{obs}},q,z),X)
]
+
\beta D_{\mathrm{KL}}(q_\phi\|p_\psi).
$$

推理：

$$
z\sim p_\psi(z\mid V_{\mathrm{obs}}),
\qquad
\hat X^{(i)}=F_\theta(V_{\mathrm{obs}},q,z^{(i)}).
$$

它保留：

- MotionForesight 的视频—几何 backbone；
- 未来 mask latent；
- frozen track decoder；
- dense pointmap 输出；
- 原有 decoded coordinate loss。

只新增：

- latent prior；
- future-aware posterior；
- latent 注入模块；
- KL 和多样性训练。

### 第二选择：扩散未来 track latent

如果发现一个低维 $z$ 无法承载足够丰富的未来模式，再把扩散过程放到**未来 track latent**上，而不是直接对高分辨率 dense 3D 点坐标扩散。这样可以避免在 $H\times W\times3\times T$ 的巨大空间中进行昂贵采样。



**结论是：有成熟先例，而且最适合 MotionForesight 的第一步不是“给坐标加噪声”，而是采用 SV2P 式的 future-mode latent。**让一个共享的 $z$ 控制整组稠密点的协调运动；训练时用能看到真实未来的 posterior 学会区分运动模式，推理时从只依赖观测的 prior 采样。若低维 latent 表达能力不足，再进一步把 MotionForesight 的 future track latent 改造成 MID 式的扩散生成器。


另一个TrajLoom的做法：
**有，最贴近你这个设想的现有做法是 [TrajLoom](https://www.alphaxiv.org/abs/2603.22606)：它不是在确定性预测器输出后随便加噪声，而是先把稠密轨迹压缩到一个**随机潜空间**，再从带噪的潜变量中生成未来稠密轨迹。这个思路比“给 MotionForesight 的 mask latent 加一点随机扰动”更可靠。

## 先区分三种“加噪声”

### 1. 推理时直接给原模型的 mask latent 加噪声

例如原始 MotionForesight 有共享的可学习 mask latent：

$$
m_{\mathrm{rgb}}^0,\qquad m_{\mathrm{pm}}^0.
$$

推理时临时改成：

$$
m_{\mathrm{rgb}}
=
m_{\mathrm{rgb}}^0+\sigma\epsilon_{\mathrm{rgb}},
\qquad
m_{\mathrm{pm}}
=
m_{\mathrm{pm}}^0+\sigma\epsilon_{\mathrm{pm}},
$$

其中 $\epsilon\sim\mathcal N(0,I)$。

然后每次采样不同的 $\epsilon$，希望得到不同未来。

这确实会让输出发生变化，但**不能保证变化是有意义的多种未来**。如果模型训练时从未见过这种扰动，随机噪声可能只造成：

- 轨迹抖动；
- 不自然的点间撕裂；
- 预测误差增加；
- 对文本语义没有稳定响应。

所以，单纯在推理时给 mask latent 加噪声不是一个完整的方法。

### 2. 训练一个随机条件回归器

可以把 mask latent 改成：

$$
m(a,z)=m^0+g_{\text{text}}(h_a)+g_{\text{noise}}(z),
$$

其中：

- $a$ 是文本动作；
- $h_a$ 是文本表示；
- $z$ 是随机变量；
- $g_{\text{text}}$ 把文本映射到动作条件；
- $g_{\text{noise}}$ 把随机变量映射到未来变化；
- $m^0$ 仍然保留原来“未来不可见”的基础语义。

模型变成：

$$
\hat X=f_\theta(V,m(a,z)),
\qquad z\sim\mathcal N(0,I).
$$

这时每次改变 $z$，模型就可能输出不同轨迹。但关键是：**训练时必须让 $z$ 学会承载真实未来之间的差异**。否则网络可能忽略 $z$，退化为：

$$
\hat X\approx f_\theta(V,a).
$$

这种方法可以用条件 VAE、latent regression、信息最大化或其他潜变量目标训练。

### 3. 在轨迹潜空间中做 flow/diffusion 生成

这是 [TrajLoom](https://www.alphaxiv.org/abs/2603.22606) 采用的方向。它先训练一个 TrajLoom-VAE，把稠密轨迹场压缩成潜变量：

$$
z_{\text{future}}\sim q_\phi(z\mid X_{\text{future}}),
$$

再训练一个条件 flow 模型，学习：

$$
p_\theta(z_{\text{future}}\mid
z_{\text{history}},V,a).
$$

推理时从带噪起点出发，通过 flow 积分生成未来潜变量，最后由 VAE decoder 恢复稠密轨迹。论文明确指出，未来运动是多模态的，确定性预测可能把多个合理未来平均成保守结果；因此它在潜空间中使用 rectified flow 来生成未来轨迹分布。[Stochastic Dense Forecasting](https://www.alphaxiv.org/abs/2603.22606?page=1)

这比直接对最终坐标加噪声好，因为噪声是在一个经过训练、结构化的轨迹潜空间中解释的。

## TrajLoom 的关键机制是什么？

它的流程可以简化为：

```text
历史稠密轨迹 + 历史视频 + 文本
              ↓
       历史轨迹潜变量
              ↓
      条件 flow 生成未来潜变量
              ↑
        随机噪声初始化
              ↓
        VAE decoder
              ↓
       未来稠密轨迹
```

它的随机性不是来自一个固定的确定性 decoder，而来自未来潜变量的随机生成过程。论文中具体做法是：

$$
z_0(k,n)
=
z_{-1}(n)+\sigma_0\eta(k,n),
\qquad
\eta\sim\mathcal N(0,I),
$$

也就是用最后历史 latent 作为边界附近的起点，再叠加高斯噪声。然后通过 flow matching 将这个带噪起点演化为未来轨迹 latent。[Noisy Latent Initialization](https://www.alphaxiv.org/abs/2603.22606?page=6)

因此它不是“输出一条轨迹后加噪”，而是：

> **从多个不同的随机潜变量起点，生成多个结构化的未来轨迹。**

TrajLoom 的 flow 模型还把历史轨迹 latent、历史可见性和视频特征作为条件，并在潜空间中联合生成完整未来窗口，而不是逐帧独立采样。[Conditional Flow](https://www.alphaxiv.org/abs/2603.22606?page=6)

## 如何把这个思路迁移到 MotionForesight？

最自然的方案不是直接修改冻结的 track decoder，而是把随机生成部分放在它之前：

```text
观测 RGB + 观测 pointmaps
        +
目标物体 mask
        +
文本动作
        ↓
条件未来 latent generator
        ↑
     随机噪声 z
        ↓
未来 track latent
        ↓
冻结的 TrackCraft3R decoder
        ↓
稠密未来 3D tracks
```

设 MotionForesight 原本的未来 track latent 为 $z_f$。可以增加一个条件生成器：

$$
z_f
\sim
p_\theta
\left(
z_f
\mid
V_{\text{obs}},
P_{\text{obs}},
a,
M_{\text{obj}}
\right).
$$

最后仍然使用原有 decoder：

$$
\hat X_{\text{future}}
=
D_{\text{track}}(z_f).
$$

这样有两个好处：

1. 保留 MotionForesight 已经学到的 pointmap/track decoder；
2. 只把确定性的未来 adapter 改造成随机的条件生成器。

这比直接在 $X$ 上预测几千个点的坐标更容易，因为模型在低维 latent 中建模不确定性。

## 你的 mask latent 想法应该怎么改？

你的想法是可行的，但最好把 mask latent 从“唯一的共享参数”改成：

$$
m(a,z)
=
m^0
+
\Delta m_{\text{text}}(a)
+
\Delta m_{\text{stoch}}(z,a).
$$

展开写就是：

$$
m_{\mathrm{rgb}}(a,z)
=
m_{\mathrm{rgb}}^0
+
G_{\mathrm{rgb}}(h_a)
+
H_{\mathrm{rgb}}(h_a,z),
$$

$$
m_{\mathrm{pm}}(a,z)
=
m_{\mathrm{pm}}^0
+
G_{\mathrm{pm}}(h_a)
+
H_{\mathrm{pm}}(h_a,z).
$$

其中：

- $m^0$：原始共享 mask latent，保留“未来输入缺失”的含义；
- $G(h_a)$：文本相关偏移，例如“打开抽屉”和“关闭抽屉”产生不同条件；
- $H(h_a,z)$：文本条件下的随机偏移，表示同一动作的不同合理执行方式；
- $z$：每次采样时重新生成的随机变量。

例如文本是“拿起杯子”：

- 文本项决定总体方向应偏向上；
- 随机项可以决定稍微向左、向右、不同速度或不同旋转方式；
- decoder 再将这些差异展开成整组物体点的协调运动。

但要注意：如果你只用两个全局 mask latent，随机变量的容量可能不够。它可能只能控制“整个未来 latent 的整体风格”，很难表达不同区域的精细差异。更好的设计是让噪声具有时间和空间结构：

$$
m_{t,k}(a,z)
=
m^0_k
+
G_k(h_a)
+
H_k(h_a,z),
$$

其中 $t$ 是未来时间位置，$k$ 是空间 latent token。这样可以表达：

- 早期接触、后期移动；
- 物体不同部位的不同变化；
- 铰接部件之间的不同运动；
- 局部形变的随机性。

不过这也改变了原始 mask latent 的含义：它不再只是“未来不可见”的标记，而变成了：

> **带有动作和随机未来信息的未来条件 token。**

所以最好保留一个明确的缺失指示，例如：

$$
c_t^{\text{future}}
=
[
m_{\text{missing}},
m_{\text{text}},
m_{\text{noise}}
].
$$

不要让模型只能通过一个混合向量同时猜测“未来不可见”和“未来应该怎么动”。

## 更推荐的轻量方案：mask latent + 条件 latent generator

如果完全复制 TrajLoom 的 VAE 和 flow，改动会比较大。一个折中设计是：

### 第一步：保留 MotionForesight 的输入结构

观测帧使用正常 RGB/pointmap latent：

$$
c_t=
[E_{\text{rgb}}(I_t);E_{\text{pm}}(P_t)].
$$

未来帧继续使用 mask latent。

### 第二步：让文本和噪声只生成未来 mask 条件

$$
M_{\text{future}}
=
M^0
+
G(h_a)
+
H(h_a,z).
$$

### 第三步：用一个未来 latent refinement module 产生 track latent

不要让主干直接从单一 mask latent 回归未来轨迹，而是让一个小型 flow/refinement network 逐步修改随机初始状态：

$$
z_0
=
z_{\text{boundary}}
+
\sigma\epsilon,
$$

$$
\frac{d z_s}{ds}
=
v_\theta
\left(
z_s,s,
c_{\text{obs}},
M_{\text{future}}(a,z)
\right).
$$

最后：

$$
\hat X_{\text{future}}
=
D_{\text{track}}(z_1).
$$

这里可以让文本同时进入：

1. mask latent；
2. latent flow 的条件输入。

这样做比只把文本塞进 mask latent 更稳，因为 mask latent 负责把条件送入未来时间槽，flow/refinement module 则负责真正建模多样性。

## 训练时不能只让随机变量“存在”

需要防止模型忽略 $z$。至少有三种常见训练思路。

### 方法一：VAE 风格的未来编码

训练时使用真实未来轨迹 $X_f$，编码得到后验：

$$
q_\phi(z\mid X_f,V,a).
$$

然后 decoder 根据 $z$ 重建未来：

$$
\hat X_f=D_\theta(V,a,z).
$$

优化：

$$
\mathcal L
=
\mathcal L_{\text{recon}}
+
\beta
D_{\mathrm{KL}}
\left(
q_\phi(z\mid X_f,V,a)
\;\|\;
\mathcal N(0,I)
\right).
$$

推理时没有真实未来，就从 $\mathcal N(0,I)$ 采样。TrajLoom 的 VAE 就是通过重建损失和 KL 项学习轨迹潜空间。[Trajectory VAE](https://www.alphaxiv.org/abs/2603.22606?page=6)

问题是普通 CVAE 可能出现 posterior collapse：模型忽略 $z$，只依赖视觉和文本条件。因此通常还需要 KL 权重、latent dropout、信息约束或更强的多样性损失。

### 方法二：直接在 latent space 做 flow matching

这更接近 TrajLoom，也更适合 MotionForesight 的稠密 latent。

训练时取真实未来 latent $z_1$，构造带噪中间状态：

$$
z_s=(1-s)z_0+sz_1+\sigma\epsilon,
$$

让模型预测从 $z_0$ 到 $z_1$ 的速度。TrajLoom 使用的就是这种条件 flow-matching 目标，并在可见未来 token 上加权训练。[Flow-Matching Objective](https://www.alphaxiv.org/abs/2603.22606?page=7)

推理时从不同 $\epsilon$ 开始，得到不同的 $z_1$，再经过 MotionForesight 的 track decoder。

### 方法三：多样本损失或 best-of-$K$

如果保留较简单的回归架构，可以一次生成 $K$ 个候选：

$$
\hat X^{(1)},\ldots,\hat X^{(K)}
=
f_\theta(V,a,z_1),\ldots,f_\theta(V,a,z_K).
$$

训练时只让最接近真实未来的候选承担重建损失：

$$
\mathcal L_{\text{best-of-}K}
=
\min_k
d(\hat X^{(k)},X_f).
$$

这能避免模型把不同未来平均到一起，但它通常不如 flow/diffusion 那样直接学习完整概率分布，而且需要额外保证候选之间有足够差异。

## 文本和随机性分别控制什么？

一个好的设计要让两者分工明确：

$$
p(X_f\mid V,a)
$$

中的：

- $V$：当前场景和已发生的运动；
- $a$：动作意图，决定未来模式；
- $z$：同一意图下的合理变化。

例如：

$$
p(X_f\mid V,\text{“打开抽屉”})
$$

应该集中在“沿滑轨打开”的轨迹，但可以有：

- 不同的速度；
- 不同的打开距离；
- 轻微的相机/物体姿态变化；
- 不同的局部遮挡和非刚性细节。

而随机变量不应该把“打开抽屉”随意变成“拿起抽屉”。如果出现这种情况，说明文本条件太弱，或者训练数据中动作标签和轨迹对应关系不够清楚。

## 我的建议

如果目标是“尽量仿照现有方法”，我会按下面优先级选择：

### 最推荐：MotionForesight decoder + TrajLoom 式 latent flow

- 保留 MotionForesight 的 dense pointmap/track decoder；
- 用一个 VAE 或现成 latent interface 压缩未来 track latent；
- 用条件 rectified flow 生成未来 latent；
- 文本通过 pooled embedding、cross-attention 或条件 mask latent 注入；
- 每次从不同随机 latent 起点采样；
- 最后用冻结 decoder 恢复稠密 3D 轨迹。

这是真正的“随机生成稠密未来”，而不是给确定性输出加噪声。

### 较小改动：文本条件的随机 mask latent

使用：

$$
m(a,z)=m^0+G(h_a)+H(h_a,z),
$$

并训练多样性相关目标。优点是改动小；缺点是随机性会被压缩到 mask latent，表达能力和可控性可能不足。

### 不推荐：只在推理时对原始 mask latent 加随机噪声

它最容易实现，但没有机制保证不同样本分别对应不同合理未来，可能只是制造轨迹噪声。

因此，最准确的结论是：

> **可以在 MotionForesight 的 mask latent 上同时加入文本条件和随机变量，但要让随机变量真正产生有意义的多样性，必须在训练目标中让它解释未来轨迹差异。现有的 TrajLoom 已经展示了一条更成熟的路线：在稠密轨迹的低维潜空间中使用条件 flow matching，并通过带噪起点采样多个未来。对 MotionForesight 来说，最自然的改造是保留其冻结的 track decoder，把确定性 future adapter 替换成文本条件的 latent flow generator。****