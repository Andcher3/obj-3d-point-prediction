# 第二阶段多样性设计复核

日期：2026-09-16

## 结论

- 以 MotionForesight 作为第二阶段主干的理由成立：其输出直接是稠密、参考帧锚定的三维点轨迹，接口与项目目标一致。MolmoMotion 暂降为对照/数据资源是合理的工程选择，但小样本测试不能推出其普遍失效；TrajLoom 是稠密二维轨迹方法，跨论文指标不能直接比较。
- MotionForesight 的 Wan 原生 T5 文本 cross-attention 是第一版文本条件的首选入口。text-conditioned mask latent 可作为辅助消融，不承担主要语义通路。
- “在 latent 中引入随机性”的方向成立，但不能把 future mask latent 加噪后继续使用确定性 L1/MSE。当随机变量没有匹配的生成目标时，模型可忽略噪声，或产生不可控误差。

## 两个 AI 方案的审核

`track latent -> VAE/interface -> rectified flow -> decoder` 只有在存在可编码真实轨迹的 encoder、稳定的 latent 解码器和明确的 noisy-state 输入时才成立。MotionForesight 的视频 VAE 不等于轨迹 VAE；现有 residual-track latent 是否适合从噪声生成，必须通过重建、插值、扰动和边界连续性实验确认。

恢复 Wan text cross-attention 的判断基本正确，但当前 MotionForesight 是固定时间步的确定性 latent 回归器，不能只替换损失就成为 flow。若采用 flow，需要把 `z_t` 显式接入 query/row stream，加入时间条件，并在推理时进行 ODE 多步积分。

## 推荐的多样性路线

先保留一个确定性、文本条件的 MotionForesight 输出作为语义锚点，再在 future residual-track latent 上建立条件生成器：

```text
c = (observed video, target/history, narration)
z_base = MotionForesight(c)
r_1 = z_future_GT - z_base
r_0 ~ p0(. | history boundary)
r_t = (1-t) r_0 + t r_1
v_theta(r_t, t | c, z_base) -> integrate -> r_hat
z_hat = z_base + r_hat
TrackCraft decoder(z_hat) -> dense 3D tracks
```

该残差 flow 比直接替换整个预测器更容易保留 MotionForesight 的单未来性能和边界条件。它仍须先验证 residual latent 的尺度、可解码性和目标编码方式；若没有可靠的轨迹 encoder，则改用专用三维 trajectory autoencoder，或把 flow 放在归一化轨迹坐标空间作为备选。

第一阶段 narration 负责跨事件差异；同一 narration 下的随机初值负责速度、幅度、路径和终点等模式内差异。可见性等离散成分不必强行与连续 xyz 共用一个高斯 flow，可先保留原有 head 或使用独立 Bernoulli 头。

## 必须保留的未决问题

1. 单个视频通常只有一个记录未来。Flow 可以从相似样本学习条件分布，但不能仅凭单一 GT 保证同一观测下出现多个真实事件；需要用分组覆盖、校准和语义一致性评估验证。
2. 仅改变随机种子只能证明采样差异，不能证明概率质量或校准。第一版应报告样本覆盖、近邻/模式质量、文本一致性和多样性，再讨论概率解释。
3. 文字条件和视频条件可能发生竞争。恢复 cross-attention 后仍需做同视频文本置换、hard-negative 和基线精度对照，确认文本确实改变宏观运动。
4. flow 的多步推理会增加成本；应先比较少量 ODE 步数与确定性基线，不预设 TrajLoom 的二维结构或损失权重可以直接迁移。

## 建议验证顺序

1. 复现确定性 MotionForesight，并导出真实 future residual 的 latent 编解码与边界检查。
2. 恢复 T5 和 cross-attention LoRA，先完成文本置换敏感性实验。
3. 无文本的 residual latent flow 基线，确认不同种子产生连续、可解码的三维样本。
4. 合并 narration 与 flow，分开评估跨事件差异和同事件内差异。
5. 最后加入 mask modulation、boundary hints、时空一致性和 on-policy fine-tuning 等增强项。

本轮未运行新训练或远程实验；上述架构仍属于待验证候选。
