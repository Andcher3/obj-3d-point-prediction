# 2026-09-04 伪分布监督、随机条件与对抗训练细化

- 日期：2026-09-04
- 完成工作：详细解释连续运动潜空间、MotionMap 式宏观概率图和模式内轨迹细化三层结构；分析能否将构造的未来分布作为监督；界定 Stochastic Conditioning、轨迹判别器、条件判别器、多样性损失和 IS 类指标的作用。
- 本地产物或改动：新增 `reports/ideas/2026-09-04-distribution-target-adversarial-training-detail.md`，并更新 `research_roadmap.md`、`reports/ideas/2026-09-04-motionmap-foundation-prior-redesign.md` 与 `reports/progress_log.md`。
- 关键结果：允许用真实后续、跨样本真实检索和外部先验构造伪分布目标；支持信息、概率信息和来源可信度分别建模。宏观概率图负责模式及概率，Stochastic Conditioning 负责模式内部随机变量利用，判别器负责边际真实性和有证据的条件兼容性。
- 关键限制：未发生未来不能统一视为负例；生成模型频率和判别器分数不能直接解释为真实概率；直接距离多样性可能制造不合理轨迹；IS 缺少适用于三维点轨迹的天然分类器，不能作为主要训练目标或唯一评测。
- 下一步：按运动潜空间、分布图、随机细化、判别器、概率校准的顺序分别验证，并进行逐项消融。
