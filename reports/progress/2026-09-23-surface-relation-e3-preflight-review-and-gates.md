# E3 前置检查复核与监督质量闸门

- 日期：2026-09-23
- 状态：MotionForesight 接口通过；监督登记未通过；Relation Head 训练暂停

## 完成工作

- 复核 E3 前置报告、汇总 JSON、浇水壶一致 pseudo-GT 审计、视频级划分与相关构建脚本。
- 区分模型接口正确性、pseudo-GT 坐标登记质量和 Relation Head 学习可行性。
- 审查 past-only 12 帧身份映射与 12 帧训练长度 pseudo-GT 的后续方案。

## 关键判断

- `P0`、output latent 与 last-DiT hidden 的形状、有限值、observed-only 输入、时间槽位映射、采样排列一致性和确定性复跑均支持进入数据准备阶段。
- 现有 E2 pseudo-GT 与 observed-only 特征不处于同一 DA3 序列上下文，不能直接作为监督。
- 浇水壶原型的中位登记残差降至 bbox 对角线的 `3.9%`，但 `p90=21.4%`、`14/48` query 超过 `10%`，最大 dense anchor 误差约 `67.7 px`，仍不足以放行训练。
- 当前 5 个独立视频只能支持 E3-lite 的工程与表示可读性试验，不能支持正式泛化结论。

## 训练前固定条件

- 明确 past-only 12 帧和 3-history+9-future 窗口的精确索引；用于身份建立的窗口不得包含 `t0` 后信息。
- 两次 TrackCraft3R/DA3 运行之间必须通过共同 `t0` 做显式一对一 query 映射，不能依赖 dense map 下标天然一致。
- 用部分 query 估计坐标变换，其余 query 做独立登记审计，并输出逐 query 残差、anchor 误差和 validity mask。
- 建议以至少 `90%` query 的归一化残差低于 `10%`、`p90 <= 10%` 且有效 query 的 2D `k=3` 诱导图保持连通作为 E3-lite 候选闸门；阈值须在重建全部 5 视频前固定。
- 先审计全部 5 视频，通过后再生成训练缓存；首轮冻结 MotionForesight，以 output latent `H=(15,48,16)` 为主输入，比较 zero correction、geometry-only 和 query-feature 三路 probe。

## 下一步

代码实现 past-only 身份映射与一致 pseudo-GT，并只提交五视频质量审计；质量闸门通过前不启动 Relation Head 训练。正式 E3 另行扩充按 source video 隔离的独立窗口或视频。
