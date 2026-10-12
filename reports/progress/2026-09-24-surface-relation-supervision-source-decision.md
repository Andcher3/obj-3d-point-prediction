# 监督来源缺失后的路线决策

- 日期：2026-09-24
- 状态：停止继续搜索公开官方 sparse 标签；真实视频 Relation Head 仍不放行

## 本轮回复的审查

代码 AI 只再次确认：公开 MotionForesight 仓库、checkpoint 包和 TrackCraft3R 子模块没有找到 `*_sparse.npz` 生成器、训练标签下载入口或公开 `L_track` 构造脚本。重建适配器通过了字段、形状和 validity 的格式检查。

本轮没有提供此前要求的完整 `T=22`、`7+15` 官方协议兼容审计结果，也没有新的未来 15 步标签质量 JSON。因此不能把任务状态写成“官方协议兼容监督链通过”，也不能放行 decoded-coordinate relation-loss pilot。

## 决策调整

1. 不再继续无边界地搜索公开官方 sparse 文件；当前公开资料不足以恢复官方训练标签。
2. 保留重建适配器和样本作为 schema/读取器测试，文件必须继续标记为 reconstructed，不能标记 `relation_loss_ready=true`。
3. 当前 TrackCraft3R/DA3 pseudo-GT 只能用于诊断、格式和数值工程测试，不能作为真实视频 Relation Head 监督。
4. 真实 Relation Head、E4、IRLS、Huber 和 learned gate 继续暂停；edge head 也不冻结。
5. 后续研究转向两条可验证路线：
   - 复现官方训练数据或获得作者提供的同坐标 sparse 标签；
   - 若无法获得，则基于有明确来源的完整视频轨迹，重新设计独立的标签质量验证，并把结果限定为替代监督研究，不称为 MotionForesight 官方监督。

## 当前研究结论

监督来源缺失已经成为数据可得性阻塞，不是继续增加登记器或鲁棒积分器可以解决的算法问题。点间相关运动假设、图积分数学和 Relation Head 工程接口仍保留为研究候选，但真实视频收益没有证据支持。
