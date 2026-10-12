# MolmoMotion 文本条件复测：校正时间协议与两视频验证

## Material Passport

- **Verification Status:** VERIFIED
- **实验对象：** `MolmoMotion-4B-H3-F30` + DA3-BASE + TrackCraft3R
- **样本：** Ego4D 剪刀交互、手持浇水壶行走，共 2 段视频
- **代码：** `D:\Pycharm\post_graduate_project\archive\experiments\molmomotion\MolmoMotion-tracker3d-retest\final\run_tracker3d_molmo_retest.py`
- **完整指标：** `D:\Pycharm\post_graduate_project\archive\experiments\molmomotion\MolmoMotion-tracker3d-retest\final\results\retest-v1\aggregate_metrics.json`
- **复跑指标：** `D:\Pycharm\post_graduate_project\archive\experiments\molmomotion\MolmoMotion-tracker3d-retest\final\results\retest-rerun-greedy\aggregate_metrics.json`
- **验证记录：** `D:\Pycharm\post_graduate_project\archive\experiments\molmomotion\MolmoMotion-tracker3d-retest\final\VERIFICATION.txt`
- **验证摘要：** 11/11 自动检查通过；6 个 greedy 结果逐字节复现。

## 结论

校正协议后，MolmoMotion 不再对不同文本输出完全相同的静止轨迹。两个视频中，每组三条文本的 greedy 轨迹哈希均不同；剪刀样本中，“保持不动”产生的路径长度为 `0.0022 m`，“拿起剪刀”为 `0.0593 m`，说明动作文字确实进入并改变了解码结果。

当前结果只支持“模型具有文本条件敏感性”。动作遵循和轨迹精度仍不可靠：

- 剪刀真实后续伪标签路径长度为 `0.2123 m`，但“拿起”预测的 FDE 为 `0.1518 m`，略差于“保持不动”的 `0.1389 m`；“多步骤”虽然改变了运动幅度，却没有可靠复现要求的方向与事件顺序。
- 浇水壶真实后续伪标签路径长度为 `0.0840 m`，“继续携带”预测路径达到 `1.2008 m`，FDE 为 `0.2668 m`，明显过度运动。三种文字的 greedy 最终方向仍高度接近，方向余弦为 `0.9731–0.9984`，文字主要改变幅度。

因此，MolmoMotion 可以继续作为第二阶段的候选接口；当前零样本的 DA3+TrackCraft 输入不能证明它能稳定执行自由文本、多步骤或反事实动作条件。

## 校正后的测试协议

1. 原视频为 30 fps，按每 2 帧采样一次，匹配 MolmoMotion 的 15 fps 时间尺度。
2. `H=3` 的历史分别为剪刀 `[171,173,175]`、浇水壶 `[131,133,135]`。
3. TrackCraft3R 对 12 帧运行。MolmoMotion 只接收前 3 帧；后 9 帧仅用于离线三维轨迹伪标签评价，没有进入预测条件。
4. 每个对象使用 8 个查询点；每段视频测试真实动作、反事实动作和较长多步骤动作。
5. 每个文本运行 greedy 和一次 `temperature=0.8, top-p=0.95, top-k=50` 的随机解码。随机结果仅用于确认解码分支可以变化，不据此估计分布或比较概率。
6. 第三个 Ego4D 样本没有清晰、持续的人—物运动目标，因此未用于本轮；两段有效运动样本满足本轮 2–3 个样本的要求。

## 主要结果

| 视频 | 文本 | greedy 路径长度 (m) | 9 帧伪 GT ADE (m) | 9 帧伪 GT FDE (m) |
|---|---|---:|---:|---:|
| 剪刀 | 拿起剪刀 | 0.0593 | 0.0775 | 0.1518 |
| 剪刀 | 保持不动 | 0.0022 | 0.0722 | 0.1389 |
| 剪刀 | 拿起→向左→放下 | 0.2976 | 0.0838 | 0.1734 |
| 浇水壶 | 继续携带 | 1.2008 | 0.1176 | 0.2668 |
| 浇水壶 | 放到地面 | 0.5883 | 0.1432 | 0.3374 |
| 浇水壶 | 停步→降低→松手 | 0.3064 | 0.0984 | 0.1995 |

两段离线 TrackCraft 伪标签都包含实际运动：剪刀 `0.2123 m`，浇水壶 `0.0840 m`。这排除了上次实验中“评价窗口内目标基本静止”的主要混淆。

![剪刀轨迹对比](D:\Pycharm\post_graduate_project\archive\experiments\molmomotion\MolmoMotion-tracker3d-retest\final\results\retest-v1\sewing_scissors\prediction_vs_pseudogt.png)

![浇水壶轨迹对比](D:\Pycharm\post_graduate_project\archive\experiments\molmomotion\MolmoMotion-tracker3d-retest\final\results\retest-v1\watering_can\prediction_vs_pseudogt.png)

## 对上轮“全部静止”的解释

上轮剪刀输入的最后历史帧约为原视频第 55 帧，接下来的约 2 秒内剪刀仍在桌面上；真正拿起发生在更晚的位置。同时，上轮 30 fps 视频每 5 帧取一帧，相当于约 6 fps，与模型的 15 fps 训练时间尺度不一致。校正到交互发生前并使用 15 fps 后，确定性输出随文本改变，说明上轮结果不能归因于“MolmoMotion 完全忽略文字”。

## 仍然存在的限制

- TrackCraft 使用未来帧得到的是离线伪标签，不是真实物理三维真值。
- MolmoMotion 官方数据管线使用 ViPE 与 AllTracker；DA3-BASE + TrackCraft3R 存在尺度和输入分布差异。
- 每个文本只有一次随机样本，不能估计模式概率、覆盖率或校准性。
- 两个视频只能验证当前输入上的现象，不能推出跨数据集泛化能力。
- 当前没有证据证明模型理解了完整多步骤事件；输出可能只响应动作词并改变运动幅度。

## 11 项推断风险检查

1. 未把“输出不同”写成“指令被正确执行”：通过。
2. 未把 TrackCraft 伪标签写成真实三维真值：通过。
3. 未用 observed-action 文本自动代表最佳预测：通过。
4. 未将一次随机样本解释为条件分布：通过。
5. 未将两个视频解释为普遍泛化：通过。
6. 未将 DA3+TrackCraft 当作官方 ViPE+AllTracker 等价输入：通过。
7. 未把生成序列分数解释为轨迹概率：通过。
8. 未以可视化替代 ADE/FDE 数值：通过。
9. 文本效应以 greedy 同输入确定性对照判断：通过。
10. 明确隔离了 MolmoMotion 历史输入与离线未来伪标签：通过。
11. 明确记录第三样本未采用的原因：通过。

**检查结果：11/11 PASS。**


