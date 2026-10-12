# MotionForesight 官方 Demo 失效现象审查

## 范围

- 来源：官方 [SSv2 validation](https://motionforesight.github.io/baseline-comparisons/val.html) 与 [OOD phone](https://motionforesight.github.io/baseline-comparisons/context.html) 对比页。
- 检查对象：Ground Truth、MotionForesight、MolmoMotion (lang)、MolmoMotion (no lang)。
- 方法：对 20 个 SSv2 样本和 20 个 OOD 样本按统一时间点抽帧，比较运动方向、幅度、终点、时序变化、点集结构与物理约束。

## 主要观察

1. **宏观结果和运动幅度不稳定。** MotionForesight 常能给出正确的大致运动趋势，但会明显欠预测或过预测位移；MolmoMotion 更常出现近静止、错误方向或幅度夸张。
2. **刚体与关节约束不足。** 门、抽屉、盖子、杯子等目标的预测点会出现扇形发散、剪切、弯曲或互相交叉，未稳定保持刚体形状、铰链轴或平移约束；MolmoMotion 更明显。
3. **旋转、翻转、抛掷和接触后的动力学较弱。** 快速旋转、上抛后下落、加速滚动、停止和接触阶段容易被平滑成短距离平移或局部摆动。
4. **语言只部分改善 MolmoMotion 的动作类别判断。** 无语言版本在多个移动、旋转和拿取样本上退化为近静止；有语言版本会产生更强运动，但仍常有错误的几何、幅度和终点，说明动作标签不能替代视觉动力学与几何建模。
5. **OOD 下错误更容易被放大。** MotionForesight 在部分开门、拿取和移动样本中保持较好的整体一致性，但也出现明显过冲；MolmoMotion 在近静止和夸张运动之间切换，稳定性较差。
6. **现有展示不表达未来不确定性。** 每个条件只显示一条预测，无法区分多种合理未来、模式概率或置信度校准。

## 代表样本

- `val_01` 胶带滚动：MotionForesight 明显低估滚动距离；MolmoMotion 位移更大，但点轨迹更发散。
- `val_08` 抛笔：MotionForesight 未恢复明显的抛物线式上升—下降；MolmoMotion (lang) 给出大弧线，但形状和终点偏差明显。
- `val_14` 橡皮翻转：MotionForesight 主要表现为平移/剪切；两个 MolmoMotion 版本接近静止。
- `context_06` 拉开抽屉：MotionForesight 只预测部分运动；MolmoMotion 把刚体平移预测成大幅扇形轨迹。
- `context_08` 打开冰箱门：各方法都难以同时匹配铰链旋转、位移幅度和门面刚体一致性。
- `context_09` 拿起喷雾瓶、`context_14` 拿起杯子、`context_16` 拿起背包：MotionForesight 多次出现幅度过大；MolmoMotion 的方向和结构稳定性更差。

## 证据边界

- 页面是作者选择的定性样本，不能据此估计整体失败率或方法排名。
- MolmoMotion (lang) 使用额外动作文本，和纯视频输入的 MotionForesight 不是完全同条件比较。
- 抽帧可显示方向、幅度和结构问题，但速度、加速度与接触时序仍需逐帧或数值轨迹评估。

## 对当前研究的含义

后续基线与改进应把误差分成三层：宏观未来模式/终点、刚体与接触几何、局部时序动力学。多未来模型还需单独报告覆盖率、模式概率或可校准置信度，不能只展示多条样本。
