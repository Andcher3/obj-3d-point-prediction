# TrajLoom 文本条件矩阵实验报告

日期：2026-09-16  
实验状态：已完成推理与描述性分析，未进行训练或概率校准。

## 目的

在同一视觉输入上比较无文本、场景一致文本、静止文本和反向文本；固定噪声种子 42 测试文本条件敏感性，并对场景一致文本使用种子 42/43/44 测试随机分支变化。结果只用于判断输出是否改变，不把单次样本解释为概率分布。

## 协议

- 视频：`10790317943435558913`、`gear_v2_9-wrist_front_rgb_img-0`
- 历史/未来：81/81 帧；Euler steps=20；分辨率 832×480
- 每个视频 4 个固定 seed=42 条件；一致文本额外 seed=43、44
- 文本编码：本地 UMT5-XXL pooled embedding
- GT 形状统一为时间优先后取未来 81 帧；原始 GT 为 `(390,162,2)`，分析脚本已修正该对齐路径。

## 结果

### 视频 `10790317943435558913`

| 条件 | seed | ADE (px) | FDE (px) | 平均路径步长 (px) |
|---|---:|---:|---:|---:|
| none | 42 | 70.122 | 118.947 | 1.427 |
| consistent | 42 | 78.006 | 136.435 | 1.142 |
| consistent | 43 | 71.900 | 122.816 | 1.405 |
| consistent | 44 | 75.794 | 131.561 | 1.208 |
| static | 42 | 78.912 | 138.382 | 1.085 |
| conflict | 42 | 74.485 | 128.373 | 1.321 |

固定 seed=42 下，none 与 consistent 的轨迹 ADE 差为 11.563 px，none 与 static 为 13.454 px，none 与 conflict 为 7.572 px。consistent 的跨种子轨迹差异 ADE 为 8.135–20.515 px。

### 视频 `gear_v2_9-wrist_front_rgb_img-0`

| 条件 | seed | ADE (px) | FDE (px) | 平均路径步长 (px) |
|---|---:|---:|---:|---:|
| none | 42 | 50.200 | 110.772 | 0.851 |
| consistent | 42 | 50.317 | 111.560 | 0.844 |
| consistent | 43 | 52.429 | 111.081 | 0.859 |
| consistent | 44 | 53.259 | 112.012 | 0.855 |
| static | 42 | 50.563 | 112.620 | 0.797 |
| conflict | 42 | 49.802 | 110.779 | 0.834 |

固定 seed=42 下，none 与 consistent 的轨迹 ADE 差为 3.485 px，none 与 static 为 2.420 px，none 与 conflict 为 5.291 px。consistent 的跨种子轨迹差异 ADE 为 9.140–9.947 px。

## 判断边界

1. 12/12 次推理成功，固定 seed 改变文本时输出哈希和轨迹均改变，说明当前 TrajLoom 推理链路确实读取文本条件。
2. 一致文本改变随机种子时轨迹也改变，说明生成器保留随机分支。
3. 两个视频中，文本条件的影响幅度不同；第二个视频的文本效应小于随机种子差异，不能据此声称已经实现事件级多模态或可靠指令遵循。
4. ADE/FDE 仅是对已有单一未来伪标签的描述性比较；本实验没有密度、模式质量、覆盖率、校准误差或概率质量评估。
5. 运行日志出现 tokenizer legacy/regex 警告，但 12 次均正常生成；该警告仍应在后续正式实验中固定 tokenizer 配置并单独做敏感性检查。

## 方向性复核

以预测段首尾点位移的平均向量作为粗略方向指标（图像坐标：x 向右、y 向下）：

- `10790317943435558913`：consistent `(-17.2,+44.6)`、static `(-18.2,+42.5)`、conflict `(-31.4,+45.9)` px；三者均为左下方向，consistent 与 static 的方向夹角约 2.1°，conflict 与 consistent 约 13.2°。
- `gear_v2_9-wrist_front_rgb_img-0`：consistent `(+16.9,-3.1)`、static `(+20.1,-2.0)`、conflict `(+14.7,-2.6)` px；三者均为右侧轻微上移，条件间方向夹角不超过 4.6°。

因此，当前 probe 中 conflict 文本没有诱导出与 consistent 相反的运动方向；文本影响主要体现为位移幅度和轨迹细节变化，视觉条件对方向的约束更强。

## 文件

- 指标：[text_matrix_metrics.json](../../../archive/experiments/trajloom/TrajLoom-probe/analysis/text_matrix_metrics.json)
- 标准输出：[text_matrix_stdout.txt](../../../archive/experiments/trajloom/TrajLoom-probe/analysis/text_matrix_stdout.txt)
- 推理日志：[text_matrix.log](../../../archive/experiments/trajloom/TrajLoom-probe/text_matrix.log)
- 分析脚本：[analyze_text_matrix.py](../../../archive/experiments/trajloom/TrajLoom-probe/analyze_text_matrix.py)

## 下一步

扩大视频数量并加入文本-视觉事件一致性标注；对每个条件使用可变样本数，报告轨迹集合覆盖、条件间宏观运动差异和校准指标。当前结果只能作为文本条件 smoke/probe，不足以支持概率建模或语义控制结论。
