# 讨论笔记：Masked Reconstruction、因果预测、LLM 语义与 HAR 标注困难

日期：2026-06-23

## 1. Masked reconstruction 是否类似 BERT

是的，masked reconstruction 和 BERT 的 masked language modeling 在思想上类似：

```text
遮住一部分输入 -> 根据上下文恢复被遮住的部分
```

区别在于：

- BERT 的词是离散 token，目标通常是分类预测被遮住的词。
- IMU 是连续数值时间序列，目标通常是回归重构被遮住的加速度/角速度值。

因此，IMU masked reconstruction 更像：

```text
continuous BERT / denoising autoencoder / masked time-series modeling
```

## 2. 重构缺失部分 vs 预测未来部分

这两个目标都合理，但服务的问题不同。

### Masked reconstruction

典型目标：

```text
给定一个完整窗口中未被遮住的部分，重构随机缺失的时间片、通道或 patch。
```

适合：

- 学习通用表示
- 做 HAR 分类预训练
- 缓解标签不足
- 建模通道之间和时间片之间的整体关系
- 做缺失传感器修复、去噪、异常检测

它的缺点是：模型可能看到被 mask 片段前后的上下文，因此不等价于真实在线预测。

### Forecasting / future prediction

典型目标：

```text
给定过去 L ms 的 IMU，预测未来 N ms 的 IMU 或姿态变化。
```

适合：

- OIS 姿态预测
- 冲击事件提前预警
- 在线流式推理
- 需要严格避免未来信息泄漏的任务

它的缺点是：目标更难，学习到的表示可能更偏动态外推，而不一定最适合分类。

## 3. 全局注意力还是因果注意力

这取决于任务。

### HAR 离线窗口分类

如果输入是一个已经完整收集到的 2.56s 窗口，例如 UCI HAR，那么可以使用全局 / 双向 self-attention：

```text
窗口内任意时间片都可以看见窗口内其他时间片。
```

这类似 BERT 或 MAE。它适合学习窗口级表示，再输出活动类别。

### 在线 HAR / 实时状态识别

如果希望模型每个时刻实时判断当前状态，可以使用：

- causal attention
- chunked attention
- sliding-window attention
- 有限 look-ahead attention

这样可以控制延迟。

### OIS 短时预测和冲击预警

这类任务必须更严格：

```text
模型只能看过去，不能看未来。
```

因此应该使用 causal attention 或其他因果序列模型。

## 4. 对当前项目的建议

不要把 masked reconstruction 和 forecasting 看成二选一。更合理的路线是：

```text
第一步：masked reconstruction 学表示
第二步：future prediction 学动态外推
第三步：多任务学习，把分类、未来 IMU 回归、事件检测结合起来
```

对 UCI HAR 原型：

```text
masked reconstruction + 下游 6 类 HAR 分类
```

对 OIS 方向：

```text
past IMU -> future gyro/acc regression + impact classification + arrival-time regression
```

## 5. LLM / 多模态大模型是否是近年趋势

是。近年已经出现多条与 sensor-language / sensor-vision-language 相关的路线：

- IMU2CLIP：将 IMU、视频、文本对齐到 CLIP 表示空间。
- SensorLLM：把传感器时间序列与趋势文本对齐，让 LLM 参与 HAR。
- SensorLM：大规模 wearable sensor-language foundation model。
- 2026 sensor-based HAR foundation model survey：将 LLM 用于推理、标注和人机交互列为主要发展方向之一。

这说明“用 foundation model 的世界知识补传感器语义不足”确实是上升趋势。

但应注意：它还不是 plug-and-play 解法。

主要难点：

- IMU 数值信号本身没有天然语言语义。
- sensor-text 配对数据很少。
- LLM 可能产生看似合理但没有传感器证据支撑的解释。
- 端侧实时部署成本高。

## 6. 和课题一、课题二的关系

### 课题一

人体动作、手机运镜、场景语义之间的组合关系，确实更适合引入多模态 / LLM 语义先验。

可能路线：

```text
IMU encoder + video encoder + text/LLM semantic prior
```

用途：

- 区分 IMU 波形相似但语义不同的活动
- 辅助组合标签生成
- 解释人体动作和手机运动的关系
- 做跨模态对齐和检索

### 课题二

OIS 姿态预测和冲击预警更偏低延迟控制支持。LLM 不适合直接放在实时闭环里。

更合理的用法：

- 离线辅助标注和数据整理
- 生成场景描述或冲击事件解释
- 用 foundation model 预训练表示，再蒸馏到轻量因果模型

实时模型仍应偏向：

```text
causal Transformer / TCN / GRU / lightweight state-space model
```

## 7. 为什么 HAR 标签很难获得

用户的疑问是合理的：如果让被试按指令做动作，确实可以得到标签。UCI HAR 就是这类受控采集。

但领域里说“标签难”，主要指真实世界、长期、复杂活动数据。

受控采集的问题：

- 动作类别有限，通常只覆盖走、坐、站、躺、跑等简单动作。
- 被试知道自己在被观察，会刻意做动作，和自然行为不同。
- 难覆盖长尾活动，例如做饭、收拾房间、开门、拍摄、上下车、使用工具。
- 难覆盖不同人、不同设备、不同佩戴位置和不同场景。
- 很难采集稀有事件，例如摔倒、碰撞、手机冲击。

真实世界采集的问题：

- 活动没有清晰边界，开始和结束时间很难精确标注。
- 多个活动常常重叠，例如边走边看手机、边做饭边说话。
- 参与者事后回忆会漏标、错标、时间粗糙。
- 实时手动标注会打断自然行为。
- 视频标注更准确，但有隐私、同步和人工成本问题。
- IMU 波形本身不直观，不能像图片那样轻松众包标注。

所以，HAR 的标签难题不是“不可能得到标签”，而是：

```text
高质量、长期、真实、多样、精确对齐的标签很难获得。
```

这也是自监督学习、弱监督、主动学习、LLM 辅助标注和多模态对齐越来越重要的原因。

## 8. 当前结论

当前项目可以把两条路线区分清楚：

```text
HAR 表示学习：全局注意力 + masked reconstruction
OIS 未来预测：因果注意力 + future regression / event prediction
```

更长远地看，课题一可以积极关注 LLM / 多模态语义对齐；课题二则应谨慎，只把 foundation model 用于预训练、蒸馏、标注和解释，不直接作为实时控制核心。
