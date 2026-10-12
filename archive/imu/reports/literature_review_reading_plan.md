# IMU/HAR 与时间序列深度学习综述阅读计划

本文用于规划接下来如何通过综述快速进入前两个课题。

当前优先目标：

```text
先建立 IMU/HAR 深度学习方法地图
再扩展到短时时序预测、冲击事件预测和 OIS 控制背景
```

## 1. 推荐先读的综述

### 1. Deep Learning in Human Activity Recognition with Wearable Sensors: A Review on Advances

链接：

- <https://pmc.ncbi.nlm.nih.gov/articles/PMC8879042/>
- <https://arxiv.org/abs/2111.00418>

推荐优先级：最高。

为什么先读：

- 和当前 UCI HAR 最贴近。
- 主题就是 wearable/sensor-based HAR 中的深度学习。
- 能建立 CNN、RNN/LSTM、混合模型、注意力、自监督等方法分类。
- 适合从“我已经会看 IMU 曲线”过渡到“我知道该用哪些模型”。

阅读目标：

- 弄清 HAR 的标准流程。
- 弄清深度学习相比传统特征工程解决了什么。
- 建立常见模型类别表：CNN、RNN、CNN-RNN、Autoencoder、Attention、Transformer、Self-supervised。
- 记录挑战：泛化、标注、数据分布变化、端侧部署、实时性。

### 2. Deep Learning for Sensor-based Human Activity Recognition

链接：

- <https://dl.acm.org/doi/10.1145/3447744>

推荐优先级：高。

为什么读：

- 这是 sensor-based HAR 的深度学习综述，主题比普通 HAR 更接近 IMU。
- 可以作为第一篇的补充，帮助确认领域共识。

阅读目标：

- 梳理 sensor-based HAR 的模型分类。
- 看它如何组织数据集、任务和评价指标。
- 对照第一篇综述，找共同反复出现的方法。

### 3. Deep Learning for Time Series Classification and Extrinsic Regression: A Current Survey

链接：

- <https://arxiv.org/abs/2302.02515>
- <https://dl.acm.org/doi/10.1145/3649448>

推荐优先级：高。

为什么读：

- UCI HAR 本质上是多变量时间序列分类。
- 这篇不局限于 HAR，能帮你理解更通用的时间序列深度学习方法。
- 对后续 InceptionTime、ResNet、FCN、Transformer 等强模型很有帮助。

阅读目标：

- 弄清时间序列分类中的主流网络架构。
- 学会把 HAR 放到更大的 TSC 框架中。
- 了解 extrinsic regression，为第二个课题的短时姿态/IMU 预测做铺垫。

### 4. Time-series Forecasting with Deep Learning: A Survey

链接：

- <https://royalsocietypublishing.org/rsta/article/379/2194/20200209/41189/Time-series-forecasting-with-deep-learning-a>

推荐优先级：中高。

为什么读：

- 第二个课题不只是分类，而是预测未来 N ms 的 IMU/姿态。
- 这篇能建立 forecasting 的基本模型地图。

阅读目标：

- 区分 one-step、multi-step、multi-horizon forecasting。
- 理解 encoder-decoder、CNN、RNN、attention 等预测框架。
- 记录预测任务常用指标：MAE、RMSE、horizon-wise error。

### 5. Transformers in Time Series: A Survey

链接：

- <https://www.ijcai.org/proceedings/2023/0759.pdf>

推荐优先级：中。

为什么读：

- Transformer 是前沿方法的重要分支。
- 但对于 UCI HAR 这种短窗口小数据，不一定是第一优先实现对象。
- 更适合作为“知道前沿在做什么”的综述。

阅读目标：

- 看 Transformer 如何用于 forecasting、classification、anomaly detection。
- 关注 patching、efficient attention、channel independence 等思想。
- 和 PatchTST 建立联系。

### 6. Deep Learning for Inertial Positioning: A Survey

链接：

- <https://arxiv.org/html/2303.03757v3>

推荐优先级：中。

为什么读：

- 它不是 HAR 综述，但非常 IMU-specific。
- 对理解 IMU 深度学习、惯性传感器误差、漂移、融合、定位/姿态估计有帮助。
- 对第二个课题的姿态预测和 OIS 背景更有启发。

阅读目标：

- 理解 IMU 数据的噪声、漂移和传感器融合问题。
- 看深度学习如何用于 inertial tracking/positioning。
- 提取对短时姿态预测有用的建模思想。

## 2. OIS 背景补充材料

OIS 方向更工程化，专门面向“深度学习 + OIS 预测”的综述较少。建议先用行业白皮书建立背景。

### STMicroelectronics OIS White Paper

链接：

- <https://www.st.com/resource/en/white_paper/ois_white_paper.pdf>

用途：

- 理解 OIS 是什么。
- 理解 lens shift、actuator、gyro feedback、控制延迟等基本概念。
- 不作为深度学习综述，而作为工程背景阅读。

### ROHM OIS White Paper

链接：

- <https://www.rohm.com/documents/11303/41217/OIS-white-paper.pdf>

用途：

- 补充理解 OIS 机构和控制方式。
- 帮助把“IMU 短时预测”与“光学防抖控制”连接起来。

## 3. 建议阅读顺序

第一阶段：理解 HAR 深度学习

```text
1. Zhang et al. wearable HAR deep learning review
2. ACM sensor-based HAR deep learning survey
3. Time series classification current survey
```

第二阶段：理解短时预测与第二课题

```text
4. Time-series forecasting with deep learning
5. Deep learning for inertial positioning
6. Transformers in time series
```

第三阶段：理解 OIS 工程背景

```text
7. ST OIS white paper
8. ROHM OIS white paper
```

## 4. 第一篇综述的阅读方式

第一篇建议读：

```text
Deep Learning in Human Activity Recognition with Wearable Sensors: A Review on Advances
```

第一轮不要逐字精读。按下面顺序读：

1. Abstract：弄清这篇综述覆盖什么。
2. Introduction：弄清 HAR 为什么重要，以及深度学习为什么被引入。
3. Taxonomy/Method sections：重点看模型分类。
4. Challenges/Future directions：记录和我们课题相关的问题。
5. Tables/Figures：优先看表格和总览图，它们通常比正文更快建立框架。

第一轮只回答 5 个问题：

- HAR 的标准任务流程是什么？
- 传统方法和深度学习方法的核心区别是什么？
- CNN、RNN、CNN-RNN 各自解决什么问题？
- 当前领域主要挑战是什么？
- 哪些挑战和我们的两个课题最相关？

## 5. 阅读产出模板

每读一篇综述，建议形成一个独立笔记：

```text
reports/literature/YYYY-MM-DD-paper-short-name.md
```

每篇笔记包含：

- 论文信息
- 为什么读它
- 文章结构
- 关键概念
- 方法分类
- 与课题一的关系
- 与课题二的关系
- 可以转化成实验的想法
- 不懂的问题

## 6. 当前下一步

现在就从第一篇开始：

```text
Deep Learning in Human Activity Recognition with Wearable Sensors: A Review on Advances
```

阅读重点不是记住所有论文，而是先建立模型地图：

```text
CNN -> RNN/LSTM/GRU -> CNN-RNN/DeepConvLSTM -> Attention/Transformer -> Self-supervised
```
