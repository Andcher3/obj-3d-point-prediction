# 方向一创新点候选：从三篇论文到可落地研究问题

日期：2026-06-27

## 1. 先给结论

目前最值得沿着方向一推进的创新主线不是“再设计一个更大的 HAR 分类器”，而是：

```text
手机 IMU / 视频
-> 人体动作 + 手机/相机运动 + 场景语义
-> 组合行为表示与识别
```

三篇论文分别解决了这个问题的不同部分：

- SensorLM：传感器与语言语义对齐，但数据是长期 wearable，不是手机短窗口和相机运动。
- SensorLLM：短窗口 motion sensor 与 LLM 对齐，但最终仍主要是封闭集 HAR 分类，趋势文本也没有充分覆盖周期、频域和跨通道关系。
- EVI-MAE：视频 + body-worn IMU 的多模态自监督，但重点是身体佩戴 IMU 与第一视角视频，不是手机作为相机时的运镜行为建模。

因此，可以抓住的研究空白是：

```text
现有 sensor-language / video-IMU 方法还没有专门处理手机拍摄场景中的
人体动作、手机/相机运动和场景语义之间的组合关系。
```

## 2. 创新点生成公式

不要从“我要发明一个模型”开始，而是从下面这个结构开始：

```text
已有方法 M
没有处理的场景/假设 G
我的课题约束 A
我提出的可验证改动 P
可以被实验检验的收益 C
```

例如：

```text
M: SensorLLM 用 trend QA 做 sensor-language alignment。
G: trend QA 只描述上升/下降/稳定，缺少周期性、频域和跨轴关系。
A: 手机 IMU HAR 中，走路/上下楼/运镜抖动强依赖周期和跨轴耦合。
P: 构造 micro-motion caption，加入频率、能量、重力方向、轴间相关性。
C: 在 few-shot、跨受试者或相似动作区分中优于只用 trend caption。
```

## 3. 候选创新方向

### 方向 A：Micro-motion Caption Alignment

核心想法：

```text
不只把 IMU 描述成 rising / falling / stable，
而是生成更适合 HAR 的短窗口运动 caption。
```

可自动生成的 caption 包括：

- 趋势：上升、下降、稳定、突变。
- 周期：是否存在明显周期、主频范围、周期强度。
- 能量：窗口内加速度/角速度波动强弱。
- 姿态：total_acc 平均方向反映的重力姿态。
- 跨轴关系：x/y/z 轴是否同步变化，acc 与 gyro 是否耦合。

实验落地：

- 数据：UCI HAR。
- 模型：小型 IMU encoder + text encoder，先做 contrastive alignment，再做 HAR 分类。
- 对比：无 caption、trend caption、micro-motion caption。
- 重点指标：few-shot、跨 subject 测试、上楼/下楼/走路相似类别混淆。

优点：

- 最接近当前已有数据和能力。
- 不需要大模型或视频数据。
- 可以作为 SensorLLM 的明确改进点。

风险：

- 创新幅度偏中等，需要实验设计证明 caption 类型确实有用。

### 方向 B：Human Motion 与 Camera Motion 的解耦表示

核心想法：

```text
手机 IMU 同时包含人体运动和手机/相机运动。
方向一真正需要的不是单一 activity label，而是分解出两个因素：
human motion tokens + camera motion tokens。
```

可能的模型设计：

- 一个共享 IMU encoder。
- 两组 latent tokens：
  - body-motion tokens：更关注步态、姿态、周期。
  - camera-motion tokens：更关注旋转、平移、抖动、运镜。
- 用不同任务约束它们：
  - 人体动作分类。
  - 相机运动分类或回归。
  - 两组 token 之间加弱解耦约束或互信息约束。

实验落地：

- 初期：用 UCI HAR 做 body-motion 分支 sanity check。
- 后续：自采手机视频 + IMU，小规模标注 camera motion，如 static、pan、tilt、walk-shake、raise/lower。
- 弱标签来源：gyro、optical flow、视频稳定性指标。

优点：

- 和方向一高度一致。
- 比普通 HAR 分类更有研究辨识度。

风险：

- 需要新数据或伪标签构造，工作量高于方向 A。

### 方向 C：Video-IMU Masked Modeling for Phone-as-Camera

核心想法：

```text
借鉴 EVI-MAE，但把任务从 body-worn IMU + egocentric action
改成 phone-held camera + phone IMU 的联合表示学习。
```

可设计的预训练目标：

- masked video patch reconstruction。
- masked IMU spectrogram reconstruction。
- video -> IMU motion prediction。
- IMU -> video motion/optical-flow prediction。
- video 与 IMU 的跨模态对比学习。

相比 EVI-MAE 的潜在差异：

- IMU 不再是身体多点佩戴，而是手机自身运动。
- 视频里的运动同时来自人体移动、相机运动、场景物体运动。
- 需要显式处理相机运动与人体运动的混合。

实验落地：

- 第一阶段可先不做语言，只做视频-IMU 自监督。
- 后续再加入 camera motion caption 或 action caption。

优点：

- 更贴近手机视频场景。
- 能自然连接方向一和第三方向的视频退化/重建。

风险：

- 数据、算力、工程复杂度较高。

### 方向 D：组合行为的语言标签空间

核心想法：

```text
不用把类别限制为 walking / sitting，
而是构造组合标签：
human activity + phone/camera motion + context。
```

示例：

```text
walking + camera panning left
standing + camera tilting up
walking upstairs + strong vertical shake
sitting + phone rotation
```

可借鉴 SensorLM / SensorLLM：

- 用文本 prompt 表示组合类别。
- sensor/video encoder 与文本类别 embedding 对齐。
- 做 zero-shot 或 few-shot 组合识别。

优点：

- 更接近“现实中有意义”的开放行为理解。
- 能解释为什么 language alignment 对方向一有价值。

风险：

- 需要定义合理的组合标签体系，否则容易变成概念堆砌。

## 4. 当前最建议的路线

按可落地程度排序：

1. 先做方向 A：Micro-motion Caption Alignment。
2. 再扩展到方向 B：Human Motion / Camera Motion 解耦。
3. 数据和工程成熟后，再推进方向 C 或 D。

原因：

- 方向 A 能直接从 UCI HAR 开始，不依赖新数据。
- 方向 A 可以形成明确实验：caption 类型消融、few-shot、相似动作混淆。
- 方向 A 的结果可以自然过渡到方向 B：把 micro-motion caption 中的周期/重力/旋转特征分别分给 body-motion 和 camera-motion。

## 5. 一个可以写进组会的初步题目

```text
面向手机 IMU 行为理解的短窗口运动语义对齐：
从趋势描述扩展到周期、姿态与跨轴关系的 Micro-motion Caption 预训练
```

更贴近方向一的版本：

```text
面向手机拍摄场景的人体动作与相机运动联合建模：
基于 IMU 微运动语义 caption 的多任务表示学习
```

## 6. 下一步具体行动

1. 在 UCI HAR 上跑一个标准监督 baseline，例如 1D CNN 或小型 Transformer。
2. 编写自动 micro-motion caption 生成器。
3. 训练三个版本：
   - baseline：IMU encoder + classifier。
   - trend-caption alignment：只用上升/下降/稳定。
   - micro-motion-caption alignment：加入周期、能量、姿态、跨轴关系。
4. 对比：
   - 全监督准确率/F1。
   - few-shot 每类 5/10/20 样本。
   - subject-wise split。
   - walking / upstairs / downstairs 混淆矩阵。
5. 如果 micro-motion caption 有收益，再进入手机视频 + IMU 小数据采集。

