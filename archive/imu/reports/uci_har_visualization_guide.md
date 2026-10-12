# UCI HAR IMU 曲线图观察指南

本次已经把 UCI HAR 训练集整理为统一的三维形式，并为 6 类动作分别画了 4 个样本。当前图册包含两组视角：

- `body_acc + body_gyro`：观察去重力后的运动加速度和角速度。
- `total_acc`：观察保留重力分量后的总加速度。

## 1. 训练集现在被组织成什么形式

整理后的训练集位于：

```text
data/processed/uci_har/
```

核心文件是：

```text
X_train_raw_float32.bin
shape.json
train_metadata.csv
channel_names.txt
```

其中 `X_train_raw_float32.bin` 的逻辑形状是：

```text
7352 x 128 x 9
```

三个维度分别表示：

- `7352`：训练集中的窗口样本数
- `128`：每个 2.56 秒窗口里的时间采样点
- `9`：IMU 通道数

9 个通道顺序见：

```text
data/processed/uci_har/channel_names.txt
```

当前顺序是：

```text
total_acc_x
total_acc_y
total_acc_z
body_acc_x
body_acc_y
body_acc_z
body_gyro_x
body_gyro_y
body_gyro_z
```

如果以后使用 NumPy，可以这样读取：

```python
import numpy as np

X = np.fromfile("data/processed/uci_har/X_train_raw_float32.bin", dtype="<f4")
X = X.reshape(7352, 128, 9)
```

## 2. body 和 total 的区别

`body_acc` 和 `total_acc` 的核心区别是：

```text
total_acc = 身体运动加速度 + 重力分量
body_acc  = 从 total_acc 中分离出重力后，更偏身体运动本身的加速度
```

所以：

- 看 `body_acc/body_gyro`：更适合观察走路、上楼、下楼这类动态行为。
- 看 `total_acc`：更适合观察坐、站、躺这类静态姿态，因为重力方向会随手机姿态变化。

## 3. 这次“画图”到底画的是什么

每一个样本是一个 2.56 秒的时间窗，里面有 128 个采样点。

所谓“画一个样本”，就是：

```text
横轴：这个 2.56 秒窗口内部的时间，从 0 到 2.56 秒
纵轴：某个传感器通道在每个采样点上的数值
曲线：传感器数值随时间变化的轨迹
```

这次每类动作画了 4 个训练样本。每张图对应一种动作，每一行是一个样本。

第一组图：

```text
左侧：body_acc_x/y/z，也就是去除重力后的身体运动加速度
右侧：body_gyro_x/y/z，也就是三轴角速度
```

第二组图：

```text
total_acc_x/y/z，也就是保留重力分量的三轴总加速度
```

颜色含义：

```text
蓝色：x 轴
红色：y 轴
绿色：z 轴
```

注意：每个小图的纵轴会根据当前样本自动缩放，所以第一眼先比较“波形形状”和“周期性”，不要急着比较不同小图之间的绝对高度。

## 4. 去哪里看图

总览网页：

```text
reports/figures/uci_har/index.html
```

单独图文件：

```text
reports/figures/uci_har/walking_samples.svg
reports/figures/uci_har/walking_upstairs_samples.svg
reports/figures/uci_har/walking_downstairs_samples.svg
reports/figures/uci_har/sitting_samples.svg
reports/figures/uci_har/standing_samples.svg
reports/figures/uci_har/laying_samples.svg
```

对应的 `total_acc` 图文件：

```text
reports/figures/uci_har/walking_total_acc_samples.svg
reports/figures/uci_har/walking_upstairs_total_acc_samples.svg
reports/figures/uci_har/walking_downstairs_total_acc_samples.svg
reports/figures/uci_har/sitting_total_acc_samples.svg
reports/figures/uci_har/standing_total_acc_samples.svg
reports/figures/uci_har/laying_total_acc_samples.svg
```

可以直接用浏览器打开 `index.html`，总览中已经同时包含这两组图。

## 5. 第一轮应该观察什么

先比较动态动作：

```text
WALKING
WALKING_UPSTAIRS
WALKING_DOWNSTAIRS
```

重点看：

- `body_acc` 是否有明显周期性波动
- `body_gyro` 是否也呈现周期性变化
- 上楼、下楼、普通走路的波峰和节奏是否有差异
- 同一个动作的 4 个样本是否完全一致，还是存在人和姿态差异

再比较静态动作：

```text
SITTING
STANDING
LAYING
```

重点看：

- `body_acc` 是否整体更平稳
- `body_gyro` 是否接近 0 或变化很小
- 坐、站、躺之间是否不如动态动作那么容易从 `body_acc/body_gyro` 直接区分

这里有一个很重要的直觉：

```text
SITTING / STANDING / LAYING 的主要差异往往和重力方向有关。
```

但本次图里优先画的是 `body_acc`，它已经去掉了重力成分，所以静态姿态之间的差异可能不如动态动作明显。后面可以再专门画 `total_acc_x/y/z` 来观察姿态差异。

再看 `total_acc`：

- 对动态动作，观察它是否仍然保留周期性。
- 对静态动作，观察 x/y/z 三轴的整体水平是否不同。
- 特别比较 `SITTING`、`STANDING`、`LAYING`，看它们是否比在 `body_acc` 中更容易区分。

## 6. 这一步和研究方向的关系

对课题一来说，这一步帮助你理解：

```text
IMU 波形如何对应人体动作
```

对课题二来说，这一步帮助你理解：

```text
IMU 波形如何随时间变化，以及哪些变化可能值得预测
```

所以现在的目标不是判断模型效果，而是先让自己建立传感器直觉。

## 7. 下一步建议

建议你先打开总览网页，按这个顺序观察：

1. 看 `WALKING`，找周期性。
2. 看 `WALKING_UPSTAIRS` 和 `WALKING_DOWNSTAIRS`，和普通走路比较。
3. 看 `SITTING/STANDING/LAYING`，体会静态动作为什么更难只靠 `body_acc/body_gyro` 区分。
4. 记录 3 到 5 条你自己的观察。

当前已经补充了：

```text
total_acc_x/y/z
```

接下来建议你重点看 `SITTING`、`STANDING`、`LAYING` 的 `total_acc` 图，尝试回答：

- 哪个轴的平均水平差异最大？
- 坐和站是否仍然接近？
- 躺下是否更容易通过重力方向区分？

如果单看样本曲线不明显，优先看：

```text
reports/figures/uci_har/total_acc_summary.svg
```

这张图把每类动作的 `total_acc_x/y/z` 均值和窗口内波动强度画在一起，更适合横向比较类别差异。
