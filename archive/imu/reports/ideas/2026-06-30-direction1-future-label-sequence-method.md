# 方向一方法草案：基于 Future Patch Label Sequence 的动作延续与切换预测

日期：2026-06-30

## 1. 方法核心

当前不把动作切换设计成独立的 `<A-2-B>` 主标签，而是让模型预测未来一段 patch 的动作标签序列：

```text
past IMU patches -> future patch label sequence
```

切换时机、切换概率、下一动作都从未来 label 序列及其分类概率中推导。

## 2. 输入形式

原始手机 IMU：

```text
acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
```

连续信号按时间切成 patch：

```text
X = [x_1, x_2, ..., x_T]
x_t in R^(P x C)
P = patch 内采样点数
C = IMU 通道数
```

每个 patch 被映射为一个 IMU token：

```text
z_t = PatchEmbed(x_t)
```

## 3. 模型结构

推荐的基础结构：

```text
IMU stream
-> patching
-> patch embedding / tokenizer
-> temporal encoder
-> future label decoder
-> patch-level classification head
```

方向一后续可扩展为双标签：

```text
human_action_head
phone_motion_head
```

即输出：

```text
future human label sequence
future phone-motion label sequence
```

第一阶段可先只做人体动作标签序列，降低复杂度。

## 4. 输出形式

给定过去或当前上下文：

```text
X_{t-L+1:t}
```

模型预测未来 H 个 patch 的标签分布：

```text
p(y_{t+1}), p(y_{t+2}), ..., p(y_{t+H})
```

每个位置都有一个分类概率分布：

```text
p(y_{t+k} = walking)
p(y_{t+k} = sitting)
p(y_{t+k} = standing)
...
```

取 argmax 得到未来标签序列：

```text
ŷ_{t+1:t+H}
```

## 5. 如何得到切换时机

设当前动作标签为：

```text
y_now
```

预测未来标签序列为：

```text
[sit, sit, sit, stand, stand, walk]
```

第一次出现：

```text
ŷ_{t+k} != y_now
```

的位置，就是预测的首次切换 patch：

```text
time_to_switch = k * patch_duration
```

如果未来 H 个 patch 内都没有变化，则认为当前动作在预测窗口内延续。

## 6. 如何得到下一动作

下一动作就是第一次发生标签变化后的类别：

```text
next_action = ŷ_{t+k}
```

例如：

```text
当前: sit
未来: sit, sit, stand, stand
=> next_action = stand
```

## 7. 如何得到延续概率和切换概率

分类头在每个未来 patch 输出标签概率。因此可以定义：

```text
continue_prob(k) = p(y_{t+k} = y_now)
```

表示未来第 k 个 patch 仍保持当前动作的概率。

未来 k 个 patch 内至少发生一次切换的概率可以近似为：

```text
switch_prob_within(k) = 1 - Π_{i=1}^k p(y_{t+i} = y_now)
```

第 k 个 patch 首次切换到某动作 `a` 的概率可近似为：

```text
first_switch_to_a(k)
= Π_{i=1}^{k-1} p(y_{t+i} = y_now) * p(y_{t+k} = a), a != y_now
```

这使得“切换概率”和“下一动作概率”都来自同一组未来标签分布，而不需要额外制造大量 transition label。

## 8. 训练目标

如果有未来 patch 级标签，则直接使用交叉熵：

```text
L_future = Σ_{k=1}^H CE(p(y_{t+k}), y_{t+k})
```

如果同时预测当前 patch 标签：

```text
L_current = CE(p(y_t), y_t)
```

总损失：

```text
L = L_current + λ * L_future
```

方向一双标签版本：

```text
L = L_human_current
  + λ1 * L_human_future
  + λ2 * L_phone_current
  + λ3 * L_phone_future
```

## 9. 与自监督预训练的衔接

由于标签数据稀缺，可以先用大量无标签 IMU 做预训练：

```text
past/masked IMU patches -> reconstruct or predict IMU patches
```

可选预训练目标：

- masked patch reconstruction
- future IMU regression
- contrastive future representation prediction
- VQ token prediction

预训练后保留 encoder，再用较少标签微调 future label decoder：

```text
pretrained IMU encoder + supervised future label decoder
```

## 10. 当前最小落地版本

第一版不要引入过多任务：

```text
输入：过去 L 个 IMU patches
输出：未来 H 个人体动作 labels
模型：patch embedding + TCN/Transformer encoder + linear classification head
训练：future patch label cross entropy
评估：
  - future label accuracy / macro-F1
  - time-to-switch error
  - next-action accuracy
  - continuation probability calibration
```

如果该方案在连续 HAR 数据上可行，再扩展到：

```text
human_action_sequence + phone_motion_sequence
```

也就是方向一真正关心的“人体动作 + 手机/相机运动”联合预测。

