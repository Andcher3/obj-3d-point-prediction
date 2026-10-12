# PointOdyssey 双卡低分辨率 F0--F2 诊断

日期：2026-09-26

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Verification Status: PARTIAL（同输入诊断完成；非正式接口裁决）

## F0 执行

- 目标：`r4_new_f`、`scene_d78_0318_3rd`。
- 协议：22 帧、15 fps、7 帧观测加 15 帧未来、48 query。
- 双 V100 分别绑定一个目标；DA3 使用 `process_res=392`，生成的输入尺寸为 `224×392`。
- TrackCraft3R query-only 使用相同 DA3 输入和缩放后的 query，得到两个 `(22,48,3)` 有限轨迹。
- MotionForesight 只读取前 7 帧，两个目标均得到有限的 `P0=(15,48,3)`、`H=(15,48,16)`、`H_last_DiT=(15,48,1536)`。

## 分辨率敏感性

与已有 7 帧高分辨率接口缓存相比，低分辨率未来输出的平均绝对差异为：

| 目标 | P0 | H latent | last-DiT H |
|---|---:|---:|---:|
| `r4_new_f` | 0.1393 | 0.5612 | 0.7886 |
| `scene_d78_0318_3rd` | 0.1212 | 0.5486 | 0.8509 |

因此 `process_res=392` 会改变具体数值表示，不能声称与高分辨率完全等价。它只作为资源受限下的同输入诊断配置。

## F1 结构漂移诊断

两个目标都通过描述性漂移信号：

| 目标 | 去质心误差晚期中位数 | 边相对误差晚期中位数 | 形状误差晚期中位数 | 漂移判据 |
|---|---:|---:|---:|---|
| `r4_new_f` | 5.90% | 52.25% | 16.02% | PASS |
| `scene_d78_0318_3rd` | 12.38% | 31.28% | 11.73% | PASS |

两组均使用同一低分辨率 DA3 输入生成 MotionForesight 预测和 TrackCraft pseudo-GT；结果支持继续做学习性诊断，不能作为外部真实精度结论。

## F2 双视频 LOO 诊断

训练/验证/测试按时间槽划分，视频级留出，5 个种子。所有方法都使用同一低分辨率输入和同一 pseudo-GT。

| 方法 | raw ADE 中位数 | 去质心误差中位数 | 边相对误差中位数 | 边向量误差中位数 |
|---|---:|---:|---:|---:|
| baseline | 0.0510 | 0.0554 | 0.4270 | 0.0517 |
| geometry edge | 0.0902 | 0.1088 | 0.6397 | 0.0822 |
| node centered | 0.0838 | 0.0919 | 0.9841 | 0.0895 |
| H edge | 0.1111 | 0.1236 | 0.9912 | 0.0991 |

这轮低分辨率同输入诊断中，三个关系修正方案均未超过 baseline；F2 结果不能证明正式接口失败，因为 PointOdyssey 整物体 GT→RGB-only 几何登记仍未通过正式质量闸门，且样本只有两个。

## 决策边界

- `process_res=392` 可继续用于资源受限的接口诊断，不能替代正式分辨率主试验。
- 低分辨率没有改变当前诊断方向：baseline 优于关系修正。
- 真实视频主方案仍不能据此裁决；正式 B/F/N/U/V 训练仍未启动。

## 输出

- 低分辨率 F0：`experiences_results/pointodyssey-interface-preflight-2026-09-26/full22_res392/`
- F2 结果：`experiences_results/pointodyssey-interface-preflight-2026-09-26/full22_res392/f2_pointodyssey_two_video.json`
