

这反而最接近 MotionForesight。

前提是原语义模型内部真的存在：
$z_1,z_2,\ldots,z_T$

这种具有明确时间位置的 representation。

例如：

```
video frame tokens
↓
temporal transformer
↓
z1 z2 z3 z4 z5 z6
↓
language decoder
↓
global narration
```

虽然最终 narration $Y$ 没有时间对齐，但内部$z_t$可能仍然与时间相关。

那么可以像 MotionForesight 一样：

```
完整视频 teacher：

z1 z2 z3 | z4 z5 z6
           ↑ future target
```

student：

```
真实 真实 真实 | MASK MASK MASK
       ↓
forecasting adapter
       ↓
ẑ4 ẑ5 ẑ6
```

监督：
$\hat z_{4:6}\approx z_{4:6}$

最后再让 language decoder：

$\hat z_{4:6} \rightarrow \text{future narration}$

这才真正恢复了 MotionForesight 的$\boxed{\text{time-aligned latent slots}}$

优势。

但如果原模型把整个视频直接压成几个**无时间含义的 global semantic tokens**，那么这种办法也用不了。