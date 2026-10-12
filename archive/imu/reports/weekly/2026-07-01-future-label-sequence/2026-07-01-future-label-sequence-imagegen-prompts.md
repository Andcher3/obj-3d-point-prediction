# Future Label Sequence 图像生成提示词记录

日期：2026-06-30

使用方式：内置 imagegen 工具。生成图像先保存在 `.codex/generated_images/`，随后复制到项目 `reports/posters/assets/`。

## 1. 模型结构图

目标文件：

```text
reports/posters/assets/future-label-sequence-transformer-architecture.png
```

提示词要点：

```text
Generate a 16:9 academic method diagram titled
"Parallel Query Transformer for Future IMU Label Sequence".

Show smartphone IMU patches with acc x/y/z and gyro x/y/z,
Patch Embedding, Transformer Encoder, learnable queries q0...qH,
Parallel Query Transformer Decoder with unmasked query self-attention,
cross-attention to encoder memory, feed-forward network,
Shared MLP Classifier, and label probabilities p(y_t)...p(y_{t+H}).

Add example final label sequence: sit | sit | stand | walk.
Use white background, large readable labels, navy text, teal/orange accents.
No causal mask, no masked self-attention, no fake performance numbers.
```

## 2. 训练策略图

目标文件：

```text
reports/posters/assets/future-label-sequence-training-pipeline.png
```

提示词要点：

```text
Generate a 16:9 training pipeline figure titled
"Training Strategy: Unlabeled IMU Pretraining + Labeled Forecast Fine-tuning".

Stage 1: self-supervised pretraining on large unlabeled IMU.
Raw smartphone IMU -> patchify -> mask some patches -> Transformer Encoder
-> reconstruction decoder -> reconstructed IMU patches.
Loss: masked reconstruction loss.

Stage 2: supervised future label sequence fine-tuning.
Labeled continuous IMU -> patchify -> pretrained Transformer Encoder
-> query Transformer Decoder -> MLP classifier -> future label sequence.
Loss: cross entropy over current and future labels.

Evaluation panel: future label accuracy, next-action accuracy,
time-to-switch error, continuation probability.
```

说明：生成图中 Stage 2 的 decoder 内部文字出现了 "Masked Self-Attention"；正式讲解中已通过海报/PPT注释修正为 unmasked parallel query decoder。Mask 仅用于 Stage 1 的 reconstruction。

## 3. 切换推理图

目标文件：

```text
reports/posters/assets/future-label-sequence-switch-inference.png
```

提示词要点：

```text
Generate a 16:9 explanatory figure titled
"Deriving Switch Timing from Future Label Probabilities".

Current label: SIT.
Future patches k=1...H:
SIT 0.92, SIT 0.84, SIT 0.61, STAND 0.74, STAND 0.86, WALK 0.68.

Mark the first STAND cell as first label change.
Derived outputs:
continuation probability,
switch probability within horizon,
time-to-switch = first changed patch,
next action = first changed label.

Bottom note: No separate transition class is required;
transition is inferred from the predicted sequence.
```

