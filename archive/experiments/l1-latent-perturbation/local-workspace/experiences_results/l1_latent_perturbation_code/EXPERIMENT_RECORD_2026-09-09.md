# Future-L1 latent 扰动短视频实验记录

## 1. 实验目的

本实验用于验证 Future-L1-SFT 在视频事件预测任务中的 latent proposal 扰动是否会产生不同的未来事件回答分支。实验只验证推理阶段的 latent 敏感性和输出分支，不等同于完整的三维轨迹预测评测。

与首轮实验相比，本轮：

- 不再使用原先约 178 秒、且不符合轨迹预测要求的视频；
- 改用较短的公开视频观察前缀，避免把后续事件直接输入模型；
- 将问题固定为 `What will happen next in the video?`；
- 比较多个 latent 扰动强度和两个随机种子；
- 保留 latent、原始生成文本和结构化统计，便于复核。

## 2. 实验环境

| 项目 | 实际设置 |
|---|---|
| 模型 | `Eurayka/Future-L1-SFT` |
| 本地模型目录 | `/root/autodl-tmp/models/future-l1` |
| Python | 3.10 环境：`/root/autodl-tmp/conda_envs/future-l1-latent` |
| PyTorch | `2.9.1+cu128` |
| Transformers | `4.57.1` |
| GPU | NVIDIA GeForce RTX 5090 |
| CUDA | 12.8 |
| 推理精度 | `bfloat16` |
| device map | `cuda:0` |
| attention | `sdpa` |
| 解码 | `do_sample=False`，`use_cache=True` |

## 3. 输入视频

视频来自可直接访问的 W3C/MDN 公共示例源。它们不是 Ego4D 或 SSV2 原始样本，而是用于验证流程的公开许可相似样本。每个输入文件都是从原视频开头截取的观察前缀；截取后段没有作为模型输入。

| 名称 | 输入文件 | 观察时长 | SHA-256 |
|---|---|---:|---|
| Sintel | `/root/autodl-tmp/datasets/future-l1-latent-test/public-action-samples/sintel_observation.mp4` | 25 秒 | `2bbdd39008edbbb01184516d59eda19bf92ffa475b1babed6dd02f688831eb04` |
| Bunny | `/root/autodl-tmp/datasets/future-l1-latent-test/public-action-samples/bunny_observation.mp4` | 15 秒 | `76ab9486d8cc7eccad469f565bd324361c94beef2de0055307726750fdba172a` |
| Flower | `/root/autodl-tmp/datasets/future-l1-latent-test/public-action-samples/flower_observation.mp4` | 2 秒 | `208152c5fab6cd96f5b053d0667c1143c032523cd7f8a6020bdd3544d640f587` |

原始可下载地址：

- `https://media.w3.org/2010/05/sintel/trailer.mp4`
- `https://media.w3.org/2010/05/bunny/trailer.mp4`
- `https://interactive-examples.mdn.mozilla.net/media/cc0-videos/flower.mp4`

Flower 只有很短的单一变化，因此主要作为“latent 改变但答案保持稳定”的负对照。Sintel 和 Bunny 也不是严格的第一视角手-物体轨迹数据，结论应解释为方法级验证，而不是数据集级性能结论。

## 4. 输入消息和模型格式

用户问题固定为：

```text
What will happen next in the video?
```

系统消息要求模型使用 textual mode 和 visual latent mode，并把最终答案放在 `<answer>...</answer>` 中：

```text
You are a multimodal reasoning assistant capable of thinking in textual and visual modes.

Use the following tags to switch your thinking mode:

1. Textual Mode: <reason>Your textual reasoning process</reason>
2. Visual Mode: <|latent_start|>Your visual reasoning process<|latent_end|>

After all thinking is complete, place the final answer inside <answer>Your Final Answer</answer>.
```

视频通过 `qwen_vl_utils.process_vision_info` 读取，使用 `nframes=8` 稀疏采样。视觉最大像素数为 `401408`，最小像素数为 `256 * 28 * 28`。

## 5. 扰动设计

### 5.1 扰动位置

扰动只施加在 Future-L1 自回归生成过程中处于 active 状态的 `latent_hidden_state` 向量：

1. 模型先按正常路径生成文本 token 和 latent span；
2. 根据 `active_mask` 选出当前 active latent vectors；
3. 为这些 latent vectors 加入独立高斯噪声；
4. 使用扰动后的 latent state 继续生成。

因此，本实验没有修改：

- 输入视频帧；
- 视频采样位置；
- 问题文本；
- 普通文本 token 的采样随机性；
- 模型权重。

### 5.2 噪声公式

默认 `scale_mode=rms`。对每个 active latent vector `h`，计算：

```text
scale(h) = sqrt(mean(h^2))
noise    = Normal(0, 1) * (sigma * scale(h))
h'       = h + noise
```

当 `sigma=0` 时，代码执行精确 no-op。`latent_noise_seed` 使用独立的 `torch.Generator`，不会改变普通 token 生成所使用的随机状态。

代码位置：

- `_make_latent_noise_generator`：创建专用随机数生成器；
- `_apply_relative_latent_noise`：计算缩放、加噪和噪声统计；
- `_FutureL1GenerationMixin`：允许额外 latent 参数通过 Transformers generation 校验；
- `_future_l1_sample`：在生成循环中读取并应用 latent 扰动。

对应文件：

`src/model/future_l1.py`

## 6. 参数设置

本轮历史结果中每个视频运行 11 次：

- baseline：1 次，`sigma=0`，`seed=0`；
- 扰动强度：`sigma ∈ {0.005, 0.01, 0.03, 0.1, 0.2}`；
- 每个 sigma 使用 `seed ∈ {11, 22}`；
- `max_new_tokens=256`；
- `nframes=8`；
- `scale_mode=rms`；
- `force_initial_latent=false`；
- `baseline_repeats=1`。

这里的 `baseline_repeats=1` 是历史运行参数，不能支持重复一致性结论。后续运行入口已强制要求 `baseline_repeats >= 2`；新实验应至少使用两个完全相同的 baseline。

实验入口：

`experiments/l1_latent_perturbation/run_l1_latent_perturbation.py`

以 Sintel 为例的实际运行命令：

```bash
cd /root/autodl-tmp/projects/future-l1/Future-L1-latent-test
/root/autodl-tmp/conda_envs/future-l1-latent/bin/python \
  experiments/l1_latent_perturbation/run_l1_latent_perturbation.py \
  --model /root/autodl-tmp/models/future-l1 \
  --video /root/autodl-tmp/datasets/future-l1-latent-test/public-action-samples/sintel_observation.mp4 \
  --question 'What will happen next in the video?' \
  --output-dir /root/autodl-tmp/results/future-l1-latent/public-short/sintel \
  --sigmas '0.005,0.01,0.03,0.1,0.2' \
  --seeds '11,22' \
  --baseline-repeats 1 \
  --max-new-tokens 256 \
  --nframes 8 \
  --scale-mode rms
```

Bunny 和 Flower 只需替换 `--video` 与 `--output-dir`。

## 7. 保存内容和统计方法

每个输出目录包含：

- `metadata.json`：模型、视频哈希、问题、硬件和运行参数；
- `summary.json`：完整结果和 aggregate 统计；
- `NN_<run>_latents.npz`：提取出的 latent vectors，保存为 float16；
- `NN_<run>_raw.txt`：包含特殊 token 的原始生成文本；
- `NN_<run>.json`：单次运行的答案、哈希、latent 统计和运行时间；
- `results.partial.json`：运行过程中可恢复的部分结果。

相对 baseline 的比较包括：

- latent `mse`；
- latent `mean_cosine`；
- latent `relative_l2`；
- 原始文本是否完全变化；
- `<answer>` 内容是否完全变化；
- answer 文本的 `SequenceMatcher` 相似度。

格式有效要求同时满足：

1. 存在 `<answer>...</answer>`；
2. 存在 latent start/end tags；
3. latent 数值全部有限。

## 8. 实验结果

结果目录：

- `/root/autodl-tmp/results/future-l1-latent/public-short/sintel/`
- `/root/autodl-tmp/results/future-l1-latent/public-short/bunny/`
- `/root/autodl-tmp/results/future-l1-latent/public-short/flower/`

| 视频 | 总运行数 | 格式有效 | 非零 latent 改变 | 非零原文改变 | 非零答案改变 | 非零答案种类 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| Sintel | 11 | 11 | 10 | 3 | 3 | 3 | 出现少量回答分支 |
| Bunny | 11 | 10 | 10 | 4 | 4 | 5 | 出现较明显回答分支，但有 1 次格式无效 |
| Flower | 11 | 11 | 10 | 0 | 0 | 1 | 输出对扰动稳定 |

三个视频均满足 `all_runs_finite=true`。由于历史运行只执行了一个 baseline，旧结果中的 `zero_noise_deterministic=true` 仅是空集合/单次比较意义上的字段，不能证明重复运行一致性。新入口已禁止单次 baseline。latent 扰动本身没有产生非有限数值。

## 9. 解释与限制

本轮实验支持以下结论：

- latent 扰动确实作用到了 active latent vectors；
- 在 Sintel 和 Bunny 上，部分扰动会改变原始生成文本和最终答案；
- 输出分支并非仅由普通 token sampling 随机性造成，因为实验使用 `do_sample=False`；
- Flower 的结果表明，latent 数值发生变化不必然导致可见答案变化。

本轮实验不能证明：

- 生成的不同答案对应真实、互斥且正确的未来轨迹；
- 方法已经在 Ego4D 或 SSV2 上得到验证；
- sigma 与预测准确率之间存在单调关系；
- 不同答案一定代表模型内部不同的视觉未来状态，而不是语言表述变化。

后续若要推进到真正的轨迹预测验证，应优先使用具有明确手-物体连续动作和可标注未来事件的数据，并为每个观察前缀保留真实后续片段和人工/数据集动作标签。

## 10. 完整方案代码边界（待 GPU 执行）

本轮新增的 `complete_scheme.py` 覆盖了此前缺失的实验编排层：

1. **SemDiD 式事件比较**：从候选答案提取内容词，计算成对事件相似度/距离，并保留可替换的语义评分回调；
2. **联合分支保留**：以候选质量和与已保留分支的最小事件距离联合打分，避免只保留同一事件的措辞变体；
3. **反馈搜索**：把当前保留分支和目标物体写入下一轮 prompt，通过 `generate(prompt, round_index)` 回调生成新候选；
4. **第二阶段轨迹解码接口**：通过 `TrajectoryDecoder.decode(candidate, video_path, target_object)` 对保留 latent 分支解码。没有真实轨迹解码器时会保持未解码状态，不会把文字答案伪装成轨迹。

三个待测视频及目标物体初始标注见 `short_video_manifest.json`。目标物体是实验配置而非自动识别结果，正式运行前应在完整视频上复核。

历史结果只做独立随机扰动基线；它没有执行上述四步，因此不能支持“已经获得合格多模式教师候选”的结论。

### GPU 开启后的入口

新增 `run_complete_search.py`，它会：

- 对同一输入执行两个 baseline（`sigma=0`）和多组 sigma/seed；
- 对初始候选执行事件比较和联合分支保留；
- 用保留结果构造下一轮反馈 prompt；
- 重复反馈搜索指定轮数；
- 保存每个候选的 raw 文本、latent `.npz` 和 `complete_search_report.json`。

示例：

```bash
/root/autodl-tmp/conda_envs/future-l1-latent/bin/python \
  experiments/l1_latent_perturbation/run_complete_search.py \
  --model /root/autodl-tmp/models/future-l1 \
  --video /root/autodl-tmp/datasets/cut_0.mp4 \
  --target-object "the pink rectangular cutting board" \
  --output-dir /root/autodl-tmp/results/future-l1-latent/complete-search/cut_0 \
  --branch-count 3 \
  --feedback-rounds 1
```

当前脚本会明确在报告中写出 `trajectory_decoder=null`，因为仓库尚未提供把 Future-L1 latent 映射为真实二维/三维轨迹的已训练 decoder。这个缺口不能用生成文字替代；接入满足 `TrajectoryDecoder.decode(...)` 接口的 decoder 后，`run_feedback_search` 会对每个保留分支执行第二阶段解码并保存结果。

## 11. 新视频完整搜索结果（2026-09-10）

GPU 已恢复后，对 `/root/autodl-tmp/datasets/cut_0.mp4`、`cut_1.mp4`、`cut_2.mp4` 各执行了完整搜索。每个视频包含：

- 2 个完全相同的 baseline（`sigma=0, seed=0`）；
- 10 个初始扰动候选（5 个 sigma × 2 个 seed）；
- 10 个反馈轮候选；
- 1 轮反馈搜索；
- 联合保留 3 个分支。

| 视频 | 初始/反馈候选总数 | baseline raw 一致 | baseline latent 一致 | 最终保留分支 | 轨迹解码 |
|---|---:|---|---|---:|---|
| `cut_0` | 22 | 是 | 是 | 3 | 未接入 |
| `cut_1` | 22 | 是 | 是 | 3 | 未接入 |
| `cut_2` | 22 | 是 | 是 | 3 | 未接入 |

结果目录：

- `/root/autodl-tmp/results/future-l1-latent/complete-search/cut_0/`
- `/root/autodl-tmp/results/future-l1-latent/complete-search/cut_1/`
- `/root/autodl-tmp/results/future-l1-latent/complete-search/cut_2/`

其中 `complete_search_report.json` 保存 pairwise event comparisons、反馈 prompt、最终保留分支和 `trajectory_decoder=null` 的明确警告。

### 结果观察

- `cut_0`（粉色矩形砧板）产生了“拿起并移动”“放到台面”“移动到其他位置”等不同文字事件，说明目标物体约束下出现了可区分的候选事件。
- `cut_1`（橙色碗）所有保留候选都描述继续清洁碗，扰动没有形成有意义的事件分离。
- `cut_2`（砧板上的食物和刀）所有保留候选都描述开始切食物，扰动没有形成有意义的事件分离。

因此，本轮证明了完整的“候选生成—事件比较—联合保留—反馈搜索”代码路径可以运行，并纠正了 baseline 重复性问题；但由于尚无第二阶段轨迹 decoder，仍不能把这些候选称为合格的视觉未来轨迹或多模式教师候选。

## 10. Ego4D 官方 CLI 下载尝试（2026-09-10）

为取得更符合第一视角动作预测的样本，按 Ego4D 官方 CLI 的 UID 筛选方式进行了最小下载尝试。

### 已完成

- 在实验环境中安装 `ego4d==1.7.3`；
- 准备 UID 文件：
  `/root/autodl-tmp/datasets/future-l1-latent-test/ego4d-sample/video_ids.txt`；
- UID 为：

  ```text
  000786a7-3f9d-4fe6-bfb3-045b368f7d44
  000a3525-6c98-4650-aaab-be7d2c7b9402
  000cd456-ff8d-499b-b0c1-4acead128a8b
  ```

- 使用的下载命令：

  ```bash
  /root/autodl-tmp/conda_envs/future-l1-latent/bin/ego4d \
    --output_directory=/root/autodl-tmp/datasets/future-l1-latent-test/ego4d-sample \
    --datasets video_540ss \
    --video_uid_file=/root/autodl-tmp/datasets/future-l1-latent-test/ego4d-sample/video_ids.txt \
    --no-metadata \
    --yes
  ```

### 当前阻塞

下载尚未开始。CLI 在连接 Ego4D S3 之前检查 AWS profile 时失败：

```text
RuntimeError: Could not find AWS profile 'default'.
```

当前服务器不存在 `/root/.aws/credentials`，也没有 `AWS_*` 环境凭证。Ego4D 官方数据需要用户接受许可并配置有效的 AWS credentials；不能通过公开 URL 或匿名方式绕过该授权。

获得授权后，可将 credentials 配置为 `default` profile，或把命令增加 `--aws_profile_name <profile>` 后重新执行。当前 `ego4d-sample` 目录只包含 UID 文件，没有下载到视频或数据集文件。
