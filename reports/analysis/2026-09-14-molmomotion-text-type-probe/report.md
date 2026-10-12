# MolmoMotion 文本类型探测实验

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-09-14
- Verification Status: VERIFIED
- Version Label: molmomotion_text_probe_v1
- Experiment ID: MM-TEXT-20260914
- Repository: `allenai/molmo-motion`
- Repository commit: `61f5b21b694ad8f854ec7ecd2400005acc73f685`
- Checkpoint: `allenai/MolmoMotion-4B-H3-F30/model.pt`
- Checkpoint SHA256: `506beccd01e9edd4d3ebf0bf88fec00b83530eec525571168187c7c4888ee205`
- Checkpoint bytes: `17847894425`
- Runtime: Python 3.11.16, PyTorch 2.9.1+cu128, RTX 5090 32 GB

## 实验问题

检查 MolmoMotion 在保持视频、二维查询点、三维点历史、模型权重和 30 帧预测范围不变时，能否接收并利用以下动作文本：

1. 官方描述；
2. 简短动名词组合；
3. 自由自然语言句子；
4. 较长的多步骤动作计划；
5. 与视频趋势冲突的动作描述。

这里的“支持”分为两层：接口能否产出有效轨迹，以及轨迹是否随文本含义产生相符变化。

## 实验设计

- 完整输入采用官方仓库自带的三组样本：`egodex_clean_surface`、`egodex_ball_base`、`davis_car_turn`。
- 每组只替换 `action` 字符串，其他输入保持固定。
- 每种文本执行一次确定性解码；官方文本另行复跑一次。
- 每次输出 8 个查询点、30 帧、三维坐标。
- 比较量为候选文本输出相对官方文本输出的 3D ADE、FDE、质心运动方向余弦和路径长度。这些量衡量文本敏感性，不是相对真实未来的准确率。
- 官方生成代码的 `sampler=None` 使用 `DeterministicSampler`。官方文本复跑得到逐元素完全一致的轨迹，最大绝对差为 `0.0`。

## 执行结果

- 15/15 次推理成功，输出形状均为 `(8, 30, 3)`，有限值比例均为 `1.0`。
- 单次推理耗时约 60.8–84.6 秒，峰值 CUDA allocated memory 约 9.60–9.77 GiB。
- 完整批次退出码为 `0`，总耗时 `1098` 秒。
- 官方文本复跑退出码为 `0`，输出与第一次逐元素相同。

| 样本 | 文本类型 | ADE to official (m) | FDE to official (m) | 方向余弦 | 路径长度 (m) |
|---|---:|---:|---:|---:|---:|
| egodex_clean_surface | compact | 0.010 | 0.016 | 0.999 | 0.910 |
| egodex_clean_surface | free_sentence | 0.021 | 0.029 | 0.999 | 0.920 |
| egodex_clean_surface | multi_step | 0.009 | 0.019 | 1.000 | 0.944 |
| egodex_clean_surface | contradictory | 0.019 | 0.018 | 0.998 | 0.899 |
| egodex_ball_base | compact | 0.079 | 0.114 | -0.002 | 0.107 |
| egodex_ball_base | free_sentence | 0.033 | 0.059 | 0.706 | 0.287 |
| egodex_ball_base | multi_step | 0.063 | 0.050 | 0.956 | 0.416 |
| egodex_ball_base | contradictory | 0.075 | 0.063 | 0.409 | 0.439 |
| davis_car_turn | compact | 0.189 | 0.513 | 0.989 | 3.078 |
| davis_car_turn | free_sentence | 0.310 | 1.085 | 0.970 | 2.405 |
| davis_car_turn | multi_step | 0.121 | 0.382 | 0.998 | 2.984 |
| davis_car_turn | contradictory | 0.192 | 0.511 | 0.997 | 3.780 |

完整数值见 `D:\Pycharm\post_graduate_project\archive\experiments\molmomotion\MolmoMotion-text-types\remote\results\text-type-probe\metrics.json` 和同目录的 `metrics.csv`。

## 观察与判断

### 接口层

自由句子和 100–119 字符的四步计划都能被处理，并稳定产出合法形状。MolmoMotion 的 `action` 接口没有把输入限制为固定 verb–noun 标签。

### 文本条件作用

- `egodex_clean_surface` 的五条轨迹几乎重合。包括“抬起并悬停”的冲突文本在内，方向余弦仍为 `0.998`，说明视觉历史在该样本中占主导。
- `davis_car_turn` 的“立刻停车并向左倒车”仍沿官方右转分支运动，方向余弦为 `0.997`。模型未可靠执行这一相反动作条件。
- `egodex_ball_base` 对文本更敏感。过度简化为 `placing a ball` 后方向余弦降至 `-0.002`，保留对象与目标关系的自由句子为 `0.706`，四步描述为 `0.956`。对象、关系和目标信息会改变输出。

接口层支持自由文本和多步骤文本；语义执行层的支持程度不稳定。多步骤文本通常仍落在与官方描述相近的主运动分支，当前结果没有证明模型能按顺序落实完整计划。30 帧约覆盖 2 秒，计划中后续步骤也可能超出预测范围。

### 对项目的含义

MolmoMotion 可以作为模式条件轨迹解码器的候选，条件文本宜保留对象、关系和目标，并接近其训练描述形式。当前权重不能直接承担“任意未来 narration 都能可靠变成对应三维轨迹”的假设。后续若第一阶段给出稀有或与视频趋势冲突的模式，需要单独验证条件遵循能力或进行适配。

## Ego4D 小样本

已通过官方 CLI 下载 3 个 FHO 短片，总计约 4.42 MiB：

| UID | 时长 | 字节 | SHA256 |
|---|---:|---:|---|
| `cb78dba8-6647-4ec7-875c-6ba1548f8312` | 6.72 s | 965896 | `1db22e17936bbaa91424ed06731bd806f2af0b5dc49e721ed3c0b5f40db31fae` |
| `c0e127bc-ff39-4a34-b4c5-0fd8cfbfa732` | 8.20 s | 980271 | `a0014dd3d72d2e5adfc87be93a34886ecd39899861f4a75267164ab3ef4526e9` |
| `356ed369-7023-4660-904b-be73823a21d3` | 9.00 s | 2691240 | `0b394526d07358aaa42ffcd445ae1c861d6640aae8214ac5733973c0c25b875f` |

本轮没有把原始 Ego4D 视频直接输入 MolmoMotion。有效输入还需要目标查询像素及三维点历史；缺少这些条件时，文本类型与前处理误差会混在一起。实际文本对照采用官方完整样本。

## 异常与边界

- 初次安装 `ego4d` 时 AutoDL 镜像源缺包；切换官方 PyPI 后安装成功。
- 指定 `--version v2` 会触发 403；CLI 默认版本成功解析为 `v2_1` 并完成下载。
- 本实验没有多条真实未来轨迹，不能判断哪条文本输出更准确，也不能把 ADE/FDE to official 解读为语义正确率。
- 三组样本不足以估计跨数据集泛化。

## Fallacy Scan

- Coverage: 11/11 checked

| 项目 | 结果 |
|---|---|
| Simpson's Paradox | 已按样本分别报告，没有用总体平均替代样本差异。 |
| Ecological Fallacy | 不适用；未从组级结果推断个体规律。 |
| Berkson's Paradox | 不适用；没有相关性估计。 |
| Collider Bias | 不适用；没有控制变量回归。 |
| Base Rate Neglect | 不适用；没有诊断概率。 |
| Regression to the Mean | 不适用；没有极端值筛选后的前后测。 |
| Survivorship Bias | 15/15 运行均报告；失败的安装与版本尝试也已记录。 |
| Look-Elsewhere Effect | 所有预设的 3×5 条结果均报告，没有挑选单一有利样本。 |
| Garden of Forking Paths | NOTE：探索性试验，文本模板和样本数尚未预注册。 |
| Correlation != Causation | 未作因果声明。 |
| Reverse Causality | 不适用。 |

## Reproducibility

- Method: deterministic re-run
- Verdict: REPRODUCIBLE for the repeated official condition
- First shape: `(8, 30, 3)`
- Re-run shape: `(8, 30, 3)`
- Exact match: `True`
- Maximum absolute difference: `0.0`

## 主要文件

- 修改脚本：`D:\Pycharm\post_graduate_project\archive\experiments\molmomotion\MolmoMotion-text-types\run_text_type_probe.py`
- 远程镜像：`D:\Pycharm\post_graduate_project\archive\experiments\molmomotion\MolmoMotion-text-types\remote`
- Ego4D 样本：`D:\Pycharm\post_graduate_project\archive\experiments\molmomotion\MolmoMotion-text-types\ego4d-samples`
- 四项验证：`D:\Pycharm\post_graduate_project\archive\experiments\molmomotion\MolmoMotion-text-types\remote\VERIFICATION.txt`
- 官方仓库：[allenai/molmo-motion](https://github.com/allenai/molmo-motion)
- 官方 Quick Start：[examples/01_quickstart.py](https://github.com/allenai/molmo-motion/blob/main/examples/01_quickstart.py)
- Ego4D CLI：[官方文档](https://ego4d-data.org/docs/CLI/)

