## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-15
- Verification Status: VERIFIED
- Version Label: validation_v1

# MolmoMotion：Tracker3D 真实输入与文本控制复核

## 结论

缺失的查询像素与三维历史已经补齐，`TrackCraft3R -> MolmoMotion` 端到端链路可以运行；本次样本没有表现出可用的文本条件控制。四条含义不同的动作文本进入模型后形成了不同的 token 张量，但模型生成的原始 `<tracks>` 文本及解析后的三维轨迹完全相同，且 30 个预测时刻内部没有运动。因此，这次实验的技术链路通过，研究目标未通过。

当前证据不支持把未经适配的 MolmoMotion 直接用作本项目第二阶段的可靠自由文本条件轨迹模型。该判断只覆盖本次真实输入协议；它没有证明 MolmoMotion 在所有分布内输入上都忽略文本。

## 实验对象与边界

| 项目 | 实际使用内容 |
|---|---|
| 视频 | Ego4D FHO sewing clip，201 帧、30 fps、1440×1080；SHA256 `1db22e...31fae` |
| 目标物体 | 桌面上的蓝柄剪刀 |
| 查询点 | TrackCraft3R 832×480 第 0 帧中人工选择 8 点 |
| 观测输入 | DA3 预处理后的 12 帧；Tracker 使用的原视频帧索引为 0,5,...,55 |
| MolmoMotion 历史 | Tracker 输出的最后 3 帧，即 `[9,10,11]` |
| 未来长度 | 30 个时刻 |
| Tracker | TrackCraft3R release checkpoint，SHA256 `d70084...a7ad` |
| 相机/深度 | `depth-anything/DA3-BASE`；仅用于集成测试，不视为 MotionForesight 原始 metric-depth 配置 |
| 轨迹模型 | `MolmoMotion-4B-H3-F30`，SHA256 `506bec...e205` |

未来原始帧没有进入 Tracker 或 MolmoMotion 条件。Tracker 只处理上述观察窗口；后续视频只保留在原始样本中，没有用于推理。

## 测试文本

1. 简短同向：拿起蓝柄剪刀并向右移动。
2. 自由同向：抓起剪刀，稳定地移向缝纫机右侧。
3. 多步骤：拿起、向右移动，再略微朝相机移动。
4. 冲突方向：让剪刀留在桌面并向左滑动。

## 结果

| 指标 | 结果 |
|---|---:|
| 有效查询点 | 8/8 |
| 四组输出形状 | 均为 `(8,30,3)` |
| finite fraction | 均为 1.0 |
| 四组输入 token | 数量与 SHA256 均不同 |
| 四组原始生成文本 | 字符数均为 3131，SHA256 均为 `774f009c...e53d98` |
| 任意两组轨迹 ADE/FDE | 0.0 m / 0.0 m |
| 轨迹数组 SHA256 | 四组均为 `e616e079...e2d769` |
| 预测段内部平均路径长度 | 0.0 m |
| 从最后历史点到最终点的平均路径长度 | 0.002147826 m |
| 最终质心位移 | `[+0.001008868,+0.000126421,-0.001846790]` m |
| 单条推理时间 | 61.5–62.4 s |
| 峰值 CUDA allocated | 约 9.62 GiB |

原始输出从时间戳 3.0 到 32.0 重复同一组量化坐标。静止结果由模型直接生成，并非解析器把不同生成结果压成相同轨迹。

## 可视检查

- 查询点落在剪刀刀身、连接处与手柄区域，数量和投影位置有效。
- 四条预测质心曲线在图中完全重合。
- 可视化仅用于核对输入点和输出重合，不用于证明三维轨迹的物理精度。

## 复现性

- 方法：相同模型、输入、脚本和四条文本完整重跑。
- 结果：四个轨迹数组逐元素最大差值均为 0；四个原始生成文本也逐字节一致。
- 判定：`REPRODUCIBLE`。

## 解释与限制

1. 已排除“文本没有送入模型”和“解析器造成四组输出相同”两种直接实现错误：输入 token 哈希不同，原始生成文本已经相同。
2. 当前结果仍可能受输入分布差异影响：Ego4D 剪刀样本、TrackCraft3R 估计的弱运动历史、DA3-BASE 相对尺度，与 MolmoMotion 的训练输入分布未必匹配。
3. 该样本的观察末段运动很小，模型可能强烈偏向静止先验；这恰好说明稀有或反事实文本在该设置下没有压过视觉/历史条件。
4. 只有一个真实视频目标，不能据此估计总体成功率，也不能区分“自由文本能力不足”和“本输入协议分布外”各自贡献。
5. 本次可以否定的是当前组合在该样本上达到预期；模型选择仍需结合更多样本或替代模型对照。

## 11/11 方法学谬误扫描

| 项目 | 检查结果 |
|---|---|
| Simpson's paradox | 无分组聚合比较，不适用 |
| Ecological fallacy | 没有从群体统计推断个体，不适用 |
| Berkson's paradox | 样本经过可运行性与人工目标选择；已限制结论范围 |
| Collider bias | 没有回归控制变量，不适用 |
| Base-rate neglect | 没有把单样本通过率解释为总体概率 |
| Regression to the mean | 没有极端值前后测，不适用 |
| Survivorship bias | 记录了 tokenizer 缓存导致的首次失败，未只报告成功运行 |
| Look-elsewhere effect | 四条文本预先按短句、自由句、多步骤、反向语义定义；未筛选最佳结果 |
| Garden of forking paths | 查询点经过一次视觉纠正；该人工选择及 DA3-BASE 替代均已披露 |
| Correlation != causation | 未作因果结论 |
| Reverse causality | 不适用 |

覆盖：11/11 checked。总体置信度：`CAUTION`，原因是单样本与输入分布差异；技术执行与确定性复现证据为 `SOLID`。

## 产物

- 脚本：`D:\Pycharm\post_graduate_project\archive\experiments\molmomotion\MolmoMotion-tracker3d-e2e\final\run_tracker3d_molmo_probe.py`
- 指标：`D:\Pycharm\post_graduate_project\archive\experiments\molmomotion\MolmoMotion-tracker3d-e2e\final\results\molmo-text-control\metrics.json`
- 原始生成文本：`D:\Pycharm\post_graduate_project\archive\experiments\molmomotion\MolmoMotion-tracker3d-e2e\final\results\molmo-text-control\future_text`
- 查询点图：`D:\Pycharm\post_graduate_project\archive\experiments\molmomotion\MolmoMotion-tracker3d-e2e\final\results\molmo-text-control\input_query_points.png`
- 预测图：`D:\Pycharm\post_graduate_project\archive\experiments\molmomotion\MolmoMotion-tracker3d-e2e\final\results\molmo-text-control\prediction_centroids.png`
- 完整事务记录：`D:\Pycharm\post_graduate_project\archive\experiments\molmomotion\MolmoMotion-tracker3d-e2e\final\VERIFICATION.txt`

