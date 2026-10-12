# SensorLM 论文汇报幻灯片 v3 生成记录

日期：2026-06-24

本版本针对反馈重点重做第 01、04、05 页，第 02、03 页沿用 v2 版本。

输出目录：

- `reports/weekly/native_sensorlm_slides_v3/`

## 修改重点

- 第 01 页：删去偏装饰性的矢量图标，改用传感器矩阵、分钟级曲线、caption 示例和任务流来展示研究动机。
- 第 04 页：修正模型结构图，明确 Sensor Encoder 与 Text Encoder 的输出同时服务于对比对齐目标和条件生成目标。
- 第 05 页：补充实验表格的读法，解释 zero-shot、few-shot、AUROC、ablation 等任务和指标含义。

## 第 01 页提示词要点

白色背景，深蓝学术汇报风格，标题为“研究动机：从可穿戴数值信号到语言语义”。不要使用装饰性矢量图标。页面分三栏：

1. 左栏展示真实 wearable 数据形态：`26 features × 1440 minutes` 热力图、心率曲线、steps/acceleration 曲线，并标注 Fitbit / Pixel Watch、59.7M+ hours、103K+ users、one person-day = one sample。
2. 中栏解释为什么不能直接交给通用 LLM：连续数值序列缺少自然语言边界、一天级 1440-minute 表格过长、sensor-text pair 稀缺、需要先训练 sensor encoder。
3. 右栏展示 SensorLM 建立的语义接口：自动生成 caption 示例、支持 zero-shot activity recognition、few-shot adaptation、sensor-text retrieval、sensor caption generation。

底部用一条横向流程表示：sensor matrix `X_i` -> caption text `T_i` -> sensor-language representation -> recognition / retrieval / generation。

## 第 04 页提示词要点

白色背景，深蓝主色，橙色表示文本分支，标题为“模型结构与数据流：encoder 输出同时服务两类目标”。重点避免把 sensor encoder 只画到对比流、text encoder 只画到生成流。

左侧输入：

- 输入 1：传感器序列 `X_i: 26 × 1440 sensor matrix`
- 输入 2：文本描述 `T_i: caption tokens`

中间结构：

- `X_i` 进入 Sensor Encoder，标注 ViT-2D patches: `2 features × 10 min`
- `T_i` 进入 Text Encoder，标注 Transformer

上半部分为对比流 `contrastive alignment`：

- pooled sensor embedding `s_i` 和 pooled text embedding `v_i`
- 分别进入 projection head
- 得到 normalized embeddings
- 计算 similarity matrix `S`
- `L_con`: matched pairs closer; unmatched pairs farther

下半部分为生成流 `conditional captioning`：

- Sensor Encoder 输出的 `sensor tokens` 以蓝色粗箭头进入 Multimodal Text Decoder
- Text Encoder 输出的 `text representations / previous caption tokens` 以橙色粗箭头进入 Multimodal Text Decoder
- Decoder 标注 causal self-attention + cross-attention
- 输出 caption tokens，并标注 `L_cap = -log P(y_t | y_<t, x)`

右侧表格解释：

- CLIP-like: `L_con only`, supports zero-shot / retrieval
- Cap-like: `L_cap only`, supports caption generation
- CoCa-like: `L_con + L_cap`, supports alignment + generation

底部给出总目标：

`L_SensorLM = lambda_con L_con + lambda_cap L_cap`

## 第 05 页提示词要点

白色背景，深蓝学术风格，标题为“实验结果怎么读：任务、指标与主要结论”。页面左侧固定为“读表方式”，右侧为 2 × 2 实验结果区。

左侧解释：

- Zero-shot：不用该任务训练标签，直接用 sensor-text 相似度选择活动类别；适用于完全没有标签的新数据。
- Few-shot：每类只给 5/10/20/50 个标签，冻结 encoder 后训练小分类器；适用于少量标注的新场景。
- AUROC：排序能力，0.50 约等于随机，越接近 1 越好。
- Ablation：删除或替换某部分，看性能变化判断其贡献。

右侧四块结果：

1. Zero-shot activity recognition：20 个活动类别，Gemma 0.50、Gemini 0.51、SensorLM 0.84，指标 AUROC。
2. Few-shot activity recognition：labels/class 为 5、10、20、50；SimCLR AUROC 为 0.54、0.55、0.58、0.60；SensorLM AUROC 为 0.73、0.79、0.83、0.88。
3. caption 类型消融：statistical 0.51、structural 0.50、semantic 0.71、structural + semantic 0.84。
4. 架构消融：CLIP-like zero-shot AUROC 0.83、linear F1 0.53；Cap-like 0.55、0.32；CoCa-like 0.84、0.57。

底部强调：优势来自弱语义 caption 与 sensor-text contrastive alignment；CoCa 在保留识别能力的同时增加生成能力。
