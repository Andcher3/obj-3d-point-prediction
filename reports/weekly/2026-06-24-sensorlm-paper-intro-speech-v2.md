# SensorLM 纯论文介绍发言稿 v2

预计时长：约 150-180 秒

大家好，下面汇报一篇近期论文 SensorLM: Learning the Language of Wearable Sensors。

这篇论文的动机不是再做一个普通的 HAR 分类器，而是希望把 wearable sensor 信号映射到自然语言可以描述的语义空间。作者认为，真实可穿戴设备数据规模很大，比如 Fitbit 和 Pixel Watch 中有 59.7M 以上小时、10.3 万以上用户，但连续数值信号本身不是语言，通用 LLM 很难直接理解一天级的长传感器表格。因此论文的目标是训练一个 sensor-language foundation model，支持 zero-shot activity recognition、few-shot adaptation、sensor-text retrieval 和 sensor caption generation。

数据上，一个训练样本对应一个 person-day。输入是 26 个分钟级特征乘以 1440 分钟，也就是一天级 sensor matrix。特征来自 PPG、accelerometer、EDA、temperature、altimeter 等模态，例如 heart rate、HRV、jerk、steps、temperature slope 和 pressure。每个 sensor matrix 会被配上一段自动生成的 caption，形成训练 pair。

caption 构造是这篇论文很关键的部分。作者把 caption 分成三层：statistical caption 描述 mean、max、min、std 这类统计值；structural caption 描述 trend、spike、drop、stable 这类时间结构；semantic caption 来自 recognized activity、sleep period 和 mood logs，例如 Outdoor Bike was recorded between minute 1121 and 1133。也就是说，文本不是人工逐条写的，而是从传感器模式和事件记录中模板化生成的。

模型结构上，SensorLM 包括 Sensor Encoder、Text Encoder 和 Multimodal Text Decoder。Sensor Encoder 用 ViT-2D 的思路，把 26×1440 的矩阵切成 2 features × 10 minutes 的 patch；Text Encoder 处理 caption tokens。训练有两条流：一条是 contrastive flow，把 sensor embedding 和 text embedding 对齐；另一条是 captioning flow，让 decoder 在 sensor 条件下自回归生成 caption。CLIP-like 只做对齐，Cap-like 只做生成，CoCa-like 同时优化两类目标。

结果上，20 类活动的 zero-shot AUROC 中，Gemma 和 Gemini 大约只有 0.50，而 SensorLM 达到 0.84。few-shot 中，每类 50 个标签时，SimCLR 是 0.60，SensorLM 是 0.88。caption 消融也说明 semantic caption 是关键：statistical 和 structural 单独使用几乎接近随机，而 structural 加 semantic 可以到 0.84。总体来看，这篇论文的核心证据是：大规模弱语义 caption 加 sensor-text contrastive alignment，能够显著提升传感器表示的语义能力。
