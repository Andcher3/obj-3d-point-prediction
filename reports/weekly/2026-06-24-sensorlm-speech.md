# SensorLM 组会发言稿

预计时长：约 150 秒

大家好，下面汇报本周围绕方向一阅读的一篇近期论文：SensorLM。

方向一关注手机 IMU 下的人体行为和手机运镜行为。传统 HAR 通常是把一段 sensor time series 映射到 activity label。SensorLM 的核心想法是：先把传感器信号对齐到自然语言可以描述的语义空间，再支持识别、检索和生成描述。

它的数据来自 Fitbit 和 Pixel Watch，输入是一整天的 26×1440 分钟级特征矩阵，规模包括 59.7M 以上小时、10.3 万以上用户。作者希望解决的问题是：连续数值信号需要一个语义接口，才能被语言模型和下游任务利用。

方法上，作者先把数据自动写成三类 caption：statistical 描述统计值，structural 描述趋势和突变，semantic 描述 Walk、Sleep、Outdoor Bike 这类活动语义。训练时，sensor 和 text 分别编码，用对比损失对齐；再用 decoder 在 sensor 条件下生成 caption。

结果上，20 类活动 zero-shot 识别中，Gemma 和 Gemini 的 AUROC 大约 0.50，SensorLM 达到 0.84。few-shot 下每类 50 个标签时，DINO、MSN、SimCLR 大约在 0.54 到 0.60，SensorLM 到 0.88。caption 消融也说明 semantic 信息很关键。

对方向一的启发是，手机 IMU 或运镜信号可以先学习一种和动作、运镜、场景语义对齐的表示，再接人体动作识别、相机运动识别或组合行为建模。下一步我准备继续看 SensorLLM 和 Video-IMU MAE：前者更直接面向 motion sensor HAR，后者更贴近相机视角和 IMU 融合。本周主线就是从传统 HAR 分类，转向 sensor-language alignment 对方向一的启发。
