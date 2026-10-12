# 梳理方向一创新点候选

- 日期：2026-06-27
- 完成内容：基于 SensorLM、SensorLLM 和 EVI-MAE 三篇论文，梳理方向一的潜在创新空白，并形成可落地的研究点候选。
- 本地产出：
  - `reports/ideas/2026-06-27-direction1-innovation-seeds.md`
  - `reports/progress/2026-06-27-direction1-innovation-seeds.md`
- 关键理解/结果：三篇论文都在处理传感器到语义表示的问题，但还没有真正覆盖手机拍摄场景下的人体动作、手机/相机运动和场景语义的组合建模。近期最可落地的创新入口是从 SensorLLM 的 trend QA 出发，构造更适合短窗口 IMU/HAR 的 micro-motion caption，加入周期、能量、姿态和跨轴关系，并在 UCI HAR 上验证 few-shot、跨受试者和相似动作区分收益。
- 建议下一步：先实现 UCI HAR 的监督 baseline，再实现 micro-motion caption 生成器和 caption alignment 原型。
