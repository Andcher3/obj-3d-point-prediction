# 细化方向一 transition token、自监督预测与通道独立设计

- 日期：2026-06-29
- 完成内容：围绕用户对 `<sit-2-stand>` transition label、next-token 自监督预测、future label sequence 预测和 P2LHAP channel-independent 设计的疑问，进一步细化方向一建模取舍。
- 本地产出：
  - `reports/ideas/2026-06-29-direction1-transition-token-refinement.md`
  - `reports/progress/2026-06-29-direction1-transition-token-refinement.md`
- 关键理解/结果：`<sit-2-stand>` 等 transition label 有合理性，但更适合作为 `boundary_flag=1` 时的辅助 transition-type 预测，而不是普通主类别；标签稀缺条件下应优先用自监督 IMU dynamics 预训练，再用少量标签训练未来 label sequence decoder；通道独立应继承 P2LHAP 作为前端或基线，但方向一需要跨通道融合来建模 acc/gyro 和三轴耦合。
- 下一步：把实验设计明确为三组结构消融：early fusion、P2LHAP-style channel independent、hybrid channel-independent stem + cross-channel fusion。

