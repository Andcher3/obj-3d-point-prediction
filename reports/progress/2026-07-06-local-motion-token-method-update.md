# 对齐 Local Motion Token 版方法构想

- 日期：2026-07-06
- 完成内容：读取用户提供的新方法说明，和此前已确认的 Motion-Primitive Causal Seq2Seq Transformer 版本进行对照。
- 本地产出：
  - `reports/progress/2026-07-06-local-motion-token-method-update.md`
- 关键理解/结果：新版本将 VQ/codebook primitive token 改为连续 local motion token；归一化单通道 patch 由轻量 1D CNN 编码为 `u_{i,c}`，再与 `mu/sigma` 和 sensor/axis metadata 拼接，经 MLP 投影得到 `h_{i,c}`，加时间位置编码得到 `z_{i,c}`。未来预测从“预测 primitive/statistical embedding”改为“预测 future content representation `hat h_{t+k,c}` 并构造 future local motion token”。
- 下一步：确认 phase 标签来源、future representation alignment 是否 stop-gradient、切换时机下标定义、camera motion 标签体系和是否保留 human-camera coupling。
