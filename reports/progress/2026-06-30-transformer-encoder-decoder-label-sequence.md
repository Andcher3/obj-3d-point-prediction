# 修正 Future Label Sequence 的 Transformer Encoder-Decoder 结构

- 日期：2026-06-30
- 完成内容：根据用户反馈，将此前“MLP decoder”的表述修正为更准确的 `Transformer encoder -> Transformer decoder -> MLP classifier` 结构，明确 MLP 只是分类头，Transformer decoder 才负责生成当前与未来位置的输出 token 序列。
- 本地产出：
  - `reports/ideas/2026-06-30-transformer-encoder-decoder-label-sequence.md`
  - `reports/progress/2026-06-30-transformer-encoder-decoder-label-sequence.md`
- 关键理解/结果：在大量无监督 IMU 预训练和后续 label sequence prediction 的路线下，更推荐使用非自回归 query-based Transformer decoder：用 `q_0...q_H` 表示当前和未来 horizon，decoder 对 encoder memory 做 cross-attention，输出 `o_0...o_H`，再由共享 MLP classifier 得到当前与未来标签分布。
- 下一步：将后续最小模型定义更新为 Transformer encoder + query Transformer decoder + classifier head，并保留 masked IMU reconstruction 作为 encoder 预训练策略。

