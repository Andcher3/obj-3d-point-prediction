# 方法构想修正：future motion token与并行预测解码

- 日期：2026-07-01
- 完成内容：根据用户补充，将forecast decoder输出从future primitive embedding修正为future dynamic token components，包括future primitive embedding与future statistical embedding，并明确完整future token由预测动态成分、已知channel metadata和未来位置编码组成；同时确认re-encoder使用causal mask，forecast decoder采用parallel query decoder。
- 本地产出：
- reports/progress_log.md
- 关键理解/结果：未来re-encoder输入应写为[z_{1:t,c}, zhat_{t+1:t+n,c}]，其中zhat_{k,c}=ehat_{k,c}+shat_{k,c}+m_c+p_k；预训练损失可采用动态部分对齐或完整token对齐。
- 下一步：将该版本同步进PPT方法页和发言稿，避免继续写成只预测future primitive embeddings。
