# 正式 E2-G / E2-M 执行边界更新

## 冻结内容

- 32/48/64 query 协议保留，48 点为主结果。
- 2D symmetric kNN `k=3` 为主图，observed 3D `k=3` 为对照。
- 公开 MolmoMotion-4B-H3-F30 的 `(3,8,3)` 限制只阻塞 E2-M，不改变 E2-G 的真实密集 query 图实验。

## 代码变化

- `surface_relation_formal_e2_runner.py` 增加 `--gate {e2-g,e2-m}`。
- 增加 `--trajectory-source`，只能取 `controlled_perturbation_baseline`、`trackcraft3r_pseudo_gt` 或 `molmomotion_prediction`。
- 每条记录和结果协议都写入 gate 与 trajectory source，并在 E2-G 结果中声明不能证明 MolmoMotion 或 Relation Head 的密集预测能力。
- 候选 manifest 增加 E2-G/E2-M 允许来源和模型声明边界。

## 当前验证

- Python 编译检查通过。
- sampler 16/48/64 通过。
- 当前旧数据目录没有符合新协议的顶层 bundle，E2-G runner 预检为 `bundle_count=0`、`formal_data_gate_passed=false`。
- 没有用 8 点复制、插值或重复推理制造密集轨迹。

## 尚未完成

6 个视频仍需人工审核 `t0` polygon mask，并生成 15 fps 时间戳、三帧历史 pointmap、未来 9 步 TrackCraft3R pseudo-GT 和真实时间元数据。

