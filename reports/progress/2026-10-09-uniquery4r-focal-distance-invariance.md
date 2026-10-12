## Material Passport

- Origin Skill: academic-research-suite / experiment-agent；research-progress-logger
- Origin Mode: validate
- Origin Date: 2026-10-09
- Verification Status: ANALYZED
- Version Label: focal_distance_invariance_v1

## 完成检查

用户询问使用此前重新估计焦距的 drift-turn 是否能降低观测与完整输入之间的三维点间距离差异。已检查本地两份诊断脚本及两份原始轨迹哈希，无服务器或模型调用。

`scripts/audit_uniquery4r_drift_cached_outputs.py` 通过 `K_test=K.copy()` 替换 fx/fy，然后执行 `project(raw['tracks_3d'], E, K_test, m)`。输入三维点、相机旋转和位置均保留；重新估计焦距只改变这些点在画面上的投影。

`scripts/compare_uniquery4r_context_pair_distances.py` 使用 `norm(X_i-X_j)` 计算距离，不读取焦距。故应用此前同一焦距修正后，此三维距离对照严格不变；无需再次模型推理或重新拟合距离。

已复核原始文件 SHA256 与既有记录一致：

- 22帧：`f997cdabf9b001f965a79eb68bbb3534813b68c864ff64c14ad9111a1d48f6e7`
- 7帧：`7da1d5237020bdd5bd75fc94633b946e60ff50f088d0ab02a2144ac6a6a1ec40`

## 结果和解释

drift-turn 共同七帧的原始三维距离差异中位数仍为 14.12%；统一整体尺度后仍为 14.19%，p90 仍为 31.24%。这些值继承已有距离计算，未声称进行了新的修正轨迹推理。

此前焦距拟合显著减少三维点的二维投影误差。这只能说明重新投影更接近输入像素位置，不能据此断言三维形状或距离更准确。缺少三维真值，本轮没有判断哪条三维轨迹正确。

如将二维位置及深度通过新焦距重新反投影，会产生另一组三维坐标；这超出此前“保持三维点不变、仅替换投影焦距”的操作。本轮未执行，也不预判其距离差异会降低。

## 输出与边界

新增本记录，更新 `reports/progress_log.md`，未修改实验数组或代码。后续状态保持原样，没有启动训练或新的模型推理。

技能影响：ARS区分二维投影误差与三维点间距离差异，进展记录技能保存源码依据和哈希证据。本轮为计算依赖关系检查，不进行显著性检验或模型性能推广。
