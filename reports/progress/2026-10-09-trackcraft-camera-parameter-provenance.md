# 既有TrackCraft3R轨迹可视化的相机参数来源

- 日期：2026-10-09
- 完成：核对本项目DAVIS轨迹输入生成、TrackCraft3R查询调用与三维投影代码。
- 范围：代码来源核查；未连接服务器或执行推理，未修改已有输入与轨迹。

## 检查结果

1. `scripts/prepare_davis_joint_probe_inputs.py::save_user_npz` 读取DA3的 `prediction.depth`、`prediction.intrinsics` 和 `prediction.extrinsics`，保存输入NPZ。
2. `scripts/run_trackcraft_query_inference.py` 从该NPZ加载内参和外参，将 `intrinsics`、`depth_map`、`extrinsics_w2c` 传入TrackCraft3R的 `predict`。本项目使用的接口返回三维tracks，没有另外返回一组新焦距。
3. `scripts/build_davis_visualization_projection_bundle.py::project` 将轨迹通过相机外参转入当前帧相机坐标，使用 `u=fx*X/Z+cx`、`v=fy*Y/Z+cy` 投影。绘制脚本再转换到原图尺寸。
4. DAVIS样本没有本项目已获取的真实相机标定；这里使用DA3估计值，不能当作焦距真值。

## 发现的近似及限制

`save_user_npz` 的 `fx_fy_cx_cy` 取 `prediction.intrinsics[0]`，整个上下文只保存这一个四元组，后续帧共用。各帧外参仍分别保存。这是本项目适配脚本的选择，不能推定为DA3只能输出首帧内参，也不能推广为所有MotionForesight官方数据协议的要求。

如果真实视频或模型估计包含逐帧焦距变化，这个近似会影响几何与投影；本次仅核查来源，没有量化它对旧实验偏移的贡献，也没有据此断言旧问题均来自焦距。

## 对当前问题的解释

既有流程同样需要焦距。相机参数来自前置DA3，再被TrackCraft3R及可视化使用；UniQuery4R则在自己的forward中同时输出轨迹和相机参数。两条流程都存在几何估计与投影不一致的可能，不能把已见二维贴合当作三维真值证明。

记录技能：research-progress-logger；本次记录保留接口事实、脚本近似与尚未量化影响的区别。
