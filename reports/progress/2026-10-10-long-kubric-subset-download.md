# 下载三段81帧Kubric真值样本

日期：2026-10-10

## Material Passport

- Origin Skill: research-progress-logger
- Origin Mode: remote download and CPU data inspection
- Verification Status: VERIFIED（下载完整性、RGB解码、真值字段/有限性、首帧坐标转换检查）；未运行模型或验证未来投影。

## 已完成

在用户当前UniQuery4R服务器的数据盘下载三个独立NPZ；使用aria2c、固定发布版本与官方LFS SHA256。首个文件从镜像直连断点续传；两个文件并行下载，CPU检查逐个进行。第三个文件后段切换至官网学术加速，续传速度约2.1MiB/s。其他任务进程未改动。

来源：`trackcraft3r/trackcraft3r-eval`，版本 `df2fc292f46c4afbfbe3459de91bac755dacda97`。

| 文件 | bytes | 实际RGB帧数/独立JPEG数 | 三维真值形状 | SHA256/ZIP CRC |
| --- | ---: | --- | --- | --- |
| `00001.npz` | 215953991 | 81/81 | `(399360,81,3)` | 通过 |
| `00002.npz` | 119899165 | 81/81 | `(399360,81,3)` | 通过 |
| `00003.npz` | 225420176 | 81/81 | `(399360,81,3)` | 通过 |

三段总561273332 bytes，约535.27MiB；RGB为832×480。全部世界坐标轨迹有限，遮挡字段、前景字段、真实内参与真实相机矩阵齐全。实际源帧率未核验，预览统一按12fps播放，不将预览播放时间作为真实运动时长。

## 坐标检查及限制

直接将`extrinsics_w2c_gt`作用于`world_coords`会产生很大的首帧投影误差。原因相关证据：真实相机矩阵以首帧归一化，首帧接近单位矩阵；轨迹保留原世界坐标。不能跳过世界坐标到首帧相机的转换。

保持发布的真实内参固定，利用部分首帧GT三维点与对应像素求解全局旋转/平移（PnP）。留出四分之一检查点不参与求解。分别检查整数像素与半像素中心约定，半像素中心的误差更低：

| 文件 | 半像素中心留出p90（px） | 整数像素留出p90（px） |
| --- | ---: | ---: |
| `00001.npz` | 0.0040545 | 0.0540110 |
| `00002.npz` | 0.0020740 | 0.0171890 |
| `00003.npz` | 0.0037311 | 0.0472551 |

结果支持首帧大偏移来自坐标转换缺项。这是使用已有GT恢复标签参考坐标的诊断，未修改原始轨迹、内参或模型输入，不构成模型性能结果。候选完整转换为`X_camera_t = E_relative_t @ A_world_to_camera0 @ X_world_t`；未来帧投影尚未独立核验。

额外坐标检查的全量读取进程曾以137退出；当前无卡环境内存上限2GiB，但检查时cgroup的`oom_kill`计数为0，不能仅凭退出码唯一归因为内核OOM。改为对ZIP内NPY分块读取后，三段检查退出码0。本地、远端各1项受控坐标单元测试通过，恢复已知旋转/平移与半像素约定。原始下载/解码检查已在此前全部完成。

## 预览与存储

- 完整NPZ保存在远端：`/root/autodl-tmp/uniquery4r-long-kubric-2026-10-10/data/kubric_da3/`。
- 本地仅接收小型RGB视频、六帧拼图和JSON；8份回传文件共685506 bytes，SHA256逐个通过。
- 本地三段MP4均完整解码81帧；已查看三段拼图，内容为物体场景，有平移、旋转、碰撞与遮挡候选，不含用于人体动作诊断的替代任务。
- 本地入口：[原始视频预览](../../experiences_results/uniquery4r-long-kubric-2026-10-10/preview-review.html)。
- 远端准备进程已完成、退出；数据盘仍剩约13GiB。

## 输出与状态

- 配置：`configs/uniquery4r_long_kubric_subset_2026-10-10.json`。
- 下载/CPU检查：`scripts/prepare_long_kubric_subset.py`。
- 坐标诊断：`scripts/audit_long_kubric_coordinates.py`；测试：`scripts/test_long_kubric_coordinates.py`。
- 本地证据：`experiences_results/uniquery4r-long-kubric-2026-10-10/status.json`、`gt-coordinate-audit.json`、`preview-file-manifest.json`。

可供后续22/64帧受控对照筛选。当前未选择具体目标、未采样或批准新query、未运行GPU/模型/训练，现有MOVi-F协议与已批准query未改动。该Kubric集合与UniQuery4R论文WorldTrack基准分开报告；前景字段不是实例分割，后续指定单个物体仍需要目标标注及用户query确认。

来源：[数据卡](https://huggingface.co/datasets/trackcraft3r/trackcraft3r-eval/blob/main/README.md)、[单文件目录](https://huggingface.co/datasets/trackcraft3r/trackcraft3r-eval/tree/main/kubric_da3)。
