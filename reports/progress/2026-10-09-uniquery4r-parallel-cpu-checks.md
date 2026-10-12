# UniQuery4R 下载期间的并行CPU准备

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent；research-progress-logger
- Origin Date: 2026-10-09
- Verification Status: VERIFIED（仅无权重CPU工程检查；不代表模型效果或GPU运行通过）
- Version Label: cpu_no_weights_v1

## 已完成

- 使用既有本地输入执行绘制测试，无额外数据下载、无模型导入或构造。
- 四视频各生成7帧和22帧合成坐标视频，共8个MP4和8张t0图片；逐帧解码检查帧数及1708×520输出尺寸。
- 验证非单位相机旋转、平移、内参的三维投影；无效值、相机后方点及画外点处理通过。验证绘制没有改变传入坐标，且t0左右面板不同的点位未被强行对齐。
- 已人工查看car-turn合成t0图片；四视频像素坐标往返辅助测试复跑通过，最大误差约5.68e-14。此误差仅来自辅助函数的数值计算，不是模型跟踪误差。
- 新增不加载权重的环境检查：官方模型类导入、BaseQuery稀疏查询容器、PyTorch/torchvision/NumPy版本、CUDA12.8及sm_120编译支持。
- 将独立CPU检查任务重新启动，日志`logs/interface-cpu-parallel.log`。依赖安装完成后执行环境检查、官方图像预处理、7/22帧查询对、合成绘制测试、`pip check`及版本记录；不等待权重文件，不调用模型forward。
- 推理入口不再要求无卡模式完整CPU加载模型；保留权重SHA256、query批准、环境/接口检查及明确GPU开启条件，实际加载仍使用官方strict接口。
- 本地Python编译检查、远端Bash语法检查、下载重试模拟测试均通过；未额外下载模型或数据到本机。

## 输出

- `scripts/test_uniquery4r_render_without_model.py`
- `scripts/check_uniquery4r_environment_without_weights.py`
- `scripts/check_uniquery4r_cpu_interface_when_ready.sh`
- `scripts/run_uniquery4r_davis_visual_review.py`
- `experiences_results/uniquery4r-davis-visual-review-2026-10-09/render-smoke-synthetic-NOT-MODEL/render-smoke-summary.json`

## CPU阶段原待检查事项

- 权重完整下载及官方SHA256。完整模型加载、GPU运行和真实轨迹质量均未检查。
- 辅助函数的合成投影通过不代表模型输出的坐标/数值表现已验证。正式可视化保留原始输出及相机参数，禁止人为移动点来制造对齐。

按用户要求只在远端下载；GPU保持关闭。研究记录技能使合成绘制测试和真实模型结果分开记录。

## 远端检查完成

- 17个CUDA/PyTorch wheel通过官方PyPI SHA256后，已直接安装本地wheel，避免pip重复下载大文件；其余依赖安装完成。
- 实测PyTorch2.8.0+cu128、torchvision0.23.0、NumPy1.26.4；编译架构含sm_120。未验证实际GPU驱动或运行。
- 官方类导入和BaseQuery稀疏查询容器检查通过；未构造模型、加载checkpoint或执行forward。
- 四视频官方预处理均为`(22,3,280,504)`；第6帧查询、7/22帧帧对、像素映射和合成投影测试通过。
- 远端合成绘制测试通过，`pip check`报告`No broken requirements found`；任务结束标记`CPU_INTERFACE_COMPLETE_NO_INFERENCE`。
- 证据已同步至`experiences_results/uniquery4r-davis-visual-review-2026-10-09/remote-cpu-checks/`。
- 用户随后重新授权本地权重下载并上传。先前“仅远端下载”限制已被这次明确授权覆盖；本记录的合成测试始终复用既有输入。

## 后续状态

权重上传与远端官方SHA256已通过；用户开启GPU后的严格模型加载和四视频实际推理已完成，见 `2026-10-09-uniquery4r-gpu-visual-inference.md`。本记录仅证明CPU工程检查。真实跟踪质量未数值评估。

按用户清理D盘要求，已删除本地16份可重建的合成视频/图片，保留 `render-smoke-summary.json`、脚本及远端副本；真实模型可视化另存于 `outputs/`，不与合成输出混用。
