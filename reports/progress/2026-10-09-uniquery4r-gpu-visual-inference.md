# UniQuery4R 四物体视频 GPU 推理与可视化

- 日期：2026-10-09
- 状态：四视频实际推理完成；跟踪质量尚未数值评估。

## 已完成

- 用户明确开启GPU后，在RTX PRO 6000 Blackwell Server Edition（约96GB显存）运行官方固定版本权重；严格加载通过。
- car-turn、drift-turn、boat、soccerball各运行22帧完整上下文和7帧观测上下文，共8次forward。每次均使用已批准的48个query，查询帧为模型索引6；未更改点位。
- 原始输出、相机参数、帧对索引、处理坐标与原图映射均保存。未进行t0强制对齐、轨迹裁剪或掩码修正。
- 生成8段轨迹视频和8张t0图：左侧二维对应位置，右侧三维点通过模型相机投影的位置。播放8fps仅用于查看。
- 峰值Torch分配显存11782798336字节（约10.97GiB），没有OOM。任务已结束，检查时GPU使用0MiB、利用率0%。

## 证据与文件

- 远端：`/root/autodl-tmp/uniquery4r-visual-review/outputs/`；日志 `logs/visual-inference-gpu.log`。
- 本地结果目录：`experiences_results/uniquery4r-davis-visual-review-2026-10-09/outputs/`。
- 传输包大小22895570字节，远端SHA256：`8874a88f53f475822f9da784b4d6f781b2c5d653b6af82b05f42c18c93acaa50`。
- 本地传输包SHA256与远端一致，32个原始结果文件已解包。另生成8份H.264播放版本、4张t0输入/教师/观测三栏对照图及关键帧拼图；解码帧数和尺寸通过，原始NPZ哈希保持不变。证据 `visual-packaging-checks.json`。
- 本地集中查看页面：`experiences_results/uniquery4r-davis-visual-review-2026-10-09/visual-review.html`；包装脚本 `scripts/package_uniquery4r_visual_review.py`。
- 脚本：`scripts/run_uniquery4r_davis_visual_review.py`。源码commit `95119742e61c64c27b09349035019c8cd7b41e62`，权重revision `f1143e4cc35eec118e46bc4a47a680d09125573a`。

## 结果解释与后续边界

本次证明模型可在该GPU和已批准输入上运行。三维输出具有模型自身的全局尺度，不能直接解释为米。两次上下文独立估计相机与三维坐标，尚未进行跨上下文登记。

目前只做原始输出可视化，未训练Relation Head、未构造表面距离，也未计算教师质量指标。按用户要求，由可视化审阅决定后续质量评估。

已查看四视频第0、6、13、21槽位拼图。car-turn、boat、soccerball的二维与三维投影总体跟随目标，局部仍有偏移；drift-turn存在明显问题，二维点也有偏离，三维投影在若干帧严重脱离汽车或不在画面中。尚未区分模型回归与该视频相机估计/投影误差，不能把这一现象归因于某一个组件。

屏幕上贴合目标不能证明三维深度正确，遮挡点的表面身份也未验证。三栏t0图分别展示批准输入点、22帧三维投影、7帧三维投影，未把输出移动到输入点。

技能影响：research-progress-logger要求记录工程完成与未验证的性能边界。
