# scene_5702：真实焦距可视化与二维偏移检查

日期：2026-10-10

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent；research-progress-logger。
- Mode：run（用户明确授权对单一样本生成真实焦距视频并检查二维偏移）。
- Verification Status：CPU渲染退出码0，本地文件SHA256与22帧完整视频解码通过；源码/论文核查完成。没有模型重新推理或训练。
- Version Label：scene5702_true_focal_and_direct2d_v1。
- 输入来源：既有scene_5702原始22帧输出、原query和MOVi-F仿真GT。GPU保持关闭。

## 完成工作

- 新增CPU脚本 `scripts/audit_uniquery4r_scene5702_true_focal.py`。
- 三维点与相机位置/朝向保持不变，只将fx/fy替换为对应处理分辨率下的仿真真实焦距；预测主点保持不变。没有换成完整真实相机，未对点逐个平移或重新生成三维坐标。
- 五列视频：直接二维、原三维投影、t0拟合焦距三维投影、真实焦距三维投影、仿真真值。22帧、2560×560、8fps慢放，原始数据12fps。
- 独立重建warp2d绝对归一化坐标到源图像素的转换；核对query往返、t0帧对、GT身份和源帧列表。
- 原始NPZ和GT渲染前后SHA256一致；原三列/四列结果均保留。
- 新增二维检查的3项单元测试全部通过；既有焦距3项单元测试复跑通过。新脚本编译通过。

## 数值证据

所有像素误差以256×256源图计；未来可见点沿用原GT辅助可见性，共682个点时刻。

| 三维回投影设置 | t0平均误差 | 未来可见点平均误差 |
|---|---:|---:|
| 原预测焦距 | 30.35 px | 34.64 px |
| t0拟合焦距 | 13.87 px | 14.90 px |
| 仿真真实焦距，仅替换fx/fy | 40.64 px | 42.04 px |

处理图252×252中的真实fx/fy均为275.625；原预测249.422/370.712；t0拟合521.739/480.465。

真实焦距没有消除偏移，误差进一步增加。对于本样本，当前三维点和相机外参与真实成像参数组合仍然有明显不一致。拟合焦距可吸收这些误差，不等同于恢复真实焦距。当前证据不足以将根因唯一定位为三维点、相机姿态或二者的共同错误。

二维输出检查：

- t0对应 `(6,6)`，在原pair_idx中索引27；没有错取未来帧。
- query与GT在t0坐标完全一致，误差0；输入归一化坐标往返最大误差0.00000813 px。
- 从原warp2d独立重算源像素位置，与缓存二维位置最大误差0。
- t0二维平均位置误差11.4990 px，中位10.2956 px，p90 20.6192 px；点群中心平均偏差为横向+8.1136、纵向+0.5845 px。
- 点群围绕自身中心的标准差，预测/输入query比例为水平0.5509、垂直1.0514。水平压缩已经存在于二维输出，不能由三维投影焦距解释。
- 未来可见点二维绝对位置误差17.3308 px；各点相对自身t0的位移误差7.7687 px。未改动轨迹，此量只用于诊断“随物体运动”和“绝对位置准确”的区别，不能称已校正的标签。
- 点群中心相对t0运动误差6.8598 px，仍有运动误差，不声称未来运动完全正确。
- 直接二维与原三维回投影在t0仍相差平均25.2640 px，未来26.8176 px；外观模式相似不代表两条坐标序列相同。

## 对用户疑问的解释与依据

论文明确warp2d是直接回归的目标绝对坐标，非从query出发的位移。部署源码用query采样源特征、与目标图特征进行cross-attention（以源点特征查询目标图），再由共享MLP生成二维/三维输出。相机由另外的网络输出。t0 self-pair也经过同一预测路径，没有 `output_2d=query_uv` 的硬性赋值。

因此，正确采样输入query不保证模型输出在t0精确等于query；训练监督希望它一致，推理计算不强制一致。二维和三维共享特征和MLP，其定位错误可以相关，焦距不是产生二维缩放/偏移的唯一机制。对于本样本，“绝对位置错误较大、相对位移错误较小”是已观察到的现象；它与粗运动较准确、初始定位有偏差的解释相容，尚不能唯一证明具体特征层或训练环节的根因。

来源：

- [论文3.1、3.3与训练描述](https://arxiv.org/html/2608.17283v1)：绝对[-1,1]目标坐标、共享表示、包含self-pair训练。
- [固定commit模型实现](https://github.com/KosmoResearch/UniQuery4R/blob/95119742e61c64c27b09349035019c8cd7b41e62/uniquery4r/models/uniquery4r.py)：同一query_pair_decoder输出warp2d/warp3d；camera参数独立输出。
- [query采样与目标特征读取](https://github.com/KosmoResearch/UniQuery4R/blob/95119742e61c64c27b09349035019c8cd7b41e62/uniquery4r/models/pair_aggregator.py)。

## 本地交付与记录

- 播放页：`experiences_results/uniquery4r-movif-experiment1-2026-10-09/scene_5702-true-focal.html`。
- 视频与无损t0图：`outputs/scene_5702/scene_5702_true_focal_review.mp4`、`scene_5702_true_focal_review_t0.png`；同目录分帧图。
- 数值原件：同目录 `scene_5702_true_focal_audit.json`。
- 本地检查：`logs/scene_5702-true-focal-local-checks.json`；传输成员哈希：`logs/scene_5702-true-focal-transfer.json`。
- 回传包3,928,742字节，SHA256 `2f0d35f30fc651371f31bae5f11f00ccbbe845eb7765b00dccaa6b550471cf53`。本地/远端一致；6个成员安全解压及哈希通过，视频完整解码22帧，抽查五列t0图。
- 本轮已完成，无新增模型推理、训练或真值修正。继续保留原三维误差与方法状态，不以显示改善作为教师可靠性证明。

技能影响：academic-research-suite要求区分坐标转换检查、模型输出误差与尚未确认的根因；research-progress-logger保存源文件哈希、数值和交付证据。
