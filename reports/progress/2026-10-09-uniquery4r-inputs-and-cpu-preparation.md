# UniQuery4R 输入确认与无卡部署准备

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-09
- Verification Status: UNVERIFIED（模型性能尚未测试）
- Version Label: preparation_v1

## 已完成

- 用户确认四段视频的48个 query 位置。car-turn、drift-turn、boat 使用源帧14–35，query/t0 为源帧20；soccerball 改为源帧8–29，query/t0 为较少遮挡的源帧14。均为模型第6帧查询、7帧观察、15帧未来。
- 原始 DAVIS palette ID=1 掩码内采样，504×280处理图上腐蚀2px后确定性最远点采样，保留原图像素及归一化坐标。输入点确认不代表输出跟踪质量通过。
- 上传四段RGB、掩码、query manifest。远端180个输入文件SHA256全部通过，与本地一致；按manifest读取，避免旧预览窗口的多余帧混入。
- 编写22帧/7帧稀疏推理及可视化脚本。只调整源/目标帧对为 `(6,t)`，保留模型参数。分别显示原始二维对应位置、三维点经预测相机投影的位置，不强制t0平移、不裁剪原始轨迹、不以掩码强制修正。
- 本地辅助函数测试通过：7/22帧对选择、归一化像素往返、合成相机投影及相机后方点处理；像素往返最大误差约5.68e-14。测试使用显式空基类，未加载或调用模型，不能证明模型接口实测成功。

## 远端部署状态

截至本次更新：权重上传和远端官方SHA256检查完成，环境与CPU接口检查通过。用户已开启GPU，四视频22/7帧推理全部完成；下文下载过程为历史记录。当前结果见 `2026-10-09-uniquery4r-gpu-visual-inference.md`。

- 服务器保持无卡模式。数据盘存放环境、权重、下载临时文件，不写凭据。
- `/etc/network_turbo` 已执行；已检查权重aria2c进程继承代理变量，未输出代理值。
- 官方Git下载失败后，改为aria2c获取固定commit的ZIP，现源码下载已完成。commit：`95119742e61c64c27b09349035019c8cd7b41e62`。
- 约6.6GB权重最初通过aria2c从官方固定revision下载。正确revision：`f1143e4cc35eec118e46bc4a47a680d09125573a`（40字符）。原下载程序使用的版本号正确；此前本记录及镜像连接测试中多录了一个字符，已修正。完成后在远端与官方SHA256比对，无卡准备不要求完整模型加载。
- 官方PyTorch/CUDA依赖下载多次超时。清华源直连连接测试通过，已停止特定的停滞pip下载进程，改用清华镜像安装同版本PyTorch2.8.0/torchvision0.23.0，要求验证CUDA12.8构建。此前HF镜像404来自错误版本号，不能用于判断正确文件的可用性。
- 尚未完成全部环境/权重检查，未开启显卡、未调用模型forward、未产生轨迹或数值质量结果。

## 文件

- `experiences_results/uniquery4r-davis-visual-review-2026-10-09/`：输入、已确认query图、哈希及辅助函数测试。
- `scripts/run_uniquery4r_davis_visual_review.py`：可视化推理脚本；需要明确 `--gpu-enabled` 才能推理。
- `scripts/audit_uniquery4r_approved_inputs.py`：传输与协议完整性检查。
- `scripts/fetch_uniquery4r_assets.py`、`scripts/prepare_uniquery4r_environment_mirror.sh`、`scripts/check_uniquery4r_download_cpu.py`：部署脚本。
- 远端目录：`/root/autodl-tmp/uniquery4r-visual-review`，日志为 `logs/prepare-aria2.log` 与 `logs/environment-mirror.log`。

## 无卡阶段原待完成事项（已完成）

权重下载及SHA256校验、环境安装、官方预处理与实际类的CPU接口检查。无卡模式实际限制为0.5 CPU和2GB内存，完整模型加载延后至用户明确开启GPU后；文件完整性可用8MB分块SHA256检查，不依赖模型加载。无卡准备通过后通知用户开启GPU，按其要求优先返回四段轨迹视频及t0图片；数值质量审计和训练均未开始。

本线程5分钟间隔准备状态检查（automation ID `uniquery4r`）当前为PAUSED，未自行恢复。检查不授权开启GPU或运行推理。

技能影响：academic-research-suite 要求区分输入/工程检查与模型性能证据；research-progress-logger 用于保留该阶段记录。

## 下载中断定位与恢复

- 日期：2026-10-09；用户授权检查中断原因并恢复、改进下载。
- 原权重aria2日志停在415MiB/6.1GiB，出现HTTP503、EOF（连接提前关闭）及超时；下载退出码1，主准备脚本随之结束。系统盘仅3%使用、数据盘有49GB余量，无磁盘满的证据。
- 故障定位到下载传输链；现有信息无法唯一区分上游服务与学术加速代理链，更未证明16连接就是503原因。
- 正确revision的1MiB区间测试：镜像直连返回206、完整接收1MiB（约105KB/s）；官方经学术加速20秒仅约341KB；镜像经代理超时。选择国内镜像直连，国际来源保留学术加速设置。
- 原checkpoint及`.aria2`控制文件保留，源码ZIP通过CRC后复用。恢复下载后至少增长到477MiB（7%）；速度有波动，不能宣称网络已永久恢复。
- 修改下载器：并发16→4、连接超时30s、读取超时180s、内部最多10次重试、每10s保存断点；传输类失败最多8轮恢复尝试，30s起递增等待，上限300s。校验/配置类错误立即停止。保留原部分文件，不自动删除或覆盖。
- 固定官方metadata：bytes=`6601806018`，SHA256=`cd71b27bbb0c7a22de36d1eacebea84d93196a4da6296c550b1ea27f3a1c5539`。镜像文件仍必须通过这一官方哈希；禁止仅因下载完成就放行GPU。
- 离线重试测试通过：模拟503/超时退出的恢复等待、代理只在下载子进程去除、低并发参数、非传输故障立即停止；未使用模型。
- 已启动`resume_preparation_cpu.sh`，新日志`logs/resume-preparation.log`。原PyTorch镜像安装未重启或覆盖，其888025420字节主wheel已完成，继续获取CUDA依赖。
- 新准备流程等待既有依赖安装完成，然后CPU接口检查、权重严格加载与`pip check`；仍无GPU推理、训练或质量指标。
- 原自动检查无法写入交互SSH会话，未获得状态。当前配置已处于PAUSED，未自行恢复该通知。已编写并上传只读监控脚本，尚未启动；可用于读取既有监控输出，避免定时任务发送远端执行命令。该替代读取方式在定时权限下尚未验证。

## 当前下载路线与资源限制

- 镜像直连子进程不继承代理，学术加速设置仅保留给外网来源；关闭全局学术加速不会改变当前镜像直连路径。
- 为尝试提速，远端权重调整为16连接并保留断点，新日志为`logs/resume-preparation-16.log`。初期速度曾提高，随后下降；最近读取为约19%、80–90KiB/s，不能保证剩余时间在1小时以内。
- CUDA依赖改用远端aria2多连接下载，复用原pip完整及部分wheel；根据官方PyPI元数据逐文件SHA256校验。最新日志显示仍在下载，尚未安装完成。
- 曾启动本地权重下载测试；用户此前禁止消耗本地流量后，已停止任务专属aria2进程PID40192，并检查本机无剩余aria2进程。当时未进行权重转传；本地已有部分文件保留。
- 无卡完整性校验只要求官方文件大小与SHA256一致；完整模型严格加载属于开启GPU后的运行验证，不用它阻塞无卡下载完整性检查。
- 本次没有开启GPU、模型forward、训练或计算跟踪质量指标。

## 用户重新授权本地下载与上传

- 用户明确允许使用本地流量下载权重后上传，覆盖此前仅远端下载的限制。恢复原aria2断点（约84%）；本地下载已完成，恢复后约14–16MiB/s。
- 本地文件大小6601806018字节，与固定官方SHA256完全一致；未加载模型验证文件完整性。
- 上传先写入独立`weights/uniquery4r.pth.incoming-verified-local`，不覆盖正在下载的远端checkpoint。前4分多钟平均约4.3MiB/s；上传尚未完成，不能把小文件测速当成最终完成时间保证。
- 已准备远端分块SHA256与安装脚本：上传完整且校验一致后才停止该任务下载进程，将原checkpoint及aria2控制文件重命名保留，安装新文件；校验失败时不替换原断点。
- CUDA依赖的17个wheel已全部通过官方PyPI SHA256；发现`pip --find-links`仍优先选择远程同版本PyTorch包，已停止重复下载，改用明确本地wheel路径安装全部大包，再补装小型依赖。
- GPU仍关闭，未调用模型forward、训练或质量指标。下载/上传完成状态以实际工具结果为准。

## 最终准备结果及本地清理

- 权重上传耗时1642.1秒，远端6601806018字节与官方SHA256完全一致；完成后才安装至正式权重路径，原远端断点保留。
- PyTorch2.8.0+cu128、sm_120编译支持、官方预处理、7/22帧query接口和 `pip check` 通过。无卡2GB内存阶段未完整加载模型；GPU开启后实际严格加载和forward通过。
- 已删除本地6.6GB权重、SHA256确认与解压文件一致的MotionForesight发布ZIP、合成绘制测试媒体及输入传输临时包。保留真实视频、掩码、query、原实验数组、报告和实际模型文件。
- 清理共约6.93GB，清理后D盘可用空间约8.30GiB；推理结果传回会占用少量空间。删除项见 `experiences_results/uniquery4r-davis-visual-review-2026-10-09/local-cleanup-verified.json`。本地权重可从已验证远端重新取得，合成媒体可由保留脚本生成，重复ZIP的解压文件保留。
