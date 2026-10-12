# 跨点信息独立预测实验：无卡准备进展

日期：2026-10-10

## Material Passport

- Origin Skill：academic-research-suite；research-progress-logger。
- 用户授权：准备实验1，下载数据并完成无卡工作；GPU待用户开启。
- 状态：准备进行中；无实际模型forward、训练或性能结果。下载进度核查和CPU代码测试已完成。

## 已完成与当前状态

2026-10-11 00:37 更新：全部资产、DA3 源码/依赖、160 场景真实输入 CPU 预处理、首帧 query/原生 12fps 接口和完整预览交付完成，`cpu_preparation_pass=true`。9+7 项合成测试本地/远端通过，无模型装载/forward/训练。可以进入需用户开启 GPU 的模型接口检查；预测/GT 尺度验证与训练仍未完成。详见 `2026-10-11-cross-point-probe-cpu-ready.md`。

- 连接当前服务器，所有新数据与模型资产写入`/root/autodl-tmp/cross-point-probe-2026-10-10`，不向本地下载大文件。GPU进程未启动，准备进程明确设置空`CUDA_VISIBLE_DEVICES`。
- 使用远端aria2c下载官方MOVi-F的30个训练分片、验证分片4–15，共42文件、6,239,995,339字节（5.811GiB），全部下载完成；CPU准备按发布者MD5重新核验后读取。
- 按固定文件/记录顺序和仅观测期运动选择目标，100训练、20验证、40测试场景均已生成；同场景不跨组，排除原18个开发场景。最终配额、GT与发布者MD5检查通过，`preparation-complete.json`已生成。
- 48点、观测0–6、未来7–21、原生12fps。由于MotionForesight dense输出对应首帧物质点，新预测实验改用首帧目标内部query，参考坐标仍为第6帧相机。未改变现有教师实验query、缓存或评测协议。
- 旧准备函数增加可选的观测选择截止帧和最后观测坐标参考帧，默认行为保持旧实验；7项既有CPU测试通过。为应用新协议，在尚未生成场景前有计划地停止准备/下载进程并从aria2断点恢复，原日志保留。
- 独立预测器输入与训练入口已实现：线性回归与128×128 MLP，两组分别提供或屏蔽8邻点过去轨迹；原有共享信息两组一致。新增6项CPU合成检查本地/远端均通过（共同特征、输入维度、置换、全局平移、训练集标准化、时间/GT输入拒绝）；不代表真实实验收益。
- 本轮准备的是RGB-only预测输入的冻结缓存接口，以及单独标注的GT过去诊断文件。GT未来文件只用于损失/评估。MotionForesight的仅观测缓存需要GPU阶段生成，尚不存在。

## 模型准备问题

当前服务器没有既有MotionForesight微调权重及底座。官方仓库核查后准备源码、TrackCraft3r底座、Wan VAE/DiT/tokenizer与DA3。未为本轮监督生成教师标签。

- 远端官方MotionForesight Google Drive入口先后出现503、超时、直连不可达，IPv4直连有界测试也超时。用户确认无旧文件且不能开启旧服务器。AutoDL官方加速域名列表不包含Google Drive，不能把代理失败推断为发布文件失效。
- 已检索Hugging Face模型/Space名称API、官方GitHub Release和三个分叉Release，均未找到替代微调包；官方公开issue仍请求作者托管至HF。本地仅64KiB探测确认官方usercontent入口可用，包大小293,148,992字节。用户明确授权中转这份约280MiB包；本地aria2c 16连接约21秒下载完成，平均15MiB/s，ZIP CRC通过，内容为best.pt/config.json/last.pt。SHA256为`1c3e4f26a8558b25e4da88f15f1df2e9916a892fefa69248ce9f11485a63d5e9`；上传、两端SHA256与远端ZIP CRC全部通过，仅解压best.pt/config.json。本地临时ZIP已删除，释放293,148,992字节；远端原包保留。未加载模型，未将自算哈希标作发布者哈希。
- 缺少网页解析依赖的首次任务失败已记录；安装`beautifulsoup4`后继续资源定位。Drive问题仍存在，因此单独启动其他资产下载，状态必须保留“缺微调权重”。
- 已将其他大文件有计划地从HF学术加速切换为镜像直连，保留aria2断点和原日志。每文件16连接、最多同时3文件；15:22最新快照总速度1.2MiB/s，TrackCraft3r约651MiB/16.54GiB、725KiB/s、估算6小时23分；Wan DiT约365MiB/5.2GiB、496KiB/s、估算2小时53分。此前TrackCraft3r的估计从18小时降至6小时，说明速度与ETA波动显著，不能承诺整个准备一小时完成。VAE已进入后续下载队列完成状态，仍待最终统一哈希验证。
- 大文件为aria2稀疏文件（不连续位置已写入），文件长度15GB不代表已下载15GB。进度采用aria2实际完成字节，不采用`stat().st_size`。
- 有界线路测试：镜像直连1MiB样本141KiB/s（含连接开销）；HF直连失败、学术加速HF小文件测试超时。镜像实际主体下载已恢复并改善总速度，但TrackCraft3r仍慢。
- 未加载模型验证权重。HF大文件完成后必须核对官方SHA256；Drive发布包只能如实报告取得方式、自算SHA256和ZIP CRC，不能伪称存在官方哈希。

## 输出

- `configs/cross_point_probe_2026-10-10.json`
- `scripts/prepare_cross_point_probe_data.py`
- `scripts/prepare_cross_point_probe_assets.py`
- `scripts/cross_point_probe_features.py`
- `scripts/run_cross_point_probe.py`
- `scripts/test_cross_point_probe.py`
- `scripts/probe_cross_point_asset_routes.py`
- `scripts/probe_motionforesight_checkpoint_sources.py`与`verify_motionforesight_release.py`：限定公开小元数据探测、授权微调包传输校验，无模型反序列化。
- 远端来源清单、aria2记录与GT进展：新实验目录`logs/`。
- 同步更新`reports/progress_log.md`；已传回准备摘要、微调包校验记录、split-manifest及验证集query总览，总览/清单两端SHA256通过；未传回大数据。训练/测试总览尚待传回。
- 已建立本线程10分钟准备检查提醒，正常下载安静，完成/失败通知；禁止GPU与forward，不能把资产下载完成等同全部准备完成。

## GPU前仍需完成

## 下载恢复与提速（约17:05）

- 用户授权重新提速。四线路的同一TrackCraft3r 1MiB区间测试：镜像旧入口159KiB/s、刷新入口179KiB/s（含连接开销），官方学术加速约9KiB/s，官方解析后CDN直连超时；均检查206和正确总大小，未在本地下载大模型。
- 原下载TrackCraft3r已经降至2连接、61KiB/s。核对进程身份后向旧aria2发送SIGINT，控制文件正常保存，旧任务退出后才启动新任务，避免并发写同一文件。
- 新`resume_cross_point_assets.py`复用固定revision与发布者SHA256清单，仅剩TrackCraft3r和Wan DiT两项；已有完成文件经原大小/哈希或ZIP CRC检查保留。没有删除或重头下载部分权重。
- 第一轮16连接短时总速度8.7MiB/s（Track3.1、Wan5.6），随后明显回落，并发现新增20KiB/s低速连接剔除规则会主动减少连接。已撤除此规则，将等待数据超时设为60秒，再次保留断点恢复。
- 当前新协调进程2145/aria2进程2146，两文件16连接。RPC只监听localhost、有保密配置；连续90秒连接≤8且速度<512KiB/s时刷新下载入口，最多每文件4次/每轮，整体最多3轮。该恢复逻辑已配置，当前未触发，不宣称已验证长期稳定。
- 最新实际RPC快照：Track24.50%、0.701MiB/s；Wan70.07%、0.622MiB/s，总1.323MiB/s。早期峰值不代表持续速度，不能承诺一小时完成。
- 交付前复核：Track25.03%、0.606MiB/s，Wan71.69%、0.647MiB/s，仍均16连接；总1.253MiB/s。已更新既有定时检查，动态读新PID与经过筛选的数字日志，权限未变时避免重复阻碍播报。
- 本地4项CPU完成状态测试通过，检查缺失文件/控制文件、LFS大小/哈希、JSON格式及ZIP CRC；脚本编译通过。原微调包与GT保留，无GPU、模型加载、forward或训练。
- 日志可能包含公开文件的临时签名下载地址；后续读取仅输出经过筛选的数字摘要或RPC数值，不打印完整aria2错误行、RPC秘密配置、私有队列或签名URL。
- 新文件：`scripts/probe_trackcraft_download_routes.py`、`resume_cross_point_assets.py`、`test_resume_cross_point_assets.py`。按research-progress-logger保存此处理和速度回落证据；OpenAI Docs仅用于更新既有定时检查，未改变研究方法。

实现依据：[aria2官方连接与RPC文档](https://aria2.github.io/manual/en/html/aria2c.html)。

### 待完成项

10月11日00:07（北京时间）最终下载核查：远端资产完成记录已于10月10日21:51:55生成，`assets_ready=true`、无缺失文件。TrackCraft3r/DA3发布者大小及SHA256通过，Wan DiT/VAE与LFS tokenizer文件的固定revision元数据补查也通过；源码和微调包CRC通过。两轮下载退出码分别-6、0，恢复后成功；协调器及下载器均已退出。数据盘剩12.23GiB，无GPU或模型加载。下载阶段结束，CPU依赖、接口和query预览准备仍未完成，不能直接通知开启GPU。完整记录见`2026-10-11-cross-point-probe-download-complete.md`。

21:47（北京时间）核查：本地会话73279失效后重新建立会话59555，主机指纹一致。远端协调/aria2进程2145/2146仍正常；TrackCraft3r为16.407/16.542GiB（99.19%），剩约138MiB，8连接、0.202MiB/s，瞬时ETA11.4分钟；近6条日志约197–211KiB/s。综合资产完成记录和本轮退出记录尚未生成，不能宣布哈希校验通过。数据盘剩12.38GiB，Wan既有完成及官方哈希记录保持；无下载参数变更、GPU或模型加载。按research-progress-logger记录此次核查。

21:11（北京时间）核查：SSH会话73279传输失效，重新连接会话80164并核对既有主机指纹一致；远端下载继续运行。TrackCraft3r已完成15.489/16.542GiB（93.64%），剩约1.053GiB，15连接、0.262MiB/s，瞬时ETA68.7分钟；近期日志约270–281KiB/s。协调/aria2进程2145/2146仍存在，综合资产完成记录尚未生成，数据盘剩13.86GiB。Wan官方大小/哈希核验通过记录保持；未改变下载参数，未加载模型或使用GPU。按research-progress-logger保存核查结果。

20:19（北京时间）核查：原本地SSH会话2621已失效，重建会话73279连接成功，主机指纹与既有记录一致。远端协调/aria2进程2145/2146仍正常；TrackCraft3r进度82.69%（13.679/16.542GiB），剩2.863GiB，16连接、0.418MiB/s，瞬时ETA116.9分钟。最近6条进度速度约420–439KiB/s，先前2.8–3.2MiB/s峰值未持续；不能沿用30分钟完成估计。数据盘剩15.46GiB；综合资产完成记录尚未生成，Wan官方哈希完成记录保持。未更改下载参数或使用GPU，按research-progress-logger更新记录。

20:00（北京时间）核查：旧本地会话2621失效，使用隐藏认证脚本重新建立会话52471，主机指纹与既有记录一致；远端下载未停止。进程2145/2146正常。TrackCraft3r为12.747/16.542GiB（77.06%），剩3.795GiB，16连接；瞬时0.436MiB/s、ETA148.5分钟，最近日志566–628KiB/s、ETA约105–117分钟。此前约3MiB/s未持续。综合资产完成记录未生成，数据盘剩16.62GiB；Wan完成/官方哈希通过记录保留。未调整下载参数、运行GPU、加载模型或训练。按research-progress-logger保存核查结果。

19:42（北京时间）核查：TrackCraft3r为11.987/16.542GiB（72.46%），剩约4.56GiB，16连接、0.554MiB/s，按当前速度约剩140分钟。最近5条日志速度541–568KiB/s；19:13的2.8–3.2MiB/s未维持，因此此前30–35分钟估计未实现，不能继续使用。进程2145/2146正常，综合资产完成记录未生成，无连接恢复记录，数据盘剩17.07GiB。Wan的既有完成/官方哈希通过记录保持；无下载参数变更、GPU、模型加载或训练。按research-progress-logger记录最新状态。

19:12–19:13（北京时间）核查：TrackCraft3r实时RPC进度65.54%（10.842/16.542GiB），剩约5.70GiB，16连接、2.838MiB/s、瞬时ETA34.3分钟。随后经过筛选的12条进度行显示最近约3分钟速度从580KiB/s逐步升至3.2MiB/s，最后ETA约30分钟；尚不能据此承诺持续速度或完成时间。协调/aria2进程2145/2146均存在，无自动连接恢复记录，综合资产完成记录未生成；数据盘剩18.28GiB。Wan已完成且官方哈希通过的记录保持。无参数更改、GPU、模型加载或训练；按research-progress-logger保存此次核查。

18:11（北京时间）核查：TrackCraft3r已下载7.992/16.542GiB（48.32%），16连接，瞬时0.677MiB/s，按瞬时速度约剩215.5分钟。17:54:23至18:11约17分钟内由41.53%升至48.32%，区间平均约1.15MiB/s；若维持该平均，剩余8.55GiB约需2小时7分钟。仅作为速度波动下的估计，不能承诺完成时刻。协调/aria2进程2145/2146均存在，无自动连接恢复记录，综合资产完成记录未生成；数据盘剩21.29GiB。Wan DiT沿用此前发布者大小/SHA256通过记录，无新增GPU、模型加载或训练，未改变下载参数。按research-progress-logger保存核查结果。

17:56（北京时间）复核：Wan DiT已完成（5,676,070,424字节），从固定官方revision的HF镜像API补查发布者LFS元数据，远端实际SHA256与发布者`96b6b242ca1c2f24e9d02cd6596066fab6d310e2d7538f33ae267cb18d957e8f`一致，大小检查通过。原manifest该文件的bytes/sha256为空，不能把与空值比较的失败解释为文件损坏；本次另存`wan-dit-publisher-audit.json`，未改动运行中的下载清单。TrackCraft3r实时进度41.53%、16连接、0.805MiB/s、估算205分钟，仍有约9.7GiB；磁盘剩22.36GiB。当前aria2 1.36.0本机帮助确认同服务器单下载连接上限为16，Track已经达到该上限，Wan结束后无法直接将另外16路连接转给Track。没有启动第二写入进程、重编译aria2或更改下载参数；两进程正常，无GPU或模型加载。按research-progress-logger更新记录和索引。

17:34（北京时间）下载核查：SSH会话2621可用，协调进程2145及aria2进程2146均存在。实时RPC显示TrackCraft3r已完成5.687/16.542GiB（34.38%），16连接、0.918MiB/s，按当前速度约剩202分钟；Wan DiT为5.042/5.286GiB（95.39%），16连接、0.935MiB/s，约剩4.4分钟。总1.853MiB/s；近期日志中Track速度约0.9–1.7MiB/s，估计仍明显波动。数据盘剩23.65GiB；综合资产完成记录尚未生成，有限连接恢复尚未触发。GT及微调包完成记录仍通过，无模型加载、GPU或训练。待完成项不变；使用research-progress-logger记录本次核查。

16:46复核：新SSH会话2621连接成功、主机指纹一致，远端准备进程1966及aria2c进程1968仍在运行。TrackCraft3r约3.5GiB/16.54GiB（21%），出现服务器EOF（连接提前结束），实际并发降至3连接、139KiB/s，当前ETA约27小时；Wan DiT约2.8GiB/5.2GiB（53%），16连接、477–487KiB/s，ETA约1小时29分。总约0.6MiB/s；数据盘剩29GB。微调包与GT完成记录仍通过，全部资产完成记录尚未生成。此次仅核查，无下载参数变更、GPU或模型运行；需处理底座连接下降后才能给出可信完成估计。

15:45复核：旧SSH session失效后已重新连接，远端下载进程正常；TrackCraft3r约1.5GiB（9%）、684KiB/s、估算剩余6小时23分，Wan DiT约0.9GiB（17%）、596KiB/s、估算2小时7分，总约1.2MiB/s。数据盘剩余33GB。微调权重/GT记录通过，全部资产审核尚未完成；无GPU运行。

160场景与GT审核、微调包远端校验已完成；仍需返回新query预览、完成全部底座哈希审核、DA3源码及兼容依赖准备、检查MotionForesight采样接口和原生12fps槽位解释。当前不具备通知开启GPU的条件。训练入口已准备，未执行真实数据拟合。

技能影响：ARS用于保持实验输入/监督边界和准备状态的证据限定；进展记录技能保存已完成CPU测试与实际下载状态。

官方来源：[MotionForesight仓库](https://github.com/brains-bots-n-behavior/motionforesight)，[TrackCraft3r仓库](https://github.com/cvlab-kaist/TrackCraft3r)，[MOVi-F数据说明](https://github.com/google-research/kubric/blob/main/challenges/movi/README.md)。

下载线路依据：[AutoDL学术加速覆盖范围](https://www.autodl.com/docs/network_turbo/)，[MotionForesight HF托管请求](https://github.com/brains-bots-n-behavior/motionforesight/issues/1)。
