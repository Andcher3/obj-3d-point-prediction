# 跨点信息独立预测实验：CPU 准备完成

日期：2026-10-11（北京时间约 00:37）

## Material Passport

- Origin Skill：academic-research-suite；research-progress-logger。
- 用户授权：继续完成无卡准备；等待用户开启 GPU，未授权当前运行模型或训练。
- 结果类型：输入、源码、依赖、离线资源和 CPU 工程检查；没有真实预测、训练或性能结论。
- GPU、模型加载和 forward：均未执行。

## 完成内容

- 沿用固定 100/20/40 场景划分、每场景 48 点、首帧 query、观测 0–6、未来 7–21、原生 12fps。160 场景均通过 RGB、query/mask、文件哈希、真值尺寸及时间对应检查；query 全部位于目标 mask 内，采样像素位于收缩 2px 后的区域。未根据预测结果选择场景。
- DA3 源码采用 MotionForesight gitlink 锁定的 `41736238f5bced4debf3f2a12375d2466874866d`；远端 aria2 下载，源码 ZIP CRC 通过。按推理导入实际需要安装依赖，没有安装 GUI、Open3D benchmark 或 3DGS 渲染组件。
- Torch 2.8.0/CUDA 12.8、torchvision 0.23.0、NumPy 1.26.4 保持；编译架构信息包含 `sm_120`。MotionForesight loader 与 DA3 API 导入通过，未构造神经网络。CUDA 构建兼容性检查不等于实际 GPU kernel 运行验证。
- 使用 DA3 官方 InputProcessor 对全部 160 场景的仅观测 7 帧进行 CPU 预处理：每场景 `(7,3,392,392)`，数值有限，不提供真值内参或外参。顺序单 worker 避免无卡半核资源下并行开销。392 为深度预处理分辨率；MotionForesight 原生输入仍为 320×576，未改模型结构。
- MotionForesight 原生输出使用首帧点身份，输入接口为 observed RGB 与 observed Pj；代码未提供 fps 参数。保留 MOVi-F 的原生 12fps，只确认输入/监督时间对应，不声称 checkpoint 的训练采样频率与本实验一致。原生相机参考为最后观测帧 6，mean/scale 来自观测 Pj。
- 全部模型资产完成状态与微调包记录交叉核验通过；Wan 固定 revision 的 DiT、VAE、两份 LFS tokenizer 文件大小/发布者 SHA256 与实际完成记录一致。远端镜像元数据重查返回 403，改为本地读取官方公开小 JSON、上传远端比对；没有中转大模型。两份非 LFS 小 JSON 仅检查发布者大小，未伪称存在 SHA256。
- TrackCraft 发布权重头部包含 242 个 BF16 文本编码器张量、约 10.58GiB。实现离线资源映射，未来实际加载时复用这些原始张量，避免重复下载。CPU 检查验证了路径映射、tokenizer 和禁止模型加载的保护；实际权重装载及编码器键/尺寸一致性仍需 GPU 阶段验证。
- 全部 160 张 query 预览及对应清单传回本地，包大小 64,469,941 字节，传输 SHA256、ZIP CRC、323 个文件的逐项哈希通过。生成分训练/验证/测试的浏览页面；只包含 query 帧，没有未来视频。已查看验证集总览，未代用户批准全部 query。
- 9 项准备工具合成测试与 7 项独立预测器合成测试本地/远端均通过；未经授权调用新缓存入口会在模型导入前拒绝执行。编译检查通过。以上只证明工程检查可运行。

## GPU 阶段入口与剩余检查

新增 `scripts/generate_cross_point_model_cache.py`，当前只编译和检查授权保护，没有运行模型：

1. DA3 仅处理 7 帧观测，保存各帧完整内参、外参和深度，不使用首帧焦距替代整段内参。
2. 官方 MotionForesight loader 加载冻结模型，使用原生 Pj 处理与稀疏输出采样。observed_xyz 来自模型解码的观测轨迹，保持首帧点身份；不把每帧同一像素的深度位置误作同一表面点轨迹。
3. 输出过去 7 帧、未来 15 帧及 16 维输出特征。缓存生成代码不打开模拟器 GT 文件，不读取未来 RGB；所有缓存明确标记 `training_coordinate_audit_pass=false`。
4. GPU 接口检查后，核实 DA3/MotionForesight 的估计尺度与模拟器真值尺度，再进入独立预测器训练。不能仅凭同为 last_observed_camera 就认为单位已经一致。训练入口已增加拒绝未审计缓存的检查，并用合成数据验证。

真实尺度审计尚未完成；它依赖模型输出，不能由本轮 CPU 导入测试代替。实际装载、GPU 内存峰值、未来预测质量和跨点信息收益均未得到验证。视频领域与取帧频率的分布差异继续保留。

## 当前状态

- `cpu-preparation-complete.json`：`cpu_preparation_pass=true`。
- 数据盘剩余约 11.66GiB；未删除既有数据或改动其他线程成果。
- query 人工审阅状态：pending；GPU 等待用户开启。
- 下载检查提醒已处于 PAUSED，未额外启动轮询或自动推理。

## 输出

- 本地预览：`experiences_results/cross-point-probe-preparation-2026-10-10/query-review-2026-10-11/query-review.html`。
- 准备证据：同目录上层的 `cpu-input-interface-audit.json`、`cpu-environment-interface-check.json`、`cpu-preparation-tests.json`、`da3-source-complete.json`、`wan-all-publisher-audit.json`、`cpu-preparation-complete.json`。
- 新脚本：`prepare_cross_point_cpu_source.py`、`cross_point_offline_runtime.py`、`audit_cross_point_cpu_inputs.py`、`check_cross_point_cpu_environment.py`、`test_cross_point_cpu_preparation.py`、`receive_cross_point_query_previews.py`、`generate_cross_point_model_cache.py`、`finalize_cross_point_cpu_preparation.py`。
- 远端实验根目录：`/root/autodl-tmp/cross-point-probe-2026-10-10`。

ARS 影响：保留未来真值与预测输入的界限，不把 CPU 准备或坐标名称等同于真实效果/尺度验证；进展记录技能保存完整检查证据和限制。
