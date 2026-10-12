# AutoDL 连接验证与首个 L1 测试目标

日期：2026-09-09

## 本轮完成

已通过 SSH 登录 AutoDL 实例，读取远程系统、当前无卡模式资源、磁盘、Python/Conda、依赖与外网连通性，并正常退出连接。项目文件中没有保存登录密码。

## 连接结果

- SSH 端点：`connect.bjb2.seetacloud.com:28369`
- 登录身份：`root`
- 远程系统：Ubuntu 22.04
- 连接、身份验证、命令执行与退出均成功。
- 本机 Windows OpenSSH 与系统 DNS 路径当前异常；本轮通过外部 DNS 解析确认地址，再使用 MSYS2 OpenSSH 建立连接。该路径已经实际验证可用。

## 当前无卡模式的实测状态

- cgroup CPU 配额：0.5 核。
- cgroup 内存上限：2 GB。
- GPU：当前无卡模式下不可访问；`nvidia-smi` 返回权限拒绝，符合用户说明。
- 系统盘：30 GB，已用约 760 MB。
- 数据盘 `/root/autodl-tmp`：50 GB，当前基本为空。
- Python 3.10.8、Conda 22.11.1、Git 2.34.1 可用。
- Conda 仅有 base 环境；NumPy 已安装，PyTorch、Transformers、Accelerate、huggingface_hub 尚未安装。
- 远程直接访问 GitHub 与 Hugging Face 均超时，后续代码和权重获取需要使用可访问镜像、AutoDL 文件传输或其他已验证传输路径。

用户提供的有卡配置为 RTX 5090 32 GB、25 vCPU、90 GB 内存；本轮实例处于无卡模式，因此这些资源未在远程运行时验证。

## 首个实验目标

首个模型测试确定为：验证 FUTURE-L1 的 latent 扰动能否产生有效的未来事件分支。

需要比较同一观察视频和目标条件下的三种生成方式：

1. FUTURE-L1 原生确定性或论文默认推理；
2. 不修改 latent 的普通随机采样；
3. 冻结 FUTURE-L1、在 latent 生成过程中加入扰动，并用事件前瞻反馈维护分支。

至少分别观察：

- 产生的事件等价类数量与重复率；
- 候选和观察条件的相容性；
- 扰动后 latent、后续文字与完整生成分支是否保持一一对应；
- 未来视觉状态是否发生有效变化，是否只改变文字表述；
- 无效、崩坏或偏离目标物体的候选比例；
- 推理时间、显存和分支预算。

本轮只确认远程执行条件和实验目标，尚未安装依赖、下载权重、编写扰动代码或运行模型。

## 下一步

实例切换到有卡模式后，先确认 RTX 5090、CUDA 与 PyTorch，再建立项目目录和可复现环境。随后取得 FUTURE-L1 代码与权重，先复现一条原生推理，记录未扰动 latent 作为基线，再加入最小扰动钩子开展受控比较。

## 本地输出

- 连接证据：`D:\Pycharm\post_graduate_project\reports\remote\2026-09-09-autodl-connectivity-probe.txt`
- 本记录：`D:\Pycharm\post_graduate_project\reports\progress\2026-09-09-autodl-connectivity-and-l1-test-target.md`
- 进展索引：`D:\Pycharm\post_graduate_project\reports\progress_log.md`
