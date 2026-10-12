# AutoDL 网络加速与数据盘 Conda 环境

日期：2026-09-09

## 已完成工作

- 在远程终端加载 `/etc/network_turbo`，GitHub 与 Hugging Face 探测均返回 HTTP 200。
- 结束加速探测后清除代理变量，再通过清华 Conda 镜像创建独立环境。
- 将 Conda 环境、包缓存、项目、模型、数据集、实验结果及常用缓存统一放到数据盘 `/root/autodl-tmp`。
- 创建激活/退出钩子，验证环境重新激活后所有大文件目录仍指向数据盘。
- 在另一份环境样本上执行回滚脚本，样本被删除，正式环境保持可用。

## 实测结果

- Conda 环境：`/root/autodl-tmp/conda_envs/future-l1-latent`
- Python：3.10.21
- pip：26.2.1
- 环境占用：约 198 MB；Conda 包缓存约 210 MB，均位于数据盘。
- 系统盘本轮仅增加约 16 KiB；主要新增数据约 407 MiB 位于数据盘。
- 最终复验：`ENV_EXISTS=True; PYTHON_VERSION=3.10.21; PREFIX_MATCH=True; ALL_LARGE_PATHS_ON_DATA_DISK=True; STATUS=PASS`

## 数据盘绑定

环境激活时会设置以下目录：

- 项目：`/root/autodl-tmp/projects/future-l1`
- 模型：`/root/autodl-tmp/models/future-l1`
- 数据集：`/root/autodl-tmp/datasets`
- 结果：`/root/autodl-tmp/results/future-l1-latent`
- Hugging Face、Torch、pip、XDG 与 Conda 包缓存：`/root/autodl-tmp/cache/*`、`/root/autodl-tmp/conda_pkgs`

激活命令：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/future-l1-latent
```

## 当前边界

本轮没有安装 PyTorch、FUTURE-L1 代码或权重。当前实例处于无卡模式，后续切换到 RTX 5090 后再按实际驱动与 CUDA 版本选择 PyTorch，并先完成 FUTURE-L1 原生推理基线。

## 输出

- 远程探测记录：`D:\Pycharm\post_graduate_project\reports\remote\2026-09-09-autodl-network-conda-probe.txt`
- 本记录：`D:\Pycharm\post_graduate_project\reports\progress\2026-09-09-autodl-network-turbo-conda-environment.md`
- 远程事务证据：`/root/autodl-tmp/transactions/2026-09-09-future-l1-conda-env/`

## 下一步

有卡模式下确认 GPU、驱动和 CUDA，安装匹配的 PyTorch 与 FUTURE-L1 依赖，复现原生推理后开展 latent 扰动有效性测试。
