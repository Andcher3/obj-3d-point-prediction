# 点间关系方案现有证据与方法调整判断

- 日期：2026-09-26
- 范围：只复核现有报告和输出；未运行模型或新增实验
- 核验状态：ANALYZED，未独立复现远端实验

## 已核对的证据

- E0/E1 与 E3-controlled 支持组件级规范固定的 L2 图积分、零修正、梯度和同步 query 置换等工程性质；oracle 恢复不代表关系预测器能从真实特征学会修正。
- E2-G 的 2D `k=3` 图在 5/5 视频连通且无 bridge，但共享边评价的 RVE AUC 比原始 3D 图高约 14.56%；后者在 4/5 视频断连。E2-G.1 的修复 3D 图改善部分指标，却依赖四条跨分量 bridge。因此没有证据可冻结唯一默认图。
- 五视频跨上下文 3D 登记均未过闸门；正式 16 帧单上下文检查仅 2/5 通过 t0 anchor 闸门，仍没有可靠的未来监督和与 observed-only `P0,H` 配对的证据。
- 最新 query-only smoke 在 12/16 帧得到有限的 `(T,48,3)` 输出；22 帧在 Wan DiT self-attention 申请额外 13.16 GiB 时 OOM。这是计算资源问题；12/16 帧的形状检查也尚未验证标签质量、三维登记和未来有效性。
- 合成数据上等参数量 edge head 与 node residual 均可优化，没有真实视频优势证据。

## 方法判断

保留冻结 MotionForesight 的 `P0,H`、观测 query 图、显式 baseline relation、时变软 edge correction 和组件级 uniform-gauge L2 integration。固定刚体边长、IRLS/Huber、面积加权 gauge、learned gate、额外物理分支均不因当前证据成为必要模块。

需要调整的是实施与主张：

1. 将 2D symmetric kNN `k=3` 标为工程主候选，修复 3D 图为次级候选；不要把连通性视作真实表面邻接或预测收益证据。当前图可称 query 邻域图。
2. 将“必须获得作者官方 sparse 文件”改为“必须获得同一 query 身份、时间、最后观测相机坐标和有效性经过独立审计的训练标签”。自行重建的标签可以成为替代来源，但字段齐全、finite 或单一上下文均不足以放行。
3. 区分 7+9 的探索性九步监督与作者 7+15 的完整协议。16 帧 query-only smoke 仅支持前者在资源上可能可行，尚未证明其质量；22 帧 OOM 不构成方法假设失败。
4. 在真实数据上尚未证明 `H` 的增量信息和 edge head 相对 node residual、decoded-coordinate relation loss 的收益；因此不冻结 Relation Head 为最终结构，也不宣称已解决点云散开。

## 风险边界

图积分确保边修正形成统一坐标，但不能保证物理合理性、真实形变或全局平移修正。2D 近邻可能连接投影重叠但 3D 分离的部位；错误边的风险仍在。既有主要证据只有五个视频或更少独立视频，按视频及运动类型的泛化结论尚不成立。

## 下一步

遵照当前暂停实验的要求，保留以上方法与监督闸门判断；不启动 TrackCraft3R、Relation Head 或新图搜索。

相关记录：`2026-09-22-surface-relation-v1-to-current-audit.md`、`2026-09-23-surface-relation-formal-e2-g1-review-and-candidate-hierarchy.md`、`2026-09-24-surface-relation-motionforesight-same-coordinate-audit.md`、`2026-09-24-surface-relation-query-only-12-16-22-smoke.md`。
