# E3 监督定位实验

- 日期：2026-09-24
- 状态：两个定位实验均完成；真实视频 Relation Head 继续暂停

## 完成工作

1. 将四份正式 16 帧质量 JSON 同步到本地并逐字段复核。
2. 对 `9069...` 与浇水壶运行以正式 t0 为第一帧的 16 帧 TrackCraft3R 教师窗口，固定 query 作为 frame-0 anchor，只使用 `t0+1...t0+9`。
3. 将教师 t0 三维点与 observed-only MotionForesight 坐标做四折留出 query 相似登记。
4. 将浇水壶以 `process_res=392` 重跑正式 `[t0-6,...,t0+9]` 窗口，完成分辨率敏感性 A/B。

## t0 首帧教师结果

| 视频 | 有效 query | 留出登记 p90 | 最大误差 | >10% | gate |
|---|---:|---:|---:|---:|---|
| `9069...` | 48 | 14.03% | 39.25% | 7/48 | FAIL |
| 浇水壶 | 48 | 40.02% | 47.26% | 37/48 | FAIL |

固定 t0 为 TrackCraft3R 第一帧并消除回溯身份匹配后，教师坐标与 observed-only 坐标仍未通过登记门槛。当前 vector relation pseudo-GT 路线满足冻结条件。

## 浇水壶分辨率 A/B

| 配置 | Anchor 有效 | p90 | 最大误差 | 漂移 p90 | gate |
|---|---:|---:|---:|---:|---|
| 原高分辨率 | 44/48 | 5.27% | 29.91% | 78.63% | FAIL |
| `process_res=392` | 24/48 | 35.42% | 41.31% | 91.07% | FAIL |

降分辨率没有改善浇水壶质量，反而明显恶化 anchor 误差。分辨率差异不是原高分辨率失败的主要解释。

## 输出

- `experiences_results/surface-relation-e3-locator-2026-09-24/`
- `scripts/surface_relation_e3_t0_teacher_audit.py`
- 四份正式 16 帧 JSON已同步到 `experiences_results/surface-relation-e3-single-context-audit-2026-09-24/`。

## 决策

停止当前 TrackCraft3R/DA3 vector relation pseudo-GT 路线，不生成真实视频训练缓存。后续仅开展合成、oracle 和 E1 受控目标上的 Relation Head 接口、梯度、零修正、constrained-L2、置换与单样本过拟合测试。
