# 参考点约束的方差依据与文献核查

- 日期：2026-10-08
- 任务：回答独立误差下的方差比较、query 身份错误，以及参考点关系约束的前人依据。
- 使用 academic-research-suite 的原始来源核查方式，分别记录数学结论、论文证据与本项目尚未验证的推断。
- 未连接服务器、未执行推理或训练、未修改实验结果；无其他代理参与。

## 方差比较

固定点 i，点 j 和中心 c 的位置估计误差分别为 epsilon_i、epsilon_j、epsilon_c。误差独立时，对任一坐标分量：

`Var(epsilon_i-epsilon_j)=sigma_i²+sigma_j²`。

`Var(epsilon_i-epsilon_c)=sigma_i²+sigma_c²`。

因此 sigma_c² < sigma_j² 时，query–中心向量的误差方差严格小于 query–query 向量。该结论是分布结论，无需用单次共同偏移例子打断用户指定的独立误差比较。参考点是否更准确仍是项目需要测量的量。此处比较两种关系标签，不把它扩展为相对坐标一定优于原点绝对坐标，也不把标签方差降低直接写成训练后预测性能提升。三维比较可使用误差协方差矩阵的迹；标量距离是非线性量，应另作误差传播分析。

## query 身份错误

同一 query 编号应在各帧表示同一物理表面点。车门把手上的点因车辆平移/旋转而移动是正确跟踪；同一编号后来落在车轮、车身其他位置或背景属于身份错误。落在目标掩码内也可能跟错同一物体上的表面点。准确中心可限制范围，单独的径向距离无法辨认同一半径上的不同表面点。这是错误类型说明，未将其写成已确认的本项目主要误差来源。

## 定向检索

- 检索日期：2026-10-08。
- 入口：网页检索；纳入作者论文、作者实验室页面、CVF 与 PMLR 官方来源。
- 关键词：point trajectory prediction distance reference constraint；SE3 Nets pivot motion prediction；particle dynamics object center；SpatialTracker ARAP ablation。
- 纳入：点运动预测或轨迹恢复中实际使用参考点相对位置、共享物体运动或距离损失，且可查方法与对照结果。
- 排除：仅输出物体位姿、没有利用该关系约束点轨迹的工作，以及第三方性能总结。
- 定向检索保留三个代表性来源，非系统综述；不宣称已经穷尽所有中心监督方法。不调用未配置的文献解析服务，不宣称多数据库完整性检查。

## 原始文献与证据

1. Byravan, A., & Fox, D. (2017). SE3-Nets: Learning rigid body motion using deep neural networks. *2017 IEEE International Conference on Robotics and Automation*, 173–180.
   - 作者项目：https://rse-lab.cs.washington.edu/projects/se3-nets-learning-rigid-body-motion-using-deep-neural-networks/
   - 核查方法与数值所用作者稿：https://rse-lab.cs.washington.edu/papers/se3-nets-under-review.pdf
   - 式 (1) 使用 `x'=R(x-p)+p+t`，同一运动部件共享参考点 p 和旋转/平移。每点到参考点的向量通过共享变换生成未来位置，属于结构约束；未使用独立中心教师的距离监督损失。
   - 作者稿 Table I 的 Single Box，小模型平均逐点运动误差为 0.041 m，对照直接运动向量网络为 0.101 m。整体结构有收益；没有单独隔离 pivot 参数的贡献。

2. Li, Y., Lin, T., Yi, K., Bear, D., Yamins, D., Wu, J., Tenenbaum, J., & Torralba, A. (2020). Visual grounding of learned physical models. *Proceedings of the 37th International Conference on Machine Learning*, 119, 5927–5936.
   - 官方条目：https://proceedings.mlr.press/v119/li20j.html
   - 原文：https://proceedings.mlr.press/v119/li20j/li20j.pdf
   - Section 3.3 明确将刚体粒子运动分为物体中心平移和粒子绕中心旋转，共享刚体变换；非刚体粒子另行更新。
   - Table 2 RigidFall 的第 20 个未来步，完整模型位置均方误差为 50.78，移除刚性处理为 198.1，均为表中统一乘以 10^4 后的数值。
   - 对照支持中心关联的共享刚体运动处理相对于独立粒子预测的长期收益，不能独立归因于中心轨迹精度，更不能外推为所有目标刚性。

3. Xiao, Y., Wang, Q., Zhang, S., Xue, N., Peng, S., Shen, Y., & Zhou, X. (2024). SpatialTracker: Tracking any 2D pixels in 3D space. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 20406–20417.
   - 官方条目：https://openaccess.thecvf.com/content/CVPR2024/html/Xiao_SpatialTracker_Tracking_Any_2D_Pixels_in_3D_Space_CVPR_2024_paper.html
   - 原文：https://arxiv.org/html/2404.04319v1
   - Section 3.3 式 (4) 用学习得到的同刚性部件权重约束点对距离随时间保持，训练时直接约束三维轨迹。
   - Table 5 去掉 ARAP 距离约束后，DAVIS 平均位置准确率由 76.3 降至 71.6；AJ 从 61.1 降至 55.1。
   - 是关系损失有收益的直接消融证据；参考点为另一条点轨迹，未专门使用中心。任务是已见视频的轨迹恢复，不能与未见未来预测等同。

## 对本项目的含义

中心参照和点间关系约束均有理论与相关实验依据。前两项更接近中心关联的结构设计，第三项直接验证距离约束的贡献。尚未核实与用户设想完全相同的“外部更准确中心轨迹 + query–中心向量监督”在 MotionForesight 输出端改善预测的工作。该性能主张仍保留为假设；当前不选定新教师、不固定刚体约束、不恢复实验。

本记录核查的是在线读取的原论文内容，未创建 literature_corpus 或声明本地原 PDF 已获取、已完成人工阅读。
