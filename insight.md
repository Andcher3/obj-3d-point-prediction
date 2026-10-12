1.似乎所有的领域都倾向于将多样性分成“事件级”与“表达级”


下载pawbench，将图和文本送入narration预测模型。![[Pasted image 20260918172140.png]]

从结果上来看现在大多数密集tracking都缺乏点与点之间的**物理约束** 
![[Pasted image 20260918180604.png]]
![[Pasted image 20260918180635.png]]
[SSv2 validation — examples](https://motionforesight.github.io/baseline-comparisons/val.html)

两个点各自误差都“不算特别大”，但：

∥X^it−X^jt∥≠∥Xi0−Xj0∥\|\hat X_i^t-\hat X_j^t\| \neq \|X_i^0-X_j^0\|

随着预测时间增长，每个点的一点小误差方向又不同，最后视觉效果就是你看到的：

**点云逐渐膨胀、收缩、剪切，甚至出现离群点。**

随时间偏移增大：
残差？
连接时间序列

点之间关系？
预测点之间距离？


MolmoMotion数据清洗策略减少真值的虚假偏移
但在制作 MolmoMotion-1M 数据时，作者遇到了：

- 点跟踪漂移；
- 深度噪声；
- 单个点突然脱离物体；
- 同一物体上的点运动不一致。

官方项目页也明确说，它们用 **object-level spatial and temporal consistency priors（物体级时空一致性先验：利用同一物体上的点理应共同运动这一事实）**过滤错误轨迹。

论文里的做法相当具体。

先从同一物体选 16 条可靠的 **anchor tracks（锚点轨迹：作为物体整体运动参考的一小组轨迹）**。

然后对普通点 ii，计算它到这些锚点 kk 的距离：

dik(t)=∥X~it−X~kt∥.d_{ik}(t) = \|\tilde X_i^t-\tilde X_k^t\|.

再计算整个时间段内的典型距离：

dˉik=median⁡tdik(t).\bar d_{ik} = \operatorname{median}_t d_{ik}(t).

如果某个时刻：

dik(t)d_{ik}(t)

突然和正常值相差很远，这个点很可能发生了 tracking drift（跟踪漂移：轨迹逐渐偏离真实物体点）。

于是可以得到一个一致性分数，并把长期不一致的点直接删掉。

这一点非常有意思，因为它实际上说明：

> **和 MotionForesight 几乎相同任务的另一篇工作，已经明确把“同一物体点之间的距离关系”当成检测异常轨迹的重要依据。**

只是 MolmoMotion 目前把这个思想用于**清洗训练数据**，没有继续把它放进 forecasting model 的训练目标里。


