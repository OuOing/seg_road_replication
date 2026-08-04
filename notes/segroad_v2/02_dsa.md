# DSA：从 SRA 到可变形注意力

## 1. 先回忆 SRA

Seg-Road v1 的 SRA 是 Spatial Reduction Attention：

```text
Q：保留 N 个位置
K/V：通过 spatial reduction 缩成 N / r^2 个位置
attention：Q 和缩减后的 K/V 做注意力
```

它解决的是计算量问题：

```text
vanilla attention：N x N
SRA：N x (N / r^2)
```

但 SRA 的采样位置仍然是固定网格。

## 2. DSA 多了什么

DSA 是 Deformable Self-Attention。它在注意力前多做一步：

```text
Q -> linear layer -> offset
offset -> resample Q -> Q_new
Q_new 与 SR(K/V) 做 attention
```

论文给出的 offset 形状是：

```text
offset: (B, N, 2)
```

其中最后的 `2` 表示二维空间坐标。论文把 offset 的值限制在 `[-1, 1]`：

```text
(-1, -1)：特征图左上角
( 1,  1)：特征图右下角
```

这是归一化坐标系，而不是原图中的像素坐标。论文称它为 offset，同时用
`[-1, 1]` 描述采样网格；复现时要再对照官方代码，确认实现使用的是绝对采样坐标，
还是“基础网格 + 位移”后的归一化坐标。

## 3. 直觉理解

SRA：

```text
我在固定位置发问：这个位置应该关注哪些 K/V？
```

DSA：

```text
我先根据内容移动一点，再从更合适的位置发问。
```

这对道路有意义，因为道路：

```text
会弯
会斜
会被树、建筑遮挡
不是规整方块
```

固定采样点可能落在背景上；可变形采样更可能贴着道路结构走。

## 4. 和 DCN 的关系

DSA 借鉴了 deformable convolution 的思想。

DCN 是：

```text
卷积采样点 + offset
```

DSA 是：

```text
attention query position + offset
```

共同点：

```text
都让模型不要死守固定网格
都让采样位置适应目标形状
```

区别：

```text
DCN 偏局部卷积
DSA 偏全局注意力
```

## 5. 公式级理解

SRA 的核心可写成：

```text
Attention(Q, K, V, r) = Softmax(Q @ SR(K)^T / sqrt(d)) @ SR(V)
```

DSA 把 `Q` 换成 `Q_new`：

```text
Q -> offset -> Q_new
Attention(Q_new, K, V, r) = Softmax(Q_new @ SR(K)^T / sqrt(d)) @ SR(V)
```

所以 DSA 本质上不是取消 SRA，而是在 SRA 前加入可学习位置偏移。

更完整的数据流是：

```text
X:                    (B, N, C)
Q, K, V:              (B, N, C)
offset = Linear(Q):   (B, N, 2)
Q_new = Sample(Q):    (B, N, C)
SR(K), SR(V):         (B, N / r^2, C)
attention score:      (B, N, N / r^2)
output:               (B, N, C)
```

论文公式中的分母是 `sqrt(d)`，并说明 `d = C`。如果实现为多头注意力，代码中常见的
缩放因子会使用每个 head 的维度；这是论文符号与具体实现之间需要核对的地方。

## 6. 输入输出关系

DSA 不应该改变 token 数量：

```text
input：  (B, N, C)
output： (B, N, C)
```

它改变的是 attention 采样和信息聚合方式，不是改变 encoder stage 的基本输出形状。

## 7. offset 和 attention weight 不是一回事

这是最容易混淆的知识点：

```text
offset：决定“从哪里取 Q 特征”
attention weight：决定“取到的 Q_new 应该聚合哪些 K/V 信息”
```

计算顺序是：

```text
先移动采样位置
-> 得到 Q_new
-> 再计算 Q_new 与 SR(K) 的相似度
-> Softmax 得到 attention weight
-> 加权聚合 SR(V)
```

所以 DSA 同时学习两件事：在哪里发问，以及更应该听谁的回答。

典型的可微重采样会使用双线性插值：采样位置落在四个网格点之间时，按距离对四个
邻近特征加权。这样 loss 的梯度才能传回 offset 预测层。论文只写了 resample，具体
插值方式应以官方实现为准。

## 8. offset 是怎样学出来的

offset 没有人工提供的“正确移动方向”标签。它由最终任务 loss 间接监督：

```text
segmentation loss / PCS loss
-> attention 输出
-> Q_new
-> 可微采样
-> offset 预测层
```

如果某种移动让分割和连通性预测更准确，反向传播就会强化这种移动。因此它是
end-to-end 学习出来的，不需要额外制作 offset 标注。

但“可学习”不等于一定会学到道路。若数据不足、训练不稳定或 offset 无约束，它也
可能采到相似纹理和背景。论文后续可视化显示，大位移采样点倾向道路及相似纹理区域，
这属于实验观察，不是结构本身的保证。

## 9. 为什么能帮助 road extraction

道路提取关心两类信息：

```text
全局连通：这条路往哪里延伸
局部边缘：路和背景在哪里分开
```

DSA 主要帮助全局连通：

```text
让注意力更集中在道路可能经过的位置
减少背景噪声参与 attention
更适应弯曲道路和不规则道路分布
```

更准确地说，DSA 不会直接保证道路连通。它先改善全局特征选择，连通性还同时依赖
decoder、PCS 监督和最终训练质量。

## 10. 一个小例子

假设低分辨率特征图中的某个 query 原本落在树冠上，但道路从树冠下穿过：

```text
SRA：query 位置固定，容易用树冠特征发问
DSA：offset 把采样位置移向道路边缘，得到 Q_new
     Q_new 再从压缩后的 K/V 中寻找远处同一条路的上下文
```

这里的关键不是“把输出像素挪走”，而是改变 encoder 获取信息的位置；最终输出仍然
与原 token 网格一一对应。

## 11. 复现时的最小目标

不要一开始追完整 SegRoadV2。最小实验可以是：

```text
在现有 SRTBlock 中，把 SRA 替换成简化 DSA
保持 decoder、loss、split、threshold 不变
只看 DSA 是否提高 IoU/F1 或减少 FP/FN
```

判断标准：

```text
如果 Precision 提升：背景线状误检减少
如果 Recall 提升：漏路减少
如果 pred+ 更接近 target+：预测厚度更合理
如果可视化道路更连续：全局结构更好
```

## 12. 本节复习题

1. DSA 相比 SRA 新增了哪条计算路径？
2. offset 与 attention weight 分别控制什么？
3. 为什么 offset 不需要人工标签也能学习？
4. 为什么 DSA 的输入输出 token 数可以保持不变？
5. 为什么只观察 IoU 还不足以证明 DSA 改善了道路连通性？
