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

论文中的 offset 形状大意是：

```text
offset: (B, N, 2)
```

每个 token 预测一个二维偏移，表示它应该往哪里看。

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

## 6. 输入输出关系

DSA 不应该改变 token 数量：

```text
input：  (B, N, C)
output： (B, N, C)
```

它改变的是 attention 采样和信息聚合方式，不是改变 encoder stage 的基本输出形状。

## 7. 为什么能帮助 road extraction

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

## 8. 复现时的最小目标

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
