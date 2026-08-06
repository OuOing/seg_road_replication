# PCS、联合损失与分阶段训练

## 1. PCS 在 V2 中不是新模块

SegRoadV2 保留了 Seg-Road v1 的 Pixel Connectivity Structure。它仍然预测与当前像素
相邻的 8 个方向：

```text
左上  上  右上
左   当前   右
左下  下  右下
```

PCS 输出与输入图像分辨率相同，方向通道数为 8。某方向标签为 1 的条件是：

```text
当前像素是道路 AND 对应方向的邻居也是道路
```

因此：

```text
segmentation target：监督“这个像素是不是道路”
PCS target：监督“这个道路像素和邻居是否连接”
```

## 2. PCS 同时作用于训练和推理

训练阶段：

```text
segmentation prediction -> L_seg
PCS prediction          -> L_pcs
```

推理阶段：

```text
PCS prediction
-> reverse derivation / 反向映射
-> connectivity road map
-> 与 segmentation road map 取交集
-> final output
```

论文 Figure 15 以左上方向为例：若某位置预测为与左上邻居连通，反向映射会把对应的邻接
关系恢复到道路图。最终取交集是一种保守筛选：分割分支认为是道路、PCS 也提供连通支持
的区域才进入最终结果。

潜在影响：

```text
可能减少缺乏连通支持的误检
也可能因 PCS 漏检而损失真实道路像素
```

因此 segmentation threshold、PCS threshold 和反向映射规则都应固定并记录。

## 3. V2 的分割损失

论文比较了 Focal、DICE 和 Cross Entropy，最后选择加权 Cross Entropy 得到 `L_seg`。

公式 18 可拆成：

```text
p[n,c] = exp(logit[n,c]) / sum_i exp(logit[n,i])

L_seg = -sum_n sum_c w_c * y[n,c] * log(p[n,c])
```

含义：

```text
n：像素编号
c：类别，主要是 road / background
y[n,c]：one-hot 真实标签
p[n,c]：Softmax 后的类别概率
w_c：类别权重
```

如果真实类别是 road，one-hot 中只有 road 项为 1，所以该像素实际只保留：

```text
-w_road * log(p_road)
```

预测 road 概率越低，惩罚越大。

## 4. 它与我们 V1 的 BCE 有何区别

论文 V2 的公式按两类 Softmax CE 描述：

```text
输出通常可理解为 2 个类别 logits
road 与 background 概率相加为 1
```

我们的 V1 简化实现使用 BCEWithLogitsLoss：

```text
输出 1 个 road logit
Sigmoid 得到 road 概率
background 概率隐含为 1 - road
```

对于纯二分类，两者都能表达 road/background，但输出格式、权重参数和数值实现不同，
checkpoint 不能直接混用。

## 5. PCS loss 的论文歧义

论文写道 `L_pcs` 仍使用 Cross Entropy，但同时描述 PCS 为 8 个二值方向通道。标准
PyTorch CrossEntropyLoss 通常需要：

```text
input： (B, C, H, W)
target： (B, H, W)，每个像素一个类别编号
```

而 8 个独立二值方向通常更自然地使用：

```text
BCE：8 个方向分别二分类
```

或者组织为：

```text
每个方向 2 类，再对 8 个方向分别做 CE
```

论文没有在该段明确给出 PCS logits 的完整类别维度和 reshape 方式，因此正式复现必须
以官方代码为准，不能仅凭“Cross Entropy”四个字决定张量格式。

## 6. 联合损失

论文公式 19：

```text
Loss = L_seg + alpha * L_pcs
alpha = 0.2
```

含义：

```text
L_seg：主任务，决定道路像素分类
L_pcs：辅助任务，学习局部连接关系
0.2：控制 PCS 对总梯度的贡献
```

`0.2` 不等于 PCS 只贡献 20% 的实际梯度，因为两个 loss 的数值尺度可能不同。实验时还
应记录 `L_seg`、`L_pcs` 及梯度量级，而不是只看系数。

## 7. 分阶段类别权重

论文给出的经验策略：

```text
训练前期 road : background = 3 : 1
训练后期 road : background = 1 : 1
```

前期提高 road 权重的原因：

```text
道路像素少
模型容易先学成全背景
提高漏检道路的代价，帮助模型建立前景响应
```

后期恢复 1:1 的原因：

```text
模型已经能找到道路
持续高权重可能造成道路过宽和背景误检
让模型重新平衡 Precision 与 Recall
```

这与我们 V1 训练早期出现的背景塌缩属于同一类问题。但论文只说“early/latter stage”，
没有明确给出类别权重切换的 epoch，属于复现缺失信息。

## 8. 论文的三阶段训练配方

论文总共训练 100 epochs，并使用 Pascal VOC 预训练参数：

| 阶段 | Epoch | Batch size | 初始学习率 | 调度 |
| --- | ---: | ---: | ---: | --- |
| 1 | 0-30 | 32 | 0.001 | 每 epoch 乘 0.92 |
| 2 | 30-90 | 16 | 0.0001 | 每 epoch 乘 0.92 |
| 3 | 90-100 | 4 | 论文称固定学习率 | 不衰减 |

实验硬件为 4 张 RTX 3090。论文没有在该段说明第三阶段固定学习率的具体数值，也没有
说明 batch size 是单卡还是总 batch，精确复现前必须核对官方配置。

## 9. 为什么 batch size 逐阶段减小

论文没有明确解释动机。合理但尚未由论文证实的理解是：

```text
前期大 batch：梯度更稳定，快速建立总体表示
中期降低学习率和 batch：进行更细致优化
末期小 batch + 固定低学习率：短程精修
```

这部分应标记为解释性推断，而不是论文事实。batch size 改变还会改变梯度噪声和每个
epoch 的 optimizer step 数，不能只复制数字而忽略训练环境。

## 10. 与我们的实验差距

论文配方与本地 V1 baseline 至少存在：

```text
100 epochs vs 本地短程训练
4 x RTX 3090 vs Apple MPS
Pascal VOC 预训练 vs 简化初始化
分阶段 batch/LR vs 本地简化调度
动态类别权重 vs 本地固定损失配置
论文 CE 描述 vs 本地 BCEWithLogitsLoss
```

因此本地 V1 IoU 低于论文不能简单归因于“模型结构不行”。训练协议本身已经明显不同。

## 11. 本节复习

```text
PCS 标签：当前像素与邻居都为道路时，该方向为 1
训练：L_seg + 0.2 * L_pcs
推理：PCS 反向映射后与 segmentation 结果取交集
类别权重：前期 3:1，后期 1:1
正式训练：100 epochs，三阶段 batch/LR，使用预训练
```

## 12. 复习题

1. 为什么 PCS 既是辅助监督，又会改变最终推理输出？
2. `alpha=0.2` 为什么不等于 PCS 恰好贡献 20% 梯度？
3. 前期 road 权重较高主要防止什么问题？
4. 为什么后期继续保持 3:1 可能降低 Precision？
5. 论文关于 PCS CE 和训练阶段还缺少哪些复现细节？
