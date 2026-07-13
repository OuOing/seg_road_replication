# Seg-Road 专题补充：损失函数与训练术语

本篇是学习过程中的专题补充，不作为主线编号笔记。它整理 Seg-Road 训练时提前遇到的几个核心概念：`seg_out`、`seg_target`、`pcs_out`、`pcs_target`、`BCEWithLogitsLoss`，以及变量名中的 `_con` 含义。

---

## 1. PCS 的全称

PCS 的全称是：

```text
Pixel Connectivity Structure
```

中文可以理解为：

```text
像素连通性结构
```

在 Seg-Road 中，PCS 的作用不是直接判断“哪里是道路”，而是判断：

```text
道路像素和周围 8 个方向上的道路像素是否连通
```

因此 PCS 是道路分割任务的辅助监督信号，用来帮助模型减少道路断裂。

---

## 2. 预测值和标准答案

训练神经网络时，经常会同时出现两类变量：

```text
模型预测值
真实标准答案
```

在 Seg-Road 中有两组对比关系：

```text
seg_out  对比  seg_target
pcs_out  对比  pcs_target
```

### 2.1 `seg_out` 和 `seg_target`

`seg` 是 segmentation 的缩写，表示道路分割。

`seg_out` 是模型的分割分支输出：

```text
seg_out shape: B x 1 x H x W
```

它回答：

```text
模型认为每个像素是不是道路？
```

`seg_target` 是真实道路 mask：

```text
seg_target shape: B x 1 x H x W
```

它回答：

```text
标准答案里每个像素是不是道路？
```

训练时，模型会比较：

```text
seg_out 和 seg_target 差多少
```

差距越大，分割损失 `loss_seg` 越大。

### 2.2 `pcs_out` 和 `pcs_target`

`pcs_out` 是模型的 PCS 分支输出：

```text
pcs_out shape: B x 8 x H x W
```

它回答：

```text
模型认为每个像素在 8 个方向上是否连通？
```

`pcs_target` 是从真实道路 mask 生成出来的 PCS 标准答案：

```text
pcs_target shape: B x 8 x H x W
```

它回答：

```text
真实道路在 8 个方向上到底是否连通？
```

训练时，模型会比较：

```text
pcs_out 和 pcs_target 差多少
```

差距越大，连通性损失 `loss_con` 越大。

---

## 3. 为什么变量里有 `_con`

`con` 是 connectivity 的缩写，意思是：

```text
连通性
```

所以代码里的名字可以这样理解：

*   `loss_seg`：segmentation loss，分割损失；
*   `loss_con`：connectivity loss，连通性损失；
*   `pos_weight_con`：connectivity 分支使用的正样本权重；
*   `bce_con`：用于 PCS 连通性分支的 BCE 损失函数。

PCS 本身就是 Pixel Connectivity Structure，所以 `_con` 通常表示和 PCS 连通性任务有关。

---

## 4. 什么是 BCE

BCE 的全称是：

```text
Binary Cross Entropy
```

中文常叫：

```text
二分类交叉熵
```

“Binary” 表示二分类，也就是每个位置只有两种答案：

```text
0 = 否
1 = 是
```

在道路分割里：

```text
0 = 不是道路
1 = 是道路
```

在 PCS 连通性里：

```text
0 = 这个方向不连通
1 = 这个方向连通
```

BCE 的作用是惩罚错误预测：

*   真实答案是 `1`，模型预测越接近 `0`，惩罚越大；
*   真实答案是 `0`，模型预测越接近 `1`，惩罚越大；
*   模型预测越接近真实答案，惩罚越小。

---

## 5. 什么是 logits

模型最后一层通常先输出原始分数，这些原始分数叫 logits。

logits 不是概率，它可以是任意实数：

```text
-3.0
0.0
2.5
8.0
```

如果要把 logits 变成 0 到 1 之间的概率，需要经过 `sigmoid`：

```text
很大的负数 -> 接近 0
0          -> 0.5
很大的正数 -> 接近 1
```

可以先粗略理解为：

```text
logits = 模型还没转成概率的原始判断分数
```

---

## 6. 什么是 `BCEWithLogitsLoss`

`BCEWithLogitsLoss` 是 PyTorch 里的一个损失函数。

它把两步合在一起做：

```text
第 1 步：对 logits 做 sigmoid，把原始分数变成概率
第 2 步：用 BCE 比较预测概率和真实 0/1 标签
```

也就是说：

```text
BCEWithLogitsLoss = sigmoid + BCE
```

代码中使用它的原因是：

```text
模型输出的是 logits，不是概率
```

而 `BCEWithLogitsLoss` 比手动先 `sigmoid` 再做 BCE 更稳定，所以训练时通常优先使用它。

在 `code/loss.py` 中：

```python
self.bce_seg = nn.BCEWithLogitsLoss(pos_weight=pos_weight_seg)
self.bce_con = nn.BCEWithLogitsLoss(pos_weight=pos_weight_con)
```

含义是：

*   `bce_seg`：负责比较 `seg_pred` 和 `seg_target`；
*   `bce_con`：负责比较 `pcs_pred` 和 `pcs_target`。

---

## 7. Seg-Road 的总损失

Seg-Road 同时训练两个任务：

```text
任务 1：道路分割
任务 2：道路连通性预测
```

对应代码：

```python
loss_seg = self.bce_seg(seg_pred, seg_target)
loss_con = self.bce_con(pcs_pred, pcs_target)
total_loss = loss_seg + self.alpha * loss_con
```

可以理解为：

```text
总损失 = 分割损失 + alpha * 连通性损失
```

其中 `alpha = 0.2`，表示：

```text
分割任务是主任务，PCS 连通性是辅助任务
```

如果 `alpha` 太大，模型可能过度关注连通性；如果 `alpha` 太小，PCS 分支的辅助作用可能不明显。

---

## 8. 本节记住

```text
seg_out 是模型预测的道路图，seg_target 是真实道路答案。
pcs_out 是模型预测的 8 方向连通性，pcs_target 是真实连通性答案。
BCEWithLogitsLoss 用来比较 logits 和 0/1 标签。
_con 表示 connectivity，也就是连通性。
PCS = Pixel Connectivity Structure。
```
