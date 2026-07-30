# Seg-Road 学习笔记：损失函数与训练目标

本篇整理 `code/loss.py` 的核心算法与相关训练术语，包括预测和标签的对应关系、`BCEWithLogitsLoss`、联合损失以及 `SegRoadLoss` 的执行流程。

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

`model.py` 返回变量时使用 `seg_out` 和 `pcs_out`；传入 `SegRoadLoss.forward(...)` 后，参数名写成 `seg_pred` 和 `pcs_pred`。它们表示的是同一批模型预测：

```text
seg_out  = seg_pred
pcs_out  = pcs_pred
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

## 8. `SegRoadLoss` 类的结构

损失函数被封装为一个 PyTorch 模块：

```python
class SegRoadLoss(nn.Module):
```

继承 `nn.Module` 后，损失对象可以像函数一样调用：

```python
loss_fn = SegRoadLoss(alpha=0.2)
total_loss, loss_seg, loss_con = loss_fn(
    seg_pred,
    seg_target,
    pcs_pred,
    pcs_target,
)
```

PyTorch 会自动进入类中的 `forward(...)` 方法完成计算。

### 8.1 初始化参数

```python
def __init__(self, alpha=0.2, pos_weight_seg=None, pos_weight_con=None):
    super().__init__()
    self.alpha = alpha
```

其中：

*   `self` 表示当前损失函数对象；
*   `super().__init__()` 初始化父类 `nn.Module`；
*   `alpha` 控制 PCS 连通性损失的权重；
*   `pos_weight_seg` 调整道路正样本的重要性；
*   `pos_weight_con` 调整连通正样本的重要性；
*   `None` 表示默认不额外加权。

道路和连通位置通常比背景少，`pos_weight` 可以让正样本预测错误受到更大惩罚。

### 8.2 `forward` 计算流程

```python
def forward(self, seg_pred, seg_target, pcs_pred, pcs_target):
    seg_target = seg_target.float()
    pcs_target = pcs_target.float()

    loss_seg = self.bce_seg(seg_pred, seg_target)
    loss_con = self.bce_con(pcs_pred, pcs_target)

    total_loss = loss_seg + self.alpha * loss_con
    return total_loss, loss_seg, loss_con
```

算法流程：

```text
target 转为浮点张量
-> 分别计算分割损失和连通性损失
-> 按 alpha 合成总损失
-> 返回总损失和两个分支损失
```

`total_loss` 用于反向传播；`loss_seg` 和 `loss_con` 用于分别观察两个任务的训练情况。

---

## 9. 自测代码的作用

`loss.py` 底部使用随机张量验证形状和计算流程：

```python
seg_pred = torch.randn(2, 1, 512, 512)
pcs_pred = torch.randn(2, 8, 512, 512)

seg_target = torch.randint(0, 2, (2, 1, 512, 512)).float()
pcs_target = torch.randint(0, 2, (2, 8, 512, 512)).float()
```

这些是假数据，只用于确认：

```text
输入形状正确
两个 BCE 可以正常计算
联合损失可以正常返回
```

正式训练时，预测来自模型，标签来自真实数据集和 PCS 标签生成函数。

---

## 10. 本节记住

```text
seg_out 是模型预测的道路图，seg_target 是真实道路答案。
pcs_out 是模型预测的 8 方向连通性，pcs_target 是真实连通性答案。
BCEWithLogitsLoss 用来比较 logits 和 0/1 标签。
_con 表示 connectivity，也就是连通性。
PCS = Pixel Connectivity Structure。
```

---

## 11. 从一次训练迭代看完整闭环

一次训练 iteration 可以理解为：模型先做题，再根据答案批改，最后修改参数。

```text
读取 image 和 road_mask
-> 根据 road_mask 生成 pcs_target
-> model(image) 得到 seg_out、pcs_out
-> loss_fn 比较预测和两个 target
-> total_loss.backward() 计算梯度
-> optimizer.step() 更新模型参数
-> optimizer.zero_grad() 清空旧梯度
```

对应的伪代码为：

```python
optimizer.zero_grad()
seg_out, pcs_out = model(image)
pcs_target = build_pcs_target(road_mask)
total_loss, loss_seg, loss_con = loss_fn(
    seg_out, road_mask, pcs_out, pcs_target
)
total_loss.backward()
optimizer.step()
```

这里最关键的方向是：

```text
预测 -> 损失 -> 梯度 -> 参数更新
```

`backward()` 不是重新预测，而是沿着计算过程计算“每个参数应该往哪个方向调整”。`step()` 才真正使用这些梯度修改参数。

### 11.1 为什么需要两个分支一起训练

如果只训练 `seg_out`，模型主要学习“这个像素是不是道路”；加入 `pcs_out` 后，模型还要学习“这个道路像素是否和指定方向上的另一个像素相连”。

```text
seg_loss: 像素类别是否正确
con_loss: 道路连接关系是否正确
total_loss = seg_loss + alpha * con_loss
```

因此 PCS 不是后处理阶段临时补救，而是训练时直接参与参数更新的辅助任务。

---

## 12. 数据、预测与评价指标

### 12.1 一个样本包含什么

道路分割任务中的一个样本通常至少包含：

```text
image:      原始遥感图像，形状通常是 (3, H, W)
road_mask:  道路真值，形状通常是 (1, H, W)
pcs_target: 根据 road_mask 生成的 8 通道连通性真值
```

加入 batch 维度后：

```text
image:      (B, 3, H, W)
road_mask:  (B, 1, H, W)
pcs_target: (B, 8, H, W)
```

其中 `B` 是一次送入模型的图片数量。模型的两个输出必须和对应 target 具有相同形状：

```text
seg_out: (B, 1, H, W)
pcs_out: (B, 8, H, W)
```

### 12.2 从 logits 变成二值预测

训练损失直接使用 logits，但评价指标需要 0/1 预测，因此推理时要先经过 sigmoid：

```text
seg_probability = sigmoid(seg_out)
seg_prediction  = seg_probability > 0.5
```

PCS 分支同理。这里的 `0.5` 是常用阈值，不是模型内部固定产生的数；实际项目也可以在验证集上调整阈值。

### 12.3 四个基本计数

把预测和真实标签逐像素比较，可以得到：

```text
TP: 预测为道路，真实也是道路
FP: 预测为道路，真实是背景
FN: 预测为背景，真实是道路
TN: 预测为背景，真实也是背景
```

道路提取通常更关注 TP、FP、FN，因为背景很多，TN 过多可能让准确率看起来虚高。

### 12.4 常用指标

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
IoU       = TP / (TP + FP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)
```

直觉上：

```text
Precision 高：预测出来的道路大多是真的，误检少
Recall 高：真实道路大多被找到，漏检少
IoU 高：预测道路和真实道路的整体重叠更好
F1 高：Precision 和 Recall 的综合表现好
```

道路断裂通常会增加 `FN`，道路周围出现大片误检通常会增加 `FP`。PCS 的目标之一，就是在不明显增加 FP 的情况下减少道路断裂造成的 FN。

### 12.5 为什么论文同时报告 IoU 和 MIoU

`IoU_road` 只关注道路类别；`MIoU` 通常先分别计算道路和背景的 IoU，再对类别取平均：

```text
MIoU = (IoU_background + IoU_road) / 2
```

因此两者不是重复指标：前者专门观察道路，后者观察道路和背景的整体分割质量。
