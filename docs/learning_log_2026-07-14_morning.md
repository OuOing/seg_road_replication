# 2026-07-14 上午学习总结

本篇记录今天上午的学习内容。今天没有正式进入 `note2.md` 的 SRT / Transformer 细节，主要补齐了训练损失 `loss.py` 和完整模型结构 `model.py` 的整体理解。

---

## 1. 学习节奏调整

今天明确了后续学习方向：

```text
基础语法只做必要补充
重点转向算法思想、模型结构、论文方法和工程实现
```

长期目标是：

```text
先读懂几篇论文
再逐步过渡到 agent 开发
```

因此后续学习应更关注：

*   论文方法为什么这样设计；
*   模型模块之间如何连接；
*   输入输出张量形状如何变化；
*   算法思想如何落到代码；
*   如何验证实现是否正确。

---

## 2. `loss.py`：Seg-Road 的训练目标

`loss.py` 的作用是计算模型预测和真实答案之间的差距，也就是 loss。

可以先把 loss 理解成：

```text
模型错得有多离谱的分数
```

训练目标是：

```text
不断调整模型参数，让 total_loss 变小
```

### 2.1 两个任务

Seg-Road 同时训练两个任务：

```text
任务 1：道路分割
任务 2：PCS 连通性预测
```

对应两组预测和答案：

```text
seg_pred  对比  seg_target
pcs_pred  对比  pcs_target
```

其中：

```text
seg_pred   = 模型预测的道路分割 logits
seg_target = 真实道路 mask
pcs_pred   = 模型预测的 8 方向 PCS logits
pcs_target = 从真实道路 mask 生成的 PCS 标签
```

形状分别是：

```text
seg_pred / seg_target shape: B x 1 x H x W
pcs_pred / pcs_target shape: B x 8 x H x W
```

### 2.2 `SegRoadLoss`

代码中用类封装损失函数：

```python
class SegRoadLoss(nn.Module):
```

它继承 `nn.Module`，因此可以像 PyTorch 模块一样被调用：

```python
loss_fn = SegRoadLoss(alpha=0.2)
total_loss, loss_seg, loss_con = loss_fn(seg_pred, seg_target, pcs_pred, pcs_target)
```

`forward` 是真正计算 loss 的地方。

### 2.3 `BCEWithLogitsLoss`

代码中有两个 BCE 损失：

```python
self.bce_seg = nn.BCEWithLogitsLoss(pos_weight=pos_weight_seg)
self.bce_con = nn.BCEWithLogitsLoss(pos_weight=pos_weight_con)
```

含义：

```text
bce_seg = 分割任务的二分类损失
bce_con = PCS 连通性任务的二分类损失
```

`BCEWithLogitsLoss` 可以理解为：

```text
BCEWithLogitsLoss = sigmoid + BCE
```

模型输出的是 logits，不是 0 到 1 的概率，所以训练时直接使用 `BCEWithLogitsLoss` 更稳定。

### 2.4 总损失

代码：

```python
loss_seg = self.bce_seg(seg_pred, seg_target)
loss_con = self.bce_con(pcs_pred, pcs_target)
total_loss = loss_seg + self.alpha * loss_con
```

含义：

```text
总损失 = 分割损失 + alpha * 连通性损失
```

论文中 `alpha = 0.2`，说明：

```text
道路分割是主任务
PCS 连通性是辅助任务
```

返回三个值：

```text
total_loss = 真正用于反向传播和更新模型
loss_seg   = 用于观察分割任务表现
loss_con   = 用于观察连通性任务表现
```

---

## 3. `model.py`：完整 Seg-Road 模型结构

`model.py` 的核心是把输入图像变成两个输出：

```text
输入 image
  -> seg_out
  -> pcs_out
```

其中：

```text
seg_out shape: B x 1 x H x W
pcs_out shape: B x 8 x H x W
```

`seg_out` 负责道路分割，`pcs_out` 负责 8 方向连通性预测。

### 3.1 整体流程

完整模型流程：

```text
输入遥感图像
  -> SRT Encoder 提取多尺度特征
  -> CNN Decoder 融合多尺度特征
  -> seg_head 输出道路分割
  -> pcs_head 输出 8 方向连通性
```

也可以写成：

```text
x
  -> f1, f2, f3, f4
  -> decoder
  -> seg_out, pcs_out
  -> 上采样回 H x W
```

### 3.2 Encoder 和 Decoder 的职责

Encoder 主要负责：

```text
下采样 + 提取更抽象、更全局的特征
```

Decoder 主要负责：

```text
上采样 + 多尺度特征融合 + 输出预测
```

在 Seg-Road 中：

```text
Encoder: 512 -> 128 -> 64 -> 32 -> 16
Decoder: 64/32/16 -> 128，与 f1 对齐并融合
最后: 128 -> 512，恢复到原图大小
```

### 3.3 多尺度特征

以小模型 `s` 和输入 `512 x 512` 为例：

```text
x : B x 3   x 512 x 512
f1: B x 32  x 128 x 128
f2: B x 64  x 64  x 64
f3: B x 160 x 32  x 32
f4: B x 256 x 16  x 16
```

这些不是把原图拆成几张独立图片，而是同一张图在网络中形成的不同尺度特征图：

```text
f1 分辨率高，细节多
f4 分辨率低，语义强，全局感更强
```

多尺度融合的目的：

```text
同时利用细节、局部结构、高级语义和全局上下文
```

### 3.4 Decoder 的四步

`SegRoadDecoder` 做四件事：

```text
1. 用 1x1 Conv 统一通道数
2. 用 interpolate 统一空间尺寸
3. 用 torch.cat 按通道拼接
4. 用 fusion_conv 融合，再分成两个 head
```

以小模型为例，4 个特征先被投影到统一通道数：

```text
32  -> 128
64  -> 128
160 -> 128
256 -> 128
```

然后空间尺寸统一到 `128 x 128`：

```text
f2: 64 x 64 -> 128 x 128
f3: 32 x 32 -> 128 x 128
f4: 16 x 16 -> 128 x 128
```

拼接后：

```text
4 个 B x 128 x 128 x 128
-> B x 512 x 128 x 128
```

再通过 `fusion_conv` 压回：

```text
B x 512 x 128 x 128
-> B x 128 x 128 x 128
```

最后两个输出头：

```text
seg_head: B x 128 x 128 x 128 -> B x 1 x 128 x 128
pcs_head: B x 128 x 128 x 128 -> B x 8 x 128 x 128
```

完整模型末尾再上采样回原图大小：

```text
B x 1 x 128 x 128 -> B x 1 x 512 x 512
B x 8 x 128 x 128 -> B x 8 x 512 x 512
```

### 3.5 小中大模型配置

`model.py` 中有三个版本：

```text
s = small
m = medium
l = large
```

主要由四个配置控制：

```text
dims      控制每个 stage 的通道数
blocks    控制每个 stage 的深度
heads     控制 attention 的头数
sr_ratios 控制空间缩减比例
```

直觉：

```text
通道越多，模型容量越强
block 越多，网络越深
head 越多，注意力视角越多
sr_ratio 越大，attention 计算越省
```

小模型通常更快，大模型通常更准但更慢。

---

## 4. 当前知识闭环

目前已经能把三个核心文件串起来：

```text
pcs.py:
  road_mask -> pcs_target

model.py:
  image -> seg_out, pcs_out

loss.py:
  seg_out  对比 seg_target
  pcs_out  对比 pcs_target
  得到 total_loss
```

训练核心闭环：

```text
image -> model -> prediction
mask -> pcs.py -> target
prediction + target -> loss.py -> loss
loss -> backward -> 更新模型
```

---

## 5. 还没有正式学习的内容

今天还没有正式进入：

```text
note2.md 中的 SRT / Transformer 编码器细节
srt.py 中的 Spatial Reduction Attention
srt.py 中的 MixFFN
完整训练脚本、Dataset、评估指标
```

下一步可以继续沿着 `model.py` 进入 `srt.py`，学习论文真正的 Transformer 编码器主体。
