# Seg-Road 学习笔记：完整模型结构

本篇整理 `code/model.py` 的整体算法流程，重点理解多尺度 Encoder、CNN Decoder、双输出分支以及张量形状变化。

---

## 1. `model.py` 的任务

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

### 1.1 整体流程

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

### 1.2 Encoder 和 Decoder 的职责

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

### 1.3 多尺度特征

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

### 1.4 Decoder 的四步

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

### 1.5 小中大模型配置

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

## 2. 当前知识闭环

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

## 3. 后续学习内容

当前还没有正式进入：

```text
notes/04_srt_encoder.md 中的 SRT / Transformer 编码器细节
srt.py 中的 Spatial Reduction Attention
srt.py 中的 MixFFN
完整训练脚本、Dataset、评估指标
```

下一步可以继续沿着 `model.py` 进入 `srt.py`，学习论文真正的 Transformer 编码器主体。
