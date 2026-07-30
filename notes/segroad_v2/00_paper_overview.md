# SegRoadV2 论文总览

## 1. 一句话理解

SegRoadV2 是 Seg-Road v1 的结构升级版：它把固定采样的注意力和卷积，升级成可变形采样，并在 decoder 中加入更适合细长道路的条带卷积，同时继续保留 PCS 连通性监督。

```text
SegRoadV1：SRA + MixFFN + CNN Decoder + PCS
SegRoadV2：DSA + GroupDCN + Strip Conv Decoder + PCS
```

核心关键词：

```text
deformable
road geometry
connectivity
strip convolution
re-parameterization
```

## 2. 为什么提出 SegRoadV2

道路提取的难点：

```text
道路细长、弯曲、跨度大
背景复杂，田埂、河岸、建筑边缘容易像道路
遮挡导致道路断裂
道路像素占比低，训练容易偏向背景
```

Seg-Road v1 已经用 PCS 加强连通性，但仍有不足：

```text
SRA 的空间采样位置仍是固定网格
普通卷积只能按固定窗口看局部区域
decoder 的普通卷积不贴合细长道路形状
```

SegRoadV2 的回答是：

```text
让全局注意力可变形
让局部卷积可变形
让 decoder 更像道路形状
继续使用 PCS 监督连通性
```

## 3. 整体结构

论文把模型仍然设计成 encoder-decoder：

```text
image
  -> encoder：DSA + GroupDCN blocks
  -> decoder：re-parameterized strip convolution
  -> segmentation branch
  -> PCS branch
  -> final road mask
```

各模块分工：

```text
DSA：全局上下文 + 可变形注意力采样
GroupDCN：局部细节 + 可变形卷积采样
Strip Conv：贴合细长道路形状
PCS：监督 8 邻域道路连通
```

## 4. 论文报告的结果

DeepGlobe 上的论文结果：

```text
SegRoadv1：67.20 IoU
SegRoadv2-t：63.81 IoU，4.73M 参数
SegRoadv2-s：66.23 IoU，16.77M 参数
SegRoadv2-m：68.09 IoU，33.55M 参数
SegRoadv2-l：69.88 IoU，92.16M 参数
```

注意：SegRoadV2-s 没有超过论文中的 SegRoadv1，m/l 才超过。这说明 v2 的收益和模型规模有关，不能只看名字就默认所有版本都更强。

我们的 v1 当前最好：

```text
IoU：0.4779
F1：0.6468
threshold：0.85
```

所以学习 v2 的目标不是立刻追 69.88，而是理解为什么这些模块能针对 v1 的错误类型做改进。

## 5. 和 v1 的关系

对应关系：

```text
SRA         -> DSA
普通局部卷积 -> GroupDCN
普通 decoder -> Strip Conv Decoder
PCS         -> PCS 保留
```

最重要的思想变化：

```text
从“固定位置看特征”
变成“根据道路形状自适应地看特征”
```

这就是第二篇的主线。
