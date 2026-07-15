# Seg-Road 学习笔记：SRT 编码器

本篇整理 **空间自适应缩减 Transformer (Spatial Reduction Transformer, SRT)** 编码器的核心知识点和 PyTorch 实现。目前已开始学习自注意力和空间缩减的基本思想，后续继续补充代码形状变化与 MixFFN。

---

## 1. 核心设计背景

传统的 Transformer（如 ViT）计算自注意力（Self-Attention）的复杂度为 `O(N^2)`，其中 `N = H x W` 是图像的像素数（或 Patch 数）。对于遥感图像常用的 `512 x 512` 大小，`N = 262,144`，其注意力矩阵的计算和内存开销是极其巨大的，常规 GPU 无法承受。

为了解决这个问题，Seg-Road 引入了 **SRT 编码器**，其包含两个核心改进：
1.  **SRA (Spatial Reduction Attention)**：降低自注意力计算复杂度。
2.  **MixFFN**：引入局部上下文，消除显式位置编码。

### 1.1 为什么道路提取需要 Attention

CNN 擅长观察局部区域，但遥感道路可能跨越整张图，并且会被树木、建筑和阴影遮挡。

模型不仅要识别局部纹理，还要判断：

```text
图像一侧的道路，是否和远处被遮挡后的道路属于同一条道路
```

Self-Attention 允许每个位置与其他位置建立联系，因此更适合提取道路的全局结构和长距离依赖。

### 1.2 特征图如何变成 token 序列

卷积特征图通常写成：

```text
B x C x H x W
```

进入 Transformer 前，会把空间位置展开成序列：

```text
B x N x C
N = H x W
```

其中：

*   `B`：batch size；
*   `C`：每个位置的特征通道数；
*   `H x W`：空间尺寸；
*   `N`：token 数量，每个 token 对应一个空间位置。

例如：

```text
H = 128
W = 128
N = 16384
```

### 1.3 Q、K、V 的直觉

Self-Attention 会把每个 token 转换成三种表示：

```text
Q = Query，当前位置想查询什么
K = Key，每个位置可以用什么特征被匹配
V = Value，匹配后真正取回的内容
```

整体流程：

```text
Q 与 K 计算相似度
-> Softmax 得到注意力权重
-> 使用权重加权 V
-> 得到融合全局信息的新特征
```

### 1.4 普通 Attention 为什么是 `O(N^2)`

普通 Self-Attention 中，每个 Query 都要和全部 Key 比较：

```text
N 个 Query x N 个 Key
```

因此注意力矩阵的大小是：

```text
N x N
```

如果特征图为 `128 x 128`：

```text
N = 16384
Attention 矩阵约为 16384 x 16384
```

这会产生超过 2.6 亿个比较位置，计算量和显存占用都很高。

### 1.5 SRA 的核心改进

普通 Attention：

```text
Q token 数量: N
K token 数量: N
V token 数量: N
```

SRA 保持 Query 的完整分辨率，只缩减 Key 和 Value：

```text
Q token 数量: N
K token 数量: N / r^2
V token 数量: N / r^2
```

Query 不缩减，是因为最终仍要为原始的每个位置生成输出。Key 和 Value 可以理解为供 Query 查询的信息库，压缩信息库能够减少计算量。

例如特征图为 `128 x 128`，且 `sr_ratio = 8`：

```text
原始 K/V: 128 x 128 = 16384 tokens
缩减 K/V: 16 x 16 = 256 tokens
```

Attention 矩阵由：

```text
16384 x 16384
```

缩小为：

```text
16384 x 256
```

因此 SRA 在保留全局建模能力的同时，显著降低了计算量。

---

## 2. 空间缩减注意力机制 (SRA)

### 2.1 数学公式
SRA 通过空间缩减操作 `SR(x)`，将特征图 `x` 的空间维度减小 `r^2` 倍（`r` 为缩减比例 `sr_ratio`）。

特征图形状可以理解为：

```text
x shape: HW x C
```

空间缩减公式可以写成：

```text
SR(x) = Norm(Reshape(x, r) * W)
```

其中 `W` 是线性投影矩阵，`Norm` 是 LayerNorm。
缩减后的自注意力机制计算如下：

```text
Attention(Q, K, V) = Softmax((Q * SR(K)^T) / sqrt(d_k)) * SR(V)
```

通过这种机制，Key (`K`) 和 Value (`V`) 的长度从 `N` 缩减为 `N / r^2`，自注意力的计算复杂度从原先的 `O(N^2)` 降低到：

```text
O(N^2 / r^2)
```

### 2.2 PyTorch 代码实现
在 `code/srt.py` 中，作者巧妙地使用了一个**二维卷积 (Conv2d)** 来高效实现 `SR(x)` 的空间缩减和通道投影：

```python
# 初始化时定义卷积层，kernel_size 和 stride 均等于 sr_ratio
self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
self.norm = nn.LayerNorm(dim)

# 前向传播 (forward)
if self.sr_ratio > 1:
    # 1. 恢复 2D 形状: (B, C, H, W)
    x_2d = x.transpose(1, 2).reshape(B, C, H, W)
    # 2. 卷积缩减尺寸: (B, C, H/r, W/r)
    x_reduced = self.sr(x_2d)
    # 3. 展平为序列并 LayerNorm: (B, N/r^2, C)
    x_reduced = x_reduced.flatten(2).transpose(1, 2)
    x_reduced = self.norm(x_reduced)
```

---

## 3. 混合前馈网络 (MixFFN)

### 3.1 消除位置编码
传统的 Transformer 需要显式的位置编码（Positional Encoding）来提供位置信息。但这会导致测试时如果图像分辨率发生变化，位置编码插值会带来性能下降。

### 3.2 深度双通道卷积注入位置信息
MixFFN 在多层感知机（MLP）的两个线性层之间注入了一个 **3x3 的深度卷积 (Depthwise Convolution)**。零填充（Zero-padding）的卷积操作能够天然地为网络提供绝对位置信息。

数学表达为：

```text
x = Linear2(GELU(DepthwiseConv3x3(Linear1(x)))) + x
```

### 3.3 PyTorch 代码实现
在 `code/srt.py` 中的实现如下：

```python
# 深度卷积: groups = hidden_features
self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features)

def forward(self, x, H, W):
    x = self.fc1(x)
    # 转换为 2D 图像格式以进行卷积操作
    B, N, C = x.shape
    x_2d = x.transpose(1, 2).view(B, C, H, W)
    x_2d = self.dwconv(x_2d)
    # 展平回 1D 序列
    x = x_2d.flatten(2).transpose(1, 2)
    x = self.act(x)
    x = self.fc2(x)
    return x
```

---

## 4. 多级编码器结构 (SRTEncoderStage)

Seg-Road 共包含 4 个 Stage，分辨率依次减半，通道数依次增加（以小模型 `s` 为例）：
*   **Stage 1**：输入 `512 x 512 x 3`，通过卷积 Patch Embedding（`11 x 11` 卷积，步长 4）降采样到 `128 x 128 x 32`。`sr_ratio` 设为 8。
*   **Stage 2**：降采样到 `64 x 64 x 64`。`sr_ratio` 设为 4。
*   **Stage 3**：降采样到 `32 x 32 x 160`。`sr_ratio` 设为 2。
*   **Stage 4**：降采样到 `16 x 16 x 256`。`sr_ratio` 设为 1（退化为标准 Attention）。

---

## 5. 实际代码流程：两种缩减要分开

SRT 中有两种容易混淆的“缩小”:

### 5.1 Encoder 的真正下采样

这是网络主干层面的尺寸变化：

```text
512 x 512
  -> 128 x 128
  -> 64 x 64
  -> 32 x 32
  -> 16 x 16
```

它会改变特征图分辨率，也会改变后续 Stage 的输入输出形状。

### 5.2 SRA 内部的 K/V 缩减

SRA 只压缩 Attention 内部的 Key 和 Value：

```text
Q: N 个 token，保持完整分辨率
K: N / r^2 个 token
V: N / r^2 个 token
```

但 SRA 最终仍然输出 `N` 个位置，因此不会改变当前 SRTBlock 的输出分辨率。

一句话区分：

```text
Encoder 下采样改变特征图大小。
SRA 缩减只改变 Attention 内部的 K/V 数量。
```

---

## 6. 多头注意力的形状

在 `SpatialReductionAttention` 中，`dim` 是每个 token 的特征维度，`num_heads` 是注意力头数。

```python
head_dim = dim // num_heads
```

要求：

```text
dim 能被 num_heads 整除
dim = num_heads x head_dim
```

例如：

```text
dim = 160
num_heads = 5
head_dim = 32
```

输入序列：

```text
B x N x C
```

经过线性层和多头拆分：

```text
B x N x C
-> B x heads x N x head_dim
```

不同 head 可以从不同特征子空间学习全局关系。

代码中的缩放因子：

```python
self.scale = head_dim ** -0.5
```

等价于 `1 / sqrt(head_dim)`，用于避免 Query 和 Key 点积过大，导致 Softmax 过度集中。

---

## 7. SRA 的 K/V 形状变化

假设输入特征图为：

```text
B x C x 128 x 128
```

展平后：

```text
B x 16384 x C
```

如果 `sr_ratio = 8`，SRA 内部将特征图变为：

```text
B x C x 128 x 128
-> B x C x 16 x 16
-> B x 256 x C
```

因此：

```text
Q: B x heads x 16384 x head_dim
K: B x heads x 256    x head_dim
V: B x heads x 256    x head_dim
```

Attention 矩阵形状是：

```text
B x heads x 16384 x 256
```

而不是普通 Attention 的：

```text
B x heads x 16384 x 16384
```

计算结果仍然是：

```text
B x heads x 16384 x head_dim
-> B x 16384 x C
```

这样每个原始位置都能获得全局信息，同时避免构造过大的完整 Attention 矩阵。

---

## 8. SRTBlock 的完整结构

一个 `SRTBlock` 由两条残差路径组成：

```text
x
  -> LayerNorm
  -> SRA：全局关系建模
  -> 与原始 x 残差相加
  -> LayerNorm
  -> MixFFN：局部特征和非线性变换
  -> 再次残差相加
```

代码：

```python
x = x + self.attn(self.norm1(x), H, W)
x = x + self.mlp(self.norm2(x), H, W)
```

SRTBlock 的输入输出形状不变：

```text
B x N x C -> B x N x C
```

### 8.1 残差连接

残差连接的思想是：

```text
新特征 = 原始特征 + 当前模块学到的变化
```

这样可以保留原始信息，也让深层网络更容易训练。

### 8.2 SRA 和 MixFFN 的分工

```text
SRA    -> 建立远距离、全局关系
MixFFN -> 处理局部空间信息和非线性特征
```

两者结合后，模型既能看远处道路的整体连通关系，也能看附近像素的边缘和纹理。

---

## 9. SRTEncoderStage 的完整结构

`SRTEncoderStage` 可以理解为：

```text
Patch Embedding 下采样 + 多个 SRTBlock
```

流程：

```text
输入二维特征图
  -> Conv2d Patch Embedding
  -> 展平为 B x N x C
  -> LayerNorm
  -> 重复执行多个 SRTBlock
  -> 还原为 B x C x H x W
```

卷积 Patch Embedding 同时完成：

```text
改变空间尺寸
把输入通道投影到 embed_dim
```

使用卷积而不是硬切 Patch，还能让相邻窗口重叠，更适合道路这种连续结构。

最终四个 Stage 的形状为：

```text
x : B x 3   x 512 x 512
f1: B x 32  x 128 x 128
f2: B x 64  x 64  x 64
f3: B x 160 x 32  x 32
f4: B x 256 x 16  x 16
```

---

## 10. 当前学习状态

已学习：

```text
Self-Attention 的目的
Q/K/V 的基本含义
普通 Attention 的 O(N^2) 问题
SRA 如何缩减 K/V
多头注意力的基本形状
SRTBlock 的 SRA + MixFFN + 残差结构
SRTEncoderStage 的下采样和 Block 堆叠
```

尚未深入：

```text
Softmax 和 Attention 权重的具体数值计算
MixFFN 的逐行代码细节
SRTEncoderStage 的完整 PyTorch 运行验证
训练数据如何进入四级 Encoder
```
