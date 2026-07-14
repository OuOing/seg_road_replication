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
