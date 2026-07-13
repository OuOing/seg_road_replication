# Seg-Road 学习笔记：第 2 步（空间自适应缩减 Transformer）

本篇笔记整理了关于 **空间自适应缩减 Transformer (Spatial Reduction Transformer, SRT)** 编码器的核心知识点、数学公式以及 PyTorch 代码实现。

---

## 1. 核心设计背景

传统的 Transformer（如 ViT）计算自注意力（Self-Attention）的复杂度为 $$O(N^2)$$，其中 $$N = H \times W$$ 是图像的像素数（或 Patch 数）。对于遥感图像常用的 $$512 \times 512$$ 大小，$$N = 262,144$$，其注意力矩阵的计算和内存开销是极其巨大的，常规 GPU 无法承受。

为了解决这个问题，Seg-Road 引入了 **SRT 编码器**，其包含两个核心改进：
1.  **SRA (Spatial Reduction Attention)**：降低自注意力计算复杂度。
2.  **MixFFN**：引入局部上下文，消除显式位置编码。

---

## 2. 空间缩减注意力机制 (SRA)

### 2.1 数学公式
SRA 通过空间缩减操作 $$SR(x)$$，将特征图 $$x \in \mathbb{R}^{HW \times C}$$ 的空间维度减小 $$r^2$$ 倍（$$r$$ 为缩减比例 `sr_ratio`）：

$$SR(x) = \text{Norm}(\text{Reshape}(x, r) \cdot W)$$

其中 $$W \in \mathbb{R}^{Cr^2 \times C}$$ 为线性投影矩阵，$$\text{Norm}$$ 为 LayerNorm。
缩减后的自注意力机制计算如下：

$$\text{Attention}(Q, K, V) = \text{Softmax}\left( \frac{Q \cdot SR(K)^T}{\sqrt{d_k}} \right) SR(V)$$

通过这种机制，Key ($$K$$) 和 Value ($$V$$) 的长度从 $$N$$ 缩减为 $$N/r^2$$，自注意力的计算复杂度从原先的 $$O(N^2)$$ 直接降低到：

$$O\left( \frac{N^2}{r^2} \right)$$

### 2.2 PyTorch 代码实现
在 [srt.py](file:///Users/bytedance/.gemini/antigravity-ide/scratch/seg_road_replication/code/srt.py) 中，作者巧妙地使用了一个**二维卷积 (Conv2d)** 来高效实现 $$SR(x)$$ 的空间缩减和通道投影：

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

$$x = \text{Linear}_{2}\left( \text{GELU}\left( \text{DepthwiseConv}_{3\times 3}\left( \text{Linear}_{1}(x) \right) \right) \right) + x$$

### 3.3 PyTorch 代码实现
在 [srt.py](file:///Users/bytedance/.gemini/antigravity-ide/scratch/seg_road_replication/code/srt.py#L82) 中的实现如下：

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
*   **Stage 1**：输入 $$512 \times 512 \times 3$$，通过卷积 Patch Embedding（$$11 \times 11$$ 卷积，步长 4）降采样到 $$128 \times 128 \times 32$$。`sr_ratio` 设为 8。
*   **Stage 2**：降采样到 $$64 \times 64 \times 64$$。`sr_ratio` 设为 4。
*   **Stage 3**：降采样到 $$32 \times 32 \times 160$$。`sr_ratio` 设为 2。
*   **Stage 4**：降采样到 $$16 \times 16 \times 256$$。`sr_ratio` 设为 1（退化为标准 Attention）。
