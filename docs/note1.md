# Seg-Road 学习笔记：第 1 步（像素连通性结构 PCS）

本篇笔记整理了关于 **像素连通性结构 (Pixel Connectivity Structure, PCS)** 的核心知识点、数学公式以及 Python 代码中的基础函数应用。

---

## 1. 核心概念与背景

### 1.1 传统遥感道路分割的问题
传统的语义分割模型（如 U-Net 等）使用像素级损失函数（如二分类交叉熵，BCE）。BCE 独立计算每个像素的分类误差，**无法约束道路的全局拓扑结构和连通性**，在有建筑物阴影、树木遮挡时，预测出的道路网经常会出现断裂和碎片化现象。

### 1.2 PCS 的解决思路
在网络输出中引入一个额外的 **PCS 预测分支**。不仅预测当前像素是否为道路，还预测它在 **8 个邻域方向**（左上、上、右上、左、右、左下、下、右下）上、距离为 $$r$$ 的像素是否也是道路。通过这种方式来学习和约束道路像素之间的拓扑连通关系。

---

## 2. 数学公式与实现原理

对于 Ground Truth 道路掩膜 $$Y \in \{0, 1\}^{H \times W}$$，PCS 标签 $$Target_{con} \in \{0, 1\}^{8 \times H \times W}$$ 的第 $$d$$ 个方向、坐标 $$(y, x)$$ 处的计算公式为：

$$Target_{con}(d, y, x) = Y(y, x) \land Y(y + dy_d, x + dx_d)$$

其中 $$(dy_d, dx_d)$$ 是第 $$d$$ 个方向上跨度为 $$r$$ 的偏移量。只有当前像素和目标像素**同为道路 (1)** 时，连通性标签才为 1，否则为 0。

---

## 3. 代码实现核心（以 NumPy 为例）

代码在 [pcs.py](file:///Users/bytedance/.gemini/antigravity-ide/scratch/seg_road_replication/code/pcs.py) 中通过**矩阵平移**与**按位与**操作来实现：

```python
# 8 个方向的偏移向量表示（二维向量）
directions = [
    (-r, -r), (-r, 0), (-r, r),
    (0, -r),           (0, r),
    (r, -r),  (r, 0),  (r, r)
]

for idx, (dy, dx) in enumerate(directions):
    shifted = np.zeros_like(road_mask)
    # 通过边界裁剪获取平移切片，防止越界
    # 最终计算并集：当前点与平移后的邻居点进行与操作
    pcs_label[idx] = road_mask & shifted
```

在推理时，通过反向映射（Reverse Mapping），若预测通道 $$d$$ 的连通值大于设定阈值，则将当前坐标 $$(y, x)$$ 及其邻居 $$(y+dy, x+dx)$$ 的分割值均设为 1，以此减少分割图的断裂。

---

## 4. 基础 Python / NumPy / PyTorch 函数

### 4.1 Python 原生内置函数
*   $$max(a, b)$$ / $$min(a, b)$$：用于平移时的边界截断，防止切片越界。
*   $$enumerate(iterable)$$：同时获取列表的索引（通道号）和元素值（偏移量）。

### 4.2 NumPy 科学计算
*   $$np.zeros(shape, dtype)$$ / $$np.zeros_like(array)$$：初始化全 0 矩阵。为了节省内存，二值标签采用低精度的 $$np.uint8$$ 数据类型（占用 1 字节）。
*   $$np.where(condition)$$：获取连通性概率大于阈值的像素坐标。
*   位运算符 $$&$$：对矩阵进行高效的按位与操作。

### 4.3 PyTorch 张量操作
*   $$tensor.dim()$$：获取张量的维度数量。
*   $$squeeze(dim)$$ / $$unsqueeze(dim)$$：去除或增加大小为 1 的维度。
*   布尔索引 (Boolean Masking)：例如 $$seg\_out[mask] = 1$$，用于无循环的高效批量像素赋值。
