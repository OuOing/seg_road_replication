# Seg-Road 学习笔记：第 1 步（像素连通性结构 PCS）

本篇笔记整理了关于 **像素连通性结构 (Pixel Connectivity Structure, PCS)** 的核心知识点、数学公式以及 Python 代码中的基础函数应用。

---

## 1. 核心概念与背景

### 1.1 传统遥感道路分割的问题
传统的语义分割模型（如 U-Net 等）使用像素级损失函数（如二分类交叉熵，BCE）。BCE 独立计算每个像素的分类误差，**无法约束道路的全局拓扑结构和连通性**，在有建筑物阴影、树木遮挡时，预测出的道路网经常会出现断裂和碎片化现象。

### 1.2 PCS 的解决思路
在网络输出中引入一个额外的 **PCS 预测分支**。不仅预测当前像素是否为道路，还预测它在 **8 个邻域方向**（左上、上、右上、左、右、左下、下、右下）上、距离为 `r` 的像素是否也是道路。通过这种方式来学习和约束道路像素之间的拓扑连通关系。

---

## 2. 数学公式与实现原理

对于 Ground Truth 道路掩膜 `Y`，它的取值只有 0 和 1，形状是：

```text
Y ∈ {0, 1}^{H x W}
```

PCS 标签 `Target_con` 有 8 个方向通道，形状是：

```text
Target_con ∈ {0, 1}^{8 x H x W}
```

第 `d` 个方向、坐标 `(y, x)` 处的计算规则为：

```text
Target_con[d, y, x] = Y[y, x] AND Y[y + dy_d, x + dx_d]
```

其中：

*   `(dy_d, dx_d)` 是第 `d` 个方向上的偏移量；
*   `r` 是偏移距离，例如 `r = 2` 表示隔 2 个像素检查；
*   `AND` 表示“并且”，只有两个位置同时为道路 `1`，结果才是 `1`。

也就是说：只有当前像素 `Y[y, x]` 和目标像素 `Y[y + dy_d, x + dx_d]` 同时为道路，PCS 连通性标签才为 1，否则为 0。

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

在推理时，通过反向映射（Reverse Mapping），若预测通道 `d` 的连通值大于设定阈值，则将当前坐标 `(y, x)` 及其邻居 `(y + dy, x + dx)` 的分割值均设为 1，以此减少分割图的断裂。

---

## 4. 基础 Python / NumPy / PyTorch 函数

### 4.1 Python 原生内置函数
*   `max(a, b)` / `min(a, b)`：用于平移时的边界截断，防止切片越界。
*   `enumerate(iterable)`：同时获取列表的索引（通道号）和元素值（偏移量）。

### 4.2 NumPy 科学计算
*   `np.zeros(shape, dtype)` / `np.zeros_like(array)`：初始化全 0 矩阵。为了节省内存，二值标签采用低精度的 `np.uint8` 数据类型（占用 1 字节）。
*   `np.where(condition)`：获取连通性概率大于阈值的像素坐标。
*   位运算符 `&`：对矩阵进行高效的按位与操作。

### 4.3 PyTorch 张量操作
*   `tensor.dim()`：获取张量的维度数量。
*   `squeeze(dim)` / `unsqueeze(dim)`：去除或增加大小为 1 的维度。
*   布尔索引 (Boolean Masking)：例如 `seg_out[mask] = 1`，用于无循环的高效批量像素赋值。

---

## 5. 零基础学习记录：PCS 函数前半段代码拆解

本节对应代码：`code/pcs.py` 中的 `generate_pcs_labels_numpy`。

### 5.1 道路分割中的 mask

道路分割的输入通常是一张彩色遥感图像，形状可以理解为：

```text
H x W x 3
```

其中：

*   `H` 表示图像高度；
*   `W` 表示图像宽度；
*   `3` 表示 RGB 三个颜色通道。

模型最终要预测的是一张黑白道路图，通常称为 **mask**：

```text
0 = 不是道路
1 = 是道路
```

例如一个 `5 x 5` 的道路 mask：

```text
0 0 0 0 0
0 1 1 1 0
0 0 0 1 0
0 0 0 1 0
0 0 0 0 0
```

普通 mask 只记录“哪里是道路”，PCS 进一步记录“道路像素之间怎么连接”。

### 5.2 `import numpy as np`

```python
import numpy as np
```

这句代码表示导入 NumPy 库，并给它起一个短名字 `np`。

NumPy 主要用于处理数组、矩阵和图像数据。例如：

```python
np.array(...)
np.zeros(...)
np.zeros_like(...)
```

在这个项目中，道路 mask 本质上就是一个二维 NumPy 数组。

### 5.3 函数定义语法

```python
def generate_pcs_labels_numpy(road_mask: np.ndarray, r: int = 2) -> np.ndarray:
```

可以翻译为：

> 定义一个函数，输入一张道路 mask 和一个距离参数 `r`，输出 PCS 连通性标签。

逐项解释：

*   `def`：定义函数；
*   `generate_pcs_labels_numpy`：函数名，表示“用 NumPy 生成 PCS 标签”；
*   `road_mask`：输入的道路二值图；
*   `road_mask: np.ndarray`：类型提示，表示建议传入 NumPy 数组；
*   `r: int = 2`：`r` 是整数，默认值为 2；
*   `-> np.ndarray`：类型提示，表示函数预计返回 NumPy 数组。

类型提示主要帮助人读代码，不会自动保证传入的数据一定正确。

### 5.4 读取图像高宽

```python
H, W = road_mask.shape
```

`shape` 表示数组形状。

例如：

```python
road_mask = np.array([
    [0, 0, 0],
    [1, 1, 0],
])
```

这张 mask 有 2 行、3 列，因此：

```python
road_mask.shape
```

结果是：

```text
(2, 3)
```

Python 支持拆包赋值：

```python
H, W = road_mask.shape
```

等价于：

```python
H = 2
W = 3
```

在图像处理中，二维图像形状通常写作 `(H, W)`，也就是先高度、后宽度。

### 5.5 初始化 PCS 标签

```python
pcs_label = np.zeros((8, H, W), dtype=np.uint8)
```

`np.zeros(...)` 用来创建一个全是 0 的数组。

这里的形状是：

```text
(8, H, W)
```

含义是创建 8 张 `H x W` 的图，每一张负责一个方向的连通性。

为什么是 8？因为每个像素周围有 8 个方向：

```text
左上  上  右上
左    当前 右
左下  下  右下
```

`dtype=np.uint8` 表示数组中的数字使用无符号 8 位整数保存。PCS 标签只需要保存 0 和 1，用 `uint8` 可以节省内存。

### 5.6 八个方向的偏移量

```python
directions = [
    (-r, -r), (-r, 0), (-r, r),
    (0, -r),           (0, r),
    (r, -r),  (r, 0),  (r, r)
]
```

每个方向写成一个二元组：

```python
(dy, dx)
```

其中：

*   `dy` 表示在纵向，也就是行方向移动多少；
*   `dx` 表示在横向，也就是列方向移动多少。

图像坐标中，`y` 向下增加，`x` 向右增加。

因此：

*   `(-r, 0)` 表示向上移动 `r` 格；
*   `(r, 0)` 表示向下移动 `r` 格；
*   `(0, -r)` 表示向左移动 `r` 格；
*   `(0, r)` 表示向右移动 `r` 格；
*   `(-r, r)` 表示向右上移动 `r` 格。

如果 `r = 2`，那么 `(-r, r)` 就是 `(-2, 2)`。

### 5.7 `for` 循环与 `enumerate`

```python
for idx, (dy, dx) in enumerate(directions):
```

这句表示：依次遍历 8 个方向，同时得到方向编号 `idx` 和方向偏移 `(dy, dx)`。

`enumerate` 会给列表中的每个元素配一个编号。

例如：

```python
directions = [(-1, 0), (1, 0)]
```

循环第一轮：

```python
idx = 0
dy = -1
dx = 0
```

循环第二轮：

```python
idx = 1
dy = 1
dx = 0
```

在 PCS 中，`idx` 用来决定当前结果写入 `pcs_label` 的第几个通道。

### 5.8 创建平移后的 mask

```python
shifted = np.zeros_like(road_mask)
```

`np.zeros_like(road_mask)` 表示创建一个和 `road_mask` 形状相同、内容全是 0 的数组。

如果 `road_mask` 是 `5 x 5`，那么 `shifted` 也是 `5 x 5`。

`shifted` 的作用是保存“平移后的道路 mask”。PCS 的核心思想就是：

1.  把原始道路 mask 按某个方向平移；
2.  将原始 mask 与平移后的 mask 对齐比较；
3.  如果两个位置同时为 1，说明这两个道路像素在该方向上连通。

核心代码是：

```python
pcs_label[idx] = road_mask & shifted
```

其中 `&` 表示按位与：只有左右两边都为 1，结果才是 1。

下一节重点学习第 35 到 48 行，也就是数组切片、边界处理和平移逻辑。

---

## 6. 零基础学习记录：数组切片、边界裁剪与 PCS 手算

### 6.1 为什么数组索引是 `[y, x]`

数学坐标常写作 `(x, y)`，但图像数组访问通常写作：

```python
road_mask[y, x]
```

原因是数组访问规则是：

```text
array[行, 列]
```

在图像中：

```text
行 = y = 上下方向
列 = x = 左右方向
```

所以：

```text
road_mask[y, x] = road_mask[第几行, 第几列]
```

如果写成 `road_mask[x, y]`，就会把行和列弄反，访问到另一个位置。

### 6.2 Python 切片规则

Python 切片写法是：

```python
a[start:end]
```

含义是：

```text
从 start 开始取，取到 end 前一个位置，不包含 end
```

例如：

```python
a[0:3]
```

取的是索引 `0, 1, 2`，不包含索引 `3`。

二维数组切片写作：

```python
road_mask[y_start:y_end, x_start:x_end]
```

也就是：

```text
road_mask[行范围, 列范围]
```

### 6.3 平移的本质

PCS 中的平移代码是：

```python
shifted[y_dst_start:y_dst_end, x_dst_start:x_dst_end] = \
    road_mask[y_src_start:y_src_end, x_src_start:x_src_end]
```

可以理解为：

```text
从原图 road_mask 取一块区域，放到 shifted 的另一个位置
```

其中：

*   `src` 是 source，表示“从哪里取”；
*   `dst` 是 destination，表示“放到哪里去”；
*   `shifted` 是平移后的道路 mask。

例如向右移动 1 格时，可以理解为：

```python
shifted[:, 1:W] = road_mask[:, 0:W-1]
```

也就是：

```text
原图左边 W-1 列 -> shifted 右边 W-1 列
```

### 6.4 `max` 和 `min` 的作用

代码中用下面这些语句自动处理边界：

```python
y_src_start = max(0, -dy)
y_src_end = H + min(0, -dy)
x_src_start = max(0, -dx)
x_src_end = W + min(0, -dx)

y_dst_start = max(0, dy)
y_dst_end = H + min(0, dy)
x_dst_start = max(0, dx)
x_dst_end = W + min(0, dx)
```

这些语句的作用是：

```text
让来源区域和目标区域大小一致，并且不越界
```

如果 `dy = 1`，表示向下移动 1 格：

```text
road_mask[0:H-1, ...] -> shifted[1:H, ...]
```

如果 `dy = -1`，表示向上移动 1 格：

```text
road_mask[1:H, ...] -> shifted[0:H-1, ...]
```

`dx` 的逻辑完全一样，只是方向从上下变成左右。

### 6.5 PCS 手算例子：向右连通

假设：

```text
r = 1
dy = 0
dx = 1
```

原始道路 mask：

```text
road_mask =
0 0 0
0 1 1
0 0 0
```

向右平移后：

```text
shifted =
0 0 0
0 0 1
0 0 0
```

做按位与：

```text
road_mask & shifted =
0 0 0
0 0 1
0 0 0
```

结果中的 `1` 表示：这个方向上存在相邻道路像素。

### 6.6 PCS 手算例子：向下连通

假设：

```text
r = 1
dy = 1
dx = 0
```

原始道路 mask：

```text
road_mask =
0 1 0
0 1 0
0 0 0
```

向下平移后：

```text
shifted =
0 0 0
0 1 0
0 1 0
```

做按位与：

```text
road_mask & shifted =
0 0 0
0 1 0
0 0 0
```

这个 `1` 表示：原图中第 0 行第 1 列和第 1 行第 1 列是上下相邻的道路像素。

---

## 7. 零基础学习记录：PyTorch 版本与反向映射

### 7.1 为什么有 NumPy 和 PyTorch 两个版本

PCS 标签生成有两个版本：

```python
generate_pcs_labels_numpy(...)
generate_pcs_labels_pytorch(...)
```

它们的算法逻辑基本相同，区别是数据类型不同：

```text
NumPy   -> np.ndarray
PyTorch -> torch.Tensor
```

NumPy 版本适合：

```text
理解算法、离线处理标签、写小例子验证逻辑
```

PyTorch 版本适合：

```text
训练模型、放进 Dataset、和 GPU 张量一起计算
```

### 7.2 `dim()` 和 `squeeze(0)`

PyTorch 版中有：

```python
if road_mask.dim() == 3:
    road_mask = road_mask.squeeze(0)
```

`dim()` 表示张量有几个维度。

如果道路 mask 是：

```text
H x W
```

它是二维。

如果道路 mask 是：

```text
1 x H x W
```

它是三维，其中最前面的 `1` 表示单通道。

`squeeze(0)` 表示如果第 0 个维度大小是 1，就删掉这个维度：

```text
1 x H x W -> H x W
```

### 7.3 `device`

PyTorch 版中还有：

```python
device = road_mask.device
pcs_label = torch.zeros((8, H, W), dtype=torch.uint8, device=device)
```

`device` 表示张量在 CPU 还是 GPU。

如果参与同一次计算的张量不在同一个设备上，PyTorch 通常会报错。所以新建 `pcs_label` 时要使用同一个 `device`。

### 7.4 什么是反向映射

前面学的是：

```text
road_mask -> pcs_label
```

反向映射做的是：

```text
pcs_pred -> seg_out
```

也就是根据 PCS 预测的连接关系，尽量还原出一张道路 mask。

核心思想是：

```text
如果 PCS 预测某个方向存在连接，就把连接的两个端点都标成道路。
```

### 7.5 `threshold` 和 `np.where`

反向映射中有：

```python
active_y, active_x = np.where(pcs_pred[idx] >= threshold)
```

`threshold` 是阈值，例如 `0.5`：

```text
预测值 >= 0.5 -> 认为连通成立
预测值 < 0.5  -> 认为连通不成立
```

`np.where(...)` 会找出所有满足条件的位置。

如果某个位置为 True，那么它的行坐标进入 `active_y`，列坐标进入 `active_x`。

### 7.6 为什么要设置当前点和邻居点

反向映射中有两步：

```python
seg_out[active_y, active_x] = 1

target_y = active_y + dy
target_x = active_x + dx
seg_out[target_y[valid], target_x[valid]] = 1
```

第一步把当前点设为道路。

第二步把对应方向上的邻居点也设为道路。

原因是 PCS 的含义是：

```text
当前点和某个方向上的邻居点连通
```

如果连接成立，那么两个端点都应该是道路。

### 7.7 `unsqueeze(0)` 和 batch

PyTorch 反向映射支持两种输入：

```text
8 x H x W
B x 8 x H x W
```

其中 `B` 是 batch size，表示一批图片的数量。

如果输入没有 batch 维，代码会执行：

```python
pcs_pred = pcs_pred.unsqueeze(0)
```

它会把：

```text
8 x H x W
```

变成：

```text
1 x 8 x H x W
```

这样后面的代码就可以统一按 batch 形式处理。

处理结束后，如果原来没有 batch 维，再用：

```python
seg_out = seg_out.squeeze(0)
```

把：

```text
1 x H x W
```

还原成：

```text
H x W
```

### 7.8 本阶段记住

```text
PCS 标签生成：road_mask -> pcs_label
PCS 反向映射：pcs_pred -> seg_out
NumPy 适合理解和预处理，PyTorch 适合训练。
dim() 查看维度，squeeze() 删除大小为 1 的维度，unsqueeze() 增加一个维度。
```
