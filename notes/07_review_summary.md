# Seg-Road 总复习速查

## 1. 一句话理解

Seg-Road 用 Encoder-Decoder 做道路分割，同时用 PCS（Pixel Connectivity Structure）监督道路像素之间的连通关系，减少道路断裂。

## 2. 完整数据流

```text
image + road_mask
    -> Dataset
    -> PCS target（8 个方向）
    -> SRT Encoder
    -> 多尺度 Decoder
    -> seg_out + pcs_out（logits）
    -> BCE segmentation loss + PCS loss
    -> backward + AdamW
    -> sigmoid + threshold
    -> road mask
```

## 3. 模型结构

### SRA

SRA（Spatial Reduction Attention）保留全部 `Q`，对 `K`、`V` 做空间缩减：

```text
Q：N 个位置
K/V：N / sr_ratio² 个位置
```

这样注意力矩阵从 `N x N` 变为 `N x (N / r²)`，降低计算量。SRA 本身不改变输出 token 数量。

### MixFFN

```text
Linear（扩展通道）
-> 3x3 Depthwise Conv
-> GELU
-> Linear（压回通道）
```

Depthwise Conv 用 `groups=hidden_features`，为 Transformer 补充局部空间和位置信息。

### SRTBlock

```text
x = x + SRA(LayerNorm(x))
x = x + MixFFN(LayerNorm(x))
```

SRA 看远距离关系，MixFFN 看局部邻域，残差连接保留原始信息并稳定训练。

### Encoder-Decoder

Patch Embedding 负责真正下采样；Encoder 越往后分辨率越低、语义越强；Decoder 通过上采样和跳跃连接恢复边界细节。

## 4. PCS 和损失

对方向偏移 `(dy, dx)`：

```text
PCS[y, x] = mask[y, x] AND mask[y + dy, x + dx]
```

只有当前像素和邻居都是道路时，标签才为 1。默认有 8 个方向，因此 `pcs_target` 和 `pcs_out` 通常是 8 通道。

模型输出是 logits：

```text
total_loss = BCEWithLogitsLoss(seg_out, seg_target)
           + alpha * BCEWithLogitsLoss(pcs_out, pcs_target)
```

训练时不要先手动 sigmoid；推理和计算指标时才执行：

```python
prob = torch.sigmoid(logits)
pred = prob > threshold
```

## 5. 训练、验证和推理

训练一个 batch：

```text
zero_grad -> forward -> loss -> backward -> optimizer.step
```

验证阶段使用 `model.eval()` 和 `torch.no_grad()`，不反向传播、不更新参数。

`best.pt` 保存验证集表现最好的模型，`last.pt` 保存最后一轮状态。恢复训练需要同时加载模型、优化器和 epoch；只推理则只需模型参数。

## 6. 指标和实验

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2PR / (P + R)
IoU       = TP / (TP + FP + FN)
```

道路断裂通常增加 FN；屋顶、阴影误检通常增加 FP。不能只看 Accuracy，因为背景像素通常远多于道路像素。

消融实验应一次只改变一个模块，例如：

```text
CNN baseline
CNN + Transformer
CNN + PCS
CNN + Transformer + PCS
```

## 7. 数据划分原则

正确顺序：

```text
先按原始大图、城市或地理区域划分 train/val/test
再在各集合内部裁剪 patch
```

不能先裁 patch 再随机划分，否则相邻 patch 可能跨集合造成数据泄漏。

## 8. 当前进度和下一步

已完成：理论主线、SRT 代码级理解、最小训练闭环、Smoke Test、指标和实验设计。

待完成：真实数据整理、无泄漏划分、正式训练、测试集评估、PCS 消融、参数量/FLOPs/速度统计。SegRoadv2 仍处于归档排队状态，尚未开始精读。

## 9. 记忆口诀

```text
SRA 看远方，MixFFN 看邻居；
seg 做分类，PCS 保连通；
BCE 用来学，IoU/F1 用来评；
训练看 train，选择看 val，最终看 test。

## 10. 真实数据检查清单

DeepGlobe 或 Massachusetts 下载后，先保留原始目录，再整理为：

```text
data/raw/<dataset>/        # 原始下载文件
data/<dataset>/images/     # RGB 图像
data/<dataset>/masks/      # 道路 mask
```

当前 `RoadDataset` 按文件名 stem 配对，例如：

```text
images/abc_001.jpg
masks/abc_001.png
```

开始训练前依次检查：

```text
image 和 mask 数量
image/mask stem 是否一一对应
原始尺寸是否一致
mask 唯一值和编码（0/1 或 0/255）
道路像素占比
image、mask 半透明叠加后是否对齐
Dataset 输出形状和数值范围
```

Dataset 输出应为：

```text
image：      (3, 512, 512)，float32，范围 0~1
mask：       (1, 512, 512)，float32，值为 0/1
pcs_target： (8, 512, 512)，float32，值为 0/1
```

如果原始文件名是 `abc_001_sat` 和 `abc_001_mask`，需要统一 stem 后再使用当前配对函数，或专门修改配对逻辑。

正式实验不要只按 patch 随机划分；应先按原始大图、城市或区域划分 train/val/test，再分别裁剪 patch。
```
