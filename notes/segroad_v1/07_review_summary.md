# Seg-Road v1 复习速查

这份笔记用于复习第一篇 Seg-Road。它保留需要真正理解和能复述的内容，去掉临时过程细节；完整训练时间线见 `09_training_journey.md`。

## 1. 一句话理解

Seg-Road 是一个道路提取分割模型：

```text
Transformer/CNN encoder-decoder 做像素级道路分割
PCS 额外监督道路像素之间的 8 邻域连通关系
```

它不只问“这个像素是不是路”，还问“这个路像素和周围路像素是否连着”。

## 2. 完整数据流

```text
image + road_mask
  -> RoadDataset
  -> 生成 pcs_target
  -> SegRoad model
  -> seg_out + pcs_out
  -> segmentation loss + alpha * PCS loss
  -> backward + optimizer.step
  -> sigmoid + threshold
  -> road prediction
```

关键形状：

```text
image：      (B, 3, H, W)
mask：       (B, 1, H, W)
pcs_target： (B, 8, H, W)
seg_out：    (B, 1, H, W)，logits
pcs_out：    (B, 8, H, W)，logits
```

训练时输入 loss 的是 logits，不要先手动 sigmoid。推理和指标计算时才：

```python
prob = torch.sigmoid(logits)
pred = prob >= threshold
```

## 3. SRA、MixFFN、SRTBlock

### SRA

SRA 是 Spatial Reduction Attention。它保留全部 `Q`，但对 `K/V` 做空间缩减：

```text
Q：N 个位置
K/V：N / sr_ratio^2 个位置
attention：N x (N / sr_ratio^2)
```

作用：降低注意力计算量，同时保留较大的上下文感受野。

### MixFFN

MixFFN 给 Transformer 补局部空间信息：

```text
Linear 扩通道
-> 3x3 depthwise conv
-> GELU
-> Linear 压回通道
```

`depthwise conv` 用 `groups=hidden_channels`，每个通道单独做局部卷积。

### SRTBlock

```text
x = x + SRA(LayerNorm(x))
x = x + MixFFN(LayerNorm(x))
```

SRA 看远距离关系，MixFFN 看局部邻域，残差连接稳定训练。

## 4. PCS

PCS 是 Pixel Connectivity Structure。对每个道路像素，检查它和 8 个方向邻居是否都属于道路。

对于某个方向偏移 `(dy, dx)`：

```text
PCS[y, x, direction] = mask[y, x] AND mask[y + dy, x + dx]
```

如果当前像素和对应邻居都是道路，PCS 标签为 1；否则为 0。

PCS 的意义：

```text
普通 mask：监督像素类别
PCS：监督像素之间是否连通
```

道路任务尤其需要 PCS，因为道路的价值不只在像素覆盖，还在拓扑连通。

## 5. Loss

基础形式：

```text
total_loss = seg_loss + alpha * pcs_loss
```

当前复现实验使用：

```text
seg_loss = weighted_BCE + dice_weight * DiceLoss
pcs_loss = weighted_BCE
alpha = 0.2
```

几个概念要分清：

```text
pos_weight：提高道路正样本错误的代价，解决类别不平衡
Dice Loss：直接鼓励预测区域和真实道路区域重叠
alpha：控制 PCS loss 对总 loss 的贡献
threshold：只影响推理和指标，不影响 loss
```

## 6. 指标

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2PR / (P + R)
IoU       = TP / (TP + FP + FN)
```

解释：

```text
FP 多：背景被误判成道路，Precision 低，预测偏粗
FN 多：道路被漏掉，Recall 低，预测断裂或缺失
IoU：区域重叠综合指标
F1：Precision 和 Recall 的平衡指标
```

不能只看 Accuracy。DeepGlobe 道路像素约 4%，全预测背景也能得到约 96% Accuracy，但 IoU/F1 为 0。

## 7. 真实实验结论

数据：

```text
DeepGlobe labeled train split：6226 张有 mask 图像
本地 train：4980
本地 validation：1246
公开 valid/test：无 mask，不能本地算 IoU/F1
```

早期失败：

```text
普通 BCE：loss 下降，Accuracy 约 0.96，但 IoU/F1 为 0
原因：背景塌缩，模型全预测背景
```

关键诊断：

```text
阈值诊断：早期概率近似常数，调 threshold 不能修结构
单图过拟合：能学到道路，说明数据流和反向传播没坏
BatchNorm 问题：小 batch 下不稳定，Decoder 改为 GroupNorm
```

最终有效配置：

```text
GroupNorm decoder
seg_pos_weight = 10
pcs_pos_weight = 10
dice_weight = 1
learning_rate = 3e-4
batch_size = 4
```

## 8. 当前最好结果

epoch 8 baseline，默认阈值 `0.5`：

```text
IoU：0.3776
F1：0.5482
Precision：0.4128
Recall：0.8155
pred+：0.0795
target+：0.0402
```

完整 validation threshold sweep 后，最佳阈值是 `0.85`：

```text
IoU：0.4460
F1：0.6169
Precision：0.6033
Recall：0.6311
pred+：0.0421
target+：0.0402
```

低学习率短程续训到 epoch 12，使用 `threshold=0.85`：

```text
IoU：0.4779
F1：0.6468
Precision：0.6839
Recall：0.6135
pred+：0.0361
target+：0.0402
```

结论：短程续训有效，主要提升 Precision 和 IoU/F1；代价是 Recall 小幅下降，模型更保守。

## 9. PCS 消融

baseline + PCS：

```text
IoU：0.3776
F1：0.5482
Precision：0.4128
Recall：0.8155
```

no PCS (`alpha=0`)：

```text
IoU：0.3729
F1：0.5432
Precision：0.4128
Recall：0.7942
```

结论：

```text
PCS 对默认阈值下的像素指标提升不大
baseline Recall 略高
no-pcs 训练早期 pred+ 更高，说明更容易乱预测道路
PCS 的价值可能更多体现在训练稳定性和道路连通性
```

后续如果要更严谨，应做可视化或拓扑指标对比。

## 10. 为什么还没达论文指标

论文报告 SegRoadv1 在 DeepGlobe 上 IoU 约 67.20%，我们当前最好是 47.79%。

差距可能来自：

```text
实现是简化复现，不完全等同论文代码
训练策略不同：论文训练更长，可能有预训练和分阶段策略
数据划分不同：论文使用 4696 train / 1530 test
当前模型规模、增强策略、后处理都较简化
硬件和 batch size 不同
```

所以当前不是“论文数值达标复现”，而是“可解释 baseline 复现”。

## 11. 过渡到 SegRoadV2 的桥

Seg-Road v1 当前暴露的问题：

```text
道路预测偏粗，容易误检线状背景
狭窄道路和遮挡路段仍会漏检
SRA 采样位置固定，不够适应弯曲道路
普通卷积固定网格，不够贴合细长道路
PCS 能补连通监督，但主干特征提取仍不够强
```

SegRoadV2 的升级正好对应这些问题：

```text
SRA -> DSA：注意力采样位置可变
普通卷积 -> GroupDCN：局部卷积采样位置可变
普通 decoder -> 条带卷积：更贴合细长道路
PCS 保留：连通性仍是道路提取主线
```

复习口诀：

```text
SRA 看远方，MixFFN 看邻居；
seg 做分类，PCS 保连通；
BCE 解决像素，Dice 解决重叠；
Precision 管误检，Recall 管漏检；
threshold 只切概率，不改模型；
SegRoadV2 的关键词是 deformable。
```
