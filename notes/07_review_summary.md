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

### 当前 DeepGlobe 数据状态

当前下载的公开镜像包含：

```text
train：6226 张图像和 6226 张 mask
valid：1243 张图像，无 mask
test：1101 张图像，无 mask
```

因此当前监督实验固定使用有标注的 train 数据划分：

```text
train.txt：4980
val.txt：1246
test.txt：空（无公开 mask，不能计算 IoU/F1）
```

无标注的 valid/test 图像可以用于最终预测或生成提交文件，但不能作为本地监督评估集。正式报告应明确说明这一点。

## 11. AI 时代的代码学习重点

不需要逐行背诵所有代码，但必须掌握关键代码的意图、输入输出、失败风险和验证方法。

### 必须深入理解

```text
code/pcs.py：PCS 标签、方向偏移、边界处理
code/srt.py：Q/K/V 形状、SRA 缩减、MixFFN 二维变换
code/train.py：数据、损失、梯度、指标和 checkpoint 闭环
```

这些地方即使代码能运行，写错后也可能产生可信但错误的实验结果。

### 需要理解流程

```text
Dataset、DataLoader、推理、checkpoint、命令行参数
```

目标是能追踪数据从哪里进入、参数在哪里更新、指标在哪里计算，以及模型在哪里保存。

### 可以交给 AI 生成

```text
文件遍历、argparse 样板、日志格式、重复性测试脚手架
```

但必须检查生成代码的形状、边界条件、数据划分和测试结果。

### 代码审查问题

```text
输入和输出形状是什么？
这一层为什么存在？
如果写错会出现什么现象？
测试是否真的覆盖了关键行为？
```

学习目标不是默写 Seg-Road，而是能解释关键数据流、发现 AI 生成代码的问题，并设计实验验证它。

## 12. 训练设备

当前 Mac 是 Apple M4 Pro（48GB）。项目支持：

```text
--device auto：依次选择 CUDA、MPS、CPU
--device mps：强制使用 Apple GPU
--device cuda：强制使用 NVIDIA GPU
--device cpu：强制使用 CPU
```

Apple 芯片本机训练应优先使用 MPS。若当前运行环境无法访问 MPS，可在本机 Terminal 验证，或使用带 NVIDIA GPU 的云环境完成正式复现。

## 13. 真实训练与类别不平衡

本节保留训练结论速查；从第一次普通 BCE 背景塌缩到 epoch 8 训练结束的完整时间线、诊断证据和实验决策，见 `09_training_journey.md`。

DeepGlobe 首轮正式训练使用：

```text
设备：Apple MPS
模型：Seg-Road-s
train：4980
validation：1246
```

前两轮出现：loss 下降，但道路 IoU/F1 接近 0。这通常表示模型在训练初期偏向预测背景。抽样统计中，道路像素约占 4.25%，背景约占 95.75%。

加权 BCE 使用 `pos_weight` 提高道路正样本的损失权重：

```text
Weighted BCE = -[pos_weight * y * log(p) + (1-y) * log(1-p)]
```

可用 `背景像素数 / 道路像素数` 估算权重，当前约为 22.5；实际实验可先尝试 5、10、15，避免权重过大导致大量误检。

`pos_weight` 与 `alpha` 不同：

```text
pos_weight：平衡道路和背景像素
alpha：平衡 segmentation loss 和 PCS loss
```

判断规则：先观察前 5 个 epoch。如果 IoU/F1 开始上升，继续普通 BCE；如果持续为 0，再停止并开展加权 BCE 对照实验。

普通 BCE 实验最终完成 20 个 epoch：

```text
val loss：0.1956
accuracy：0.9598
IoU / F1 / Precision / Recall：0
```

这确认模型发生背景塌缩：约 96% 的背景使 Accuracy 看起来很高，但预测道路比例为 0。下一轮使用独立输出目录和加权 BCE：

```text
seg_pos_weight = 10
pcs_pos_weight = 10
```

每轮同时记录 Precision、Recall 和 `predicted_positive_ratio`，观察是否从全背景恢复，以及是否因权重过大产生道路误检。

### 训练诊断结果

依次完成了以下诊断：

```text
普通 BCE，lr=1e-4：20 epoch 全背景
加权 BCE(10)，lr=1e-4：5 epoch 全背景
加权 BCE(10)，lr=1e-3：短暂预测道路后退回背景
单图 100 步过拟合：IoU 约 0.38，证明模型和反向传播能学习
```

概率分析发现模型曾输出接近常数的 `0.316~0.343`，因此不是单纯把 threshold 从 0.5 调低就能解决。

### Dice 与小 Batch 归一化

加入可选 Dice Loss：

```text
seg_loss = weighted_BCE + dice_weight * DiceLoss
total_loss = seg_loss + alpha * PCS_BCE
```

Decoder 原先使用 BatchNorm。当前 batch size 只有 2~4，训练与验证之间的统计量不稳定，验证结果会在全背景和全道路之间跳变。因此改为不依赖 batch 统计的 GroupNorm，并通过小数据对照实验验证稳定性。

### 权重选择实验

在固定 200 train / 50 validation 上，只改变正样本权重：

```text
pos_weight=22.5：best IoU 0.1061，pred+ 约 0.30，误报较多
pos_weight=10：  best IoU 0.1219，pred+ 约 0.15，综合最好
pos_weight=5：   best IoU 0.1042，容易退回背景
```

当前选定的全量实验配置：

```text
GroupNorm
seg_pos_weight = 10
pcs_pos_weight = 10
dice_weight = 1
learning_rate = 3e-4
batch_size = 4
num_workers = 2
```

训练脚本现已支持 `--resume`、batch 进度和每轮耗时。训练产物继续保存在被 Git 忽略的 `runs/` 下。

### 全量八轮试跑

使用 4980 train / 1246 validation、512 输入和上述候选配置完成 8 个 epoch：

```text
epoch 1：val IoU 0.1181，F1 0.2113，P 0.1248，R 0.6861，pred+ 0.2211
epoch 2：val IoU 0.2260，F1 0.3687，P 0.3158，R 0.4428，pred+ 0.0564
epoch 3：val IoU 0.2546，F1 0.4059，P 0.2697，R 0.8200，pred+ 0.1224
epoch 4：val IoU 0.3387，F1 0.5060，P 0.3778，R 0.7658，pred+ 0.0816
epoch 5：val IoU 0.2662，F1 0.4205，P 0.2781，R 0.8616，pred+ 0.1247
epoch 6：val IoU 0.3231，F1 0.4884，P 0.3441，R 0.8408，pred+ 0.0983
epoch 7：val IoU 0.3738，F1 0.5442，P 0.4084，R 0.8151，pred+ 0.0803
epoch 8：val IoU 0.3776，F1 0.5482，P 0.4128，R 0.8155，pred+ 0.0795
真实道路像素比例 target+：0.0402
```

结论：模型已经明确摆脱全背景塌缩。IoU 总体上升，但 epoch 5 出现明显回落，表明固定学习率和较强正样本权重下仍有震荡。epoch 7~8 的指标接近，当前开始进入平台期。预测道路比例仍约为真实比例的两倍，因此主要问题是误检，而不是道路完全漏掉。

当前最佳检查点为：

```text
runs/deepglobe/segroad-s-full-probe/best.pt
epoch = 8
val IoU = 0.3776
```

### 预测可视化结论

使用 epoch 8 最佳检查点抽取 6 个固定验证样本，生成原图、真实 mask、预测 mask 和错误叠加图：

```text
绿色：TP，正确道路
红色：FP，误检道路
蓝色：FN，漏检道路
```

模型已经能识别主干道路和部分交叉口，但预测道路通常偏粗。误检主要来自施工带、田埂、河岸等细长纹理；漏检主要是狭窄支路和被遮挡路段。这与高 Recall、较低 Precision 的指标一致。

可视化保存在：

```text
runs/deepglobe/segroad-s-full-probe/visualizations/comparison_sheet.png
```

下一阶段不应直接按 `3e-4` 再跑很多轮。更合理的是保留 epoch 8 最佳权重，加入学习率衰减后继续短跑，并对 `pos_weight=5~10` 或推理阈值进行验证集对照。
