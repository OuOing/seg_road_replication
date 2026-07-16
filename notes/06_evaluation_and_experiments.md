# Seg-Road 学习笔记：评估、实验与数据划分

本篇汇总模型评估、论文实验和数据划分知识。详细的训练代码与术语仍保留在 `05_training_pipeline.md`，这里主要用于快速复习。

## 1. 评估闭环

```text
模型 logits
-> sigmoid 得到置信度
-> threshold 得到 0/1 mask
-> 和真实 mask 比较
-> 统计 TP、FP、FN、TN
-> 计算 Precision、Recall、F1、IoU
```

### 1.1 四种结果

```text
TP：预测道路，真实也是道路
FP：预测道路，真实是背景
FN：预测背景，真实是道路
TN：预测背景，真实也是背景
```

道路断裂主要增加 `FN`；把屋顶、阴影等误判为道路主要增加 `FP`。

### 1.2 常用指标

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
IoU       = TP / (TP + FP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)
```

直觉：

```text
Precision：不要乱报道路
Recall：不要漏掉道路
F1：Precision 和 Recall 都要兼顾
IoU：预测道路和真实道路的区域重叠程度
```

### 1.3 IoU 和 MIoU

```text
IoU_road：只计算道路类别
MIoU：对道路和背景等所有类别的 IoU 取平均
```

背景像素通常很多，因此看道路提取效果时不能只看 Accuracy 或 MIoU，还应单独关注 `IoU_road`、Recall 和 F1。

## 2. 阈值

`threshold=0.5` 是常用起点，不一定是最优值。

```text
降低阈值：更容易预测道路，Recall 可能提高，FP 可能增加
提高阈值：预测更谨慎，Precision 可能提高，FN 可能增加
```

更合适的做法是在验证集上测试多个阈值，选择 IoU 或 F1 较好的值，不能反复根据测试集调整。

## 3. Baseline、SOTA 与消融实验

### 3.1 Baseline

`Baseline` 是参考模型，用来回答新方法比基础方案提升了多少。

### 3.2 SOTA

`SOTA` 是 `State Of The Art`，表示特定数据集、指标和时间范围内的先进水平，不是永久称号。

### 3.3 Ablation Study

`Ablation Study` 是消融实验：保持其他条件不变，一次移除或替换一个模块。

针对 Seg-Road，可以比较：

```text
CNN baseline
CNN + Transformer
CNN + PCS
CNN + Transformer + PCS
```

这样才能分别分析 Transformer 的全局建模贡献和 PCS 的连通性监督贡献。

### 3.4 公平比较

需要尽量保持：

```text
相同数据集划分
相同输入尺寸
相同训练轮数
相同优化器和学习率策略
相同评价代码
相同阈值或阈值选择规则
```

还要同时观察参数量、FLOPs、FPS 和显存，而不是只看最高精度。

## 4. 数据集划分

### 4.1 三种数据集

```text
train：用于更新模型参数
validation：用于选择模型和超参数
test：用于最终客观评价
```

测试集不能反复用于调整学习率、阈值或网络结构，否则测试结果会失去客观性。

### 4.2 道路数据为什么容易泄漏

遥感道路数据经常先把一张大图裁成多个小块。如果先裁剪再随机划分，来自同一张大图的相邻区域可能分别进入训练集和验证集：

```text
同一城市大图
-> patch A 进入训练集
-> 相邻 patch B 进入验证集
```

由于两个 patch 的道路纹理、建筑、颜色和拍摄条件非常相似，模型可能已经间接见过验证区域，导致验证分数虚高。

这种问题叫 `Data Leakage`（数据泄漏）。

### 4.3 更合理的划分顺序

优先按照原始大图、城市或地理区域划分：

```text
先划分原始大图或区域
-> 再分别裁剪 train、validation、test
```

而不是：

```text
先把所有大图裁成 patch
-> 再把所有 patch 随机打乱划分
```

### 4.4 当前代码的局限

当前 `code/train.py` 使用文件级随机划分：

```text
全部图片/patch
-> 随机打乱
-> 按 val_ratio 划分
```

它适合最小训练验证，但正式复现实验应提供明确的训练集和验证集列表，或者根据原始大图/区域分组划分，避免相邻 patch 泄漏。

## 5. 当前项目实验状态

已经完成：

```text
Dataset 和 DataLoader
PCS target 生成
训练与验证循环
IoU、F1、Precision、Recall 计算
checkpoint 保存
单图推理脚本
合成数据 Smoke Test
```

尚未完成：

```text
下载并整理真实数据集
按原始区域进行无泄漏划分
正式训练 Seg-Road-s/m/l
复现论文指标
PCS 融合与消融实验
速度、参数量和 FLOPs 统计
```

## 6. 本篇记住

```text
训练集负责学习，验证集负责选择，测试集负责最终评价。
指标必须结合 TP、FP、FN 理解。
消融实验通过控制变量证明模块贡献。
道路 patch 应按原始大图或地理区域划分，避免数据泄漏。
```
