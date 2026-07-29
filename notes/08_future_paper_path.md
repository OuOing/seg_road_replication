# 后续论文学习路线

当前实践以 Seg-Road（2023）为起点。完成本轮 DeepGlobe 基线实验后，按下面的顺序继续学习和复现。

## 1. SegRoadV2：从固定算子到可变形算子

论文：SegRoadV2: A Hybrid Deformable Self-Attention and Convolutional Network for Road Extraction with Connectivity Structure（2025）

核心变化：

```text
Seg-Road 的 Transformer       -> Deformable Self-Attention (DSA)
普通局部卷积                 -> Groupable Deformable Convolution (GroupDCN)
普通 decoder                 -> 可重参数化条带卷积
PCS                          -> 继续保留，用于道路连通性监督
DeepGlobe 论文 IoU           -> 69.88%
```

学习重点：先理解 offset 如何让采样位置适应弯曲道路，再理解条带卷积为什么适合细长目标。这篇与当前代码关系最近，适合作为第一个升级实验。

建议实验：保持数据划分、损失和评估不变，只替换一个模块，分别验证 DSA、GroupDCN 和 strip convolution 的增益。

进入本篇前的 Seg-Road v1 最小收口条件：

```text
完成 threshold sweep，明确默认阈值与最佳阈值的差异
完成 PCS alpha=0 消融，知道连通性监督在当前实现中的贡献
保留统一 summary.md，作为 SegRoadV2 的 baseline 对照
```

阅读 SegRoadV2 时不要先追全部公式。优先问三个问题：

```text
DSA 比 SRA 多出来的 offset 解决了什么？
GroupDCN 比普通卷积多出来的可变形采样解决了什么？
条带卷积为什么适合道路这种细长目标？
```

## 2. DiffRoad：把分割改写成条件去噪

论文：DiffRoad: A Conditional Diffusion-Based Network for Accurate Road Extraction（2026）

核心变化：

```text
直接预测二值 mask            -> 对带噪 mask 做条件去噪
常规多尺度 encoder           -> DSA-FPN
常规 decoder                 -> PAGDecoder
扩散先验                     -> 作为粗粒度 attention mask 指导解码
推理                          -> 单步去噪
DeepGlobe 论文 IoU           -> 70.27%
```

学习重点：理解前向加噪、反向去噪、条件信息，以及为什么 mask 的结构先验有助于补全被遮挡或断裂的道路。除了 IoU/F1，还要学习 APLS 和 TOPO-F1。

建议实验：先在现有模型输出上实现 mask 加噪与去噪小实验，再讨论完整 DSA-FPN 和 PAGDecoder，避免一开始同时引入过多变量。

## 3. FDMamba：频域、状态空间模型与图级拓扑

论文：FDMamba: Frequency-Enhanced Deformable Mamba for Topology-Aware Road Extraction（2026）

核心变化：

```text
Transformer 全局建模         -> Mamba / State Space Model 线性扫描
固定二维扫描路径             -> Deformable Mamba 自适应扫描
纯空间域特征                 -> FreqMamba 频域螺旋扫描
普通 decoder attention       -> 基于 DCT 的 Frequency Attention
像素分割评估                 -> 同时关注 APLS 等图级拓扑指标
```

论文摘要报告：DeepGlobe IoU 比 SegFormer 高 3.63 个百分点，作为拓扑提取器的 backbone 时 APLS 最多提高 2.81 个百分点。

学习重点：理解 SSM/Mamba 的选择性扫描、FFT/DCT 中低频和高频分别代表什么，以及为什么道路骨架可先从低频全局结构中学习，再补充高频边缘细节。

建议实验：先实现频域可视化和 DCT attention，再学习 Mamba。该方向依赖最多，应放在 SegRoadV2 和 DiffRoad 之后。

## 统一实验原则

三条方向都必须使用同一套 DeepGlobe split，报告以下指标：

```text
IoU / F1 / Precision / Recall
predicted_positive_ratio
参数量、训练时间、推理时间
APLS / TOPO-F1（进入拓扑实验后）
```

每次只引入一个主要变量，并与当前 Seg-Road baseline 对照。论文中的结果使用各自的数据划分、预训练和训练策略，不能直接与当前试验数值等同。
